"""
Unit tests for the DistributedOrchestrator class.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import mlx.core as mx

from src.coordinator.orchestrator import (
    DistributedOrchestrator, 
    InferenceRequest, 
    InferenceResponse
)
from src.core.config import DeviceRole


class TestDistributedOrchestrator:
    """Test cases for DistributedOrchestrator."""
    
    def test_init_with_coordinator_role(self, mock_cluster_config):
        """Test orchestrator initialization with coordinator role."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        assert orchestrator.config == mock_cluster_config
        assert orchestrator.device_config.role == DeviceRole.COORDINATOR
        assert orchestrator.device_config.device_id == "coordinator"
        assert not orchestrator.is_initialized
        assert orchestrator.model is None
        assert orchestrator.tokenizer is None
        assert orchestrator.layer_processor is None
        assert orchestrator.connection_pool is None
    
    def test_init_with_non_coordinator_role(self, mock_cluster_config):
        """Test orchestrator initialization fails with non-coordinator role."""
        # Mock get_local_device to return worker device
        worker_device = mock_cluster_config.devices[1]  # worker_1
        with patch.object(mock_cluster_config, 'get_local_device', return_value=worker_device):
            with pytest.raises(ValueError, match="Orchestrator must run on coordinator device"):
                DistributedOrchestrator(mock_cluster_config)
    
    def test_init_with_no_local_device(self, mock_cluster_config):
        """Test orchestrator initialization fails when no local device found."""
        with patch.object(mock_cluster_config, 'get_local_device', return_value=None):
            with pytest.raises(ValueError, match="Orchestrator must run on coordinator device"):
                DistributedOrchestrator(mock_cluster_config)
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_cluster_config, mock_mlx_model, mock_tokenizer):
        """Test successful orchestrator initialization."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock the model loading
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_full_model.return_value = (mock_mlx_model, mock_tokenizer)
            
            # Mock LayerProcessor
            with patch('src.coordinator.orchestrator.LayerProcessor') as mock_layer_processor:
                # Mock ConnectionPool
                with patch('src.coordinator.orchestrator.ConnectionPool') as mock_pool_class:
                    mock_pool = mock_pool_class.return_value
                    
                    # Mock worker verification
                    mock_workers = mock_cluster_config.get_workers()
                    mock_client = MagicMock()
                    mock_client.health_check.return_value = {'healthy': True}
                    mock_pool.get_client.return_value = mock_client
                    
                    await orchestrator.initialize()
                    
                    assert orchestrator.is_initialized
                    assert orchestrator.model == mock_mlx_model
                    assert orchestrator.tokenizer == mock_tokenizer
                    assert orchestrator.layer_processor is not None
                    assert orchestrator.connection_pool is not None
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_cluster_config):
        """Test that initialize does nothing if already initialized."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        
        # Should return immediately without doing anything
        await orchestrator.initialize()
        
        assert orchestrator.model is None  # Should not have been set
    
    @pytest.mark.asyncio
    async def test_initialize_model_loading_failure(self, mock_cluster_config):
        """Test initialization failure during model loading."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_full_model.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception, match="Model loading failed"):
                await orchestrator.initialize()
            
            assert not orchestrator.is_initialized
    
    @pytest.mark.asyncio
    async def test_verify_workers_success(self, mock_cluster_config):
        """Test successful worker verification."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock connection pool
        mock_pool = MagicMock()
        mock_client = MagicMock()
        mock_client.health_check.return_value = {'healthy': True}
        mock_pool.get_client.return_value = mock_client
        orchestrator.connection_pool = mock_pool
        
        # Should not raise exception
        await orchestrator._verify_workers()
        
        # Verify health check was called for each worker
        workers = mock_cluster_config.get_workers()
        assert mock_pool.get_client.call_count == len(workers)
    
    @pytest.mark.asyncio
    async def test_verify_workers_unhealthy_worker(self, mock_cluster_config):
        """Test worker verification failure with unhealthy worker."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock connection pool
        mock_pool = MagicMock()
        mock_client = MagicMock()
        mock_client.health_check.return_value = {'healthy': False}
        mock_pool.get_client.return_value = mock_client
        orchestrator.connection_pool = mock_pool
        
        with pytest.raises(RuntimeError, match="Worker .* is not healthy"):
            await orchestrator._verify_workers()
    
    @pytest.mark.asyncio
    async def test_verify_workers_no_connection(self, mock_cluster_config):
        """Test worker verification failure with no connection."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock connection pool
        mock_pool = MagicMock()
        mock_pool.get_client.return_value = None  # No connection
        orchestrator.connection_pool = mock_pool
        
        with pytest.raises(RuntimeError, match="No connection to worker"):
            await orchestrator._verify_workers()
    
    @pytest.mark.asyncio
    async def test_process_request_not_initialized(self, mock_cluster_config, test_inference_request):
        """Test processing request when not initialized triggers initialization."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock initialization
        orchestrator.initialize = AsyncMock()
        
        # Mock the rest of the processing pipeline
        with patch.object(orchestrator, '_format_messages', return_value="test prompt"):
            with patch.object(orchestrator, '_distributed_forward', 
                            new_callable=AsyncMock, return_value=mx.ones((1, 10, 512))):
                with patch.object(orchestrator, '_generate_response', 
                                new_callable=AsyncMock, return_value="test response"):
                    orchestrator.tokenizer = mock_tokenizer()
                    
                    await orchestrator.process_request(test_inference_request)
                    
                    orchestrator.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_success(self, mock_cluster_config, test_inference_request, 
                                         mock_mlx_model, mock_tokenizer):
        """Test successful request processing."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock the processing pipeline
        with patch.object(orchestrator, '_distributed_forward', 
                         new_callable=AsyncMock, return_value=mx.ones((1, 10, 512))):
            with patch.object(orchestrator, '_generate_response', 
                            new_callable=AsyncMock, return_value="test response"):
                
                response = await orchestrator.process_request(test_inference_request)
                
                assert isinstance(response, InferenceResponse)
                assert response.request_id == test_inference_request.request_id
                assert response.content == "test response"
                assert response.tokens_generated > 0
                assert response.total_time_ms > 0
                assert isinstance(response.device_times, dict)
    
    @pytest.mark.asyncio
    async def test_distributed_forward_coordinator_only(self, mock_cluster_config, 
                                                       sample_input_tensor):
        """Test distributed forward pass with coordinator only."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool with no next device
        mock_pool = MagicMock()
        mock_pool.get_next_device_client.return_value = None
        orchestrator.connection_pool = mock_pool
        
        device_times = {}
        result = await orchestrator._distributed_forward(
            sample_input_tensor, "test_request", device_times
        )
        
        assert result.shape == (1, 5, 512)
        assert 'embedding' in device_times
        assert orchestrator.device_config.device_id in device_times
        mock_processor.process_embedding.assert_called_once_with(sample_input_tensor)
    
    @pytest.mark.asyncio
    async def test_distributed_forward_with_workers(self, mock_cluster_config, 
                                                   sample_input_tensor):
        """Test distributed forward pass with workers."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool and worker client
        mock_pool = MagicMock()
        mock_client = MagicMock()
        
        # Mock processing result from worker
        from src.communication.grpc_client import ProcessingResult
        mock_result = ProcessingResult(
            output_tensor=mx.ones((1, 5, 512)),
            processing_time_ms=100.0,
            device_id="worker_1"
        )
        mock_client.process_layers.return_value = mock_result
        
        # Set up mock to return client once, then None
        mock_pool.get_next_device_client.side_effect = [mock_client, None]
        orchestrator.connection_pool = mock_pool
        
        device_times = {}
        result = await orchestrator._distributed_forward(
            sample_input_tensor, "test_request", device_times
        )
        
        assert result.shape == (1, 5, 512)
        assert 'embedding' in device_times
        assert 'worker_1' in device_times
        assert device_times['worker_1'] == 100.0
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_cluster_config, sample_hidden_states, 
                                   mock_tokenizer):
        """Test text generation from hidden states."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor for output processing
        mock_processor = MagicMock()
        mock_logits = mx.ones((1, 1, 32000))  # vocab size
        mock_processor.process_output.return_value = mock_logits
        orchestrator.layer_processor = mock_processor
        
        # Mock sampler
        with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
            mock_sampler = MagicMock()
            mock_sampler.return_value = mx.array([[2]])  # EOS token
            mock_make_sampler.return_value = mock_sampler
            
            # Mock distributed forward for generation steps
            with patch.object(orchestrator, '_distributed_forward', 
                            new_callable=AsyncMock, return_value=sample_hidden_states):
                
                result = await orchestrator._generate_response(
                    sample_hidden_states, max_tokens=10, temperature=0.7, 
                    top_p=0.9, repetition_penalty=1.1
                )
                
                assert result == "mocked response"
                mock_tokenizer.decode.assert_called_once()
    
    def test_format_messages_single_message(self, mock_cluster_config):
        """Test message formatting with single message."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = orchestrator._format_messages(messages)
        
        assert result == "user: Hello\nassistant: "
    
    def test_format_messages_conversation(self, mock_cluster_config):
        """Test message formatting with conversation."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = orchestrator._format_messages(messages)
        
        expected = "user: Hello\nassistant: Hi there!\nuser: How are you?\nassistant: "
        assert result == expected
    
    def test_get_last_device_id(self, mock_cluster_config):
        """Test getting the last device ID in the pipeline."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # The mock config has worker_2 with layers [6, 7, 8], which should be last
        last_device_id = orchestrator._get_last_device_id()
        assert last_device_id == "worker_2"
    
    def test_get_last_device_id_empty_layers(self, mock_cluster_config):
        """Test getting last device ID when no layers assigned."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Clear layer distributions
        mock_cluster_config.model.layer_distribution = {}
        
        last_device_id = orchestrator._get_last_device_id()
        assert last_device_id is None


class TestInferenceRequest:
    """Test cases for InferenceRequest dataclass."""
    
    def test_inference_request_creation(self):
        """Test creating an inference request."""
        request = InferenceRequest(
            request_id="test_123",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        assert request.request_id == "test_123"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.repetition_penalty == 1.2
    
    def test_inference_request_defaults(self):
        """Test inference request with default values."""
        request = InferenceRequest(
            request_id="test_123",
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 1.0
        assert request.repetition_penalty == 1.1


class TestInferenceResponse:
    """Test cases for InferenceResponse dataclass."""
    
    def test_inference_response_creation(self):
        """Test creating an inference response."""
        device_times = {"coordinator": 100.0, "worker_1": 150.0}
        
        response = InferenceResponse(
            request_id="test_123",
            content="Generated text",
            tokens_generated=25,
            total_time_ms=300.5,
            device_times=device_times
        )
        
        assert response.request_id == "test_123"
        assert response.content == "Generated text"
        assert response.tokens_generated == 25
        assert response.total_time_ms == 300.5
        assert response.device_times == device_times
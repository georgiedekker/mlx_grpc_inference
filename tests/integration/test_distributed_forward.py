"""
Integration tests for distributed forward pass through the pipeline.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import mlx.core as mx

from src.coordinator.orchestrator import DistributedOrchestrator, InferenceRequest
from src.model.inference import LayerProcessor
from src.communication.grpc_client import ConnectionPool, ProcessingResult, GRPCInferenceClient


class TestDistributedForwardIntegration:
    """Integration tests for distributed forward pass."""
    
    @pytest.mark.asyncio
    async def test_full_distributed_forward_pipeline(self, mock_cluster_config, 
                                                   mock_mlx_model, mock_tokenizer):
        """Test complete distributed forward pass through all devices."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool and workers
        mock_pool = MagicMock()
        
        # Create mock clients for worker_1 and worker_2
        worker1_client = MagicMock()
        worker2_client = MagicMock()
        
        worker1_result = ProcessingResult(
            output_tensor=mx.ones((1, 5, 512)),
            processing_time_ms=100.0,
            device_id="worker_1"
        )
        
        worker2_result = ProcessingResult(
            output_tensor=mx.ones((1, 5, 512)),
            processing_time_ms=120.0,
            device_id="worker_2"
        )
        
        worker1_client.process_layers.return_value = worker1_result
        worker2_client.process_layers.return_value = worker2_result
        
        # Set up connection pool to return clients in sequence
        mock_pool.get_next_device_client.side_effect = [worker1_client, worker2_client, None]
        orchestrator.connection_pool = mock_pool
        
        # Execute distributed forward pass
        input_tensor = mx.array([[1, 2, 3, 4, 5]])
        device_times = {}
        
        result = await orchestrator._distributed_forward(
            input_tensor, "test_request", device_times
        )
        
        # Verify results
        assert result.shape == (1, 5, 512)
        assert 'embedding' in device_times
        assert 'coordinator' in device_times
        assert 'worker_1' in device_times
        assert 'worker_2' in device_times
        assert device_times['worker_1'] == 100.0
        assert device_times['worker_2'] == 120.0
        
        # Verify layer processing calls
        mock_processor.process_embedding.assert_called_once_with(input_tensor)
        assert worker1_client.process_layers.call_count == 1
        assert worker2_client.process_layers.call_count == 1
    
    @pytest.mark.asyncio
    async def test_coordinator_only_forward(self, mock_cluster_config, mock_mlx_model, mock_tokenizer):
        """Test forward pass when only coordinator has layers."""
        # Modify config so only coordinator has layers
        mock_cluster_config.model.layer_distribution = {
            "coordinator": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "worker_1": [],
            "worker_2": []
        }
        
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool with no workers
        mock_pool = MagicMock()
        mock_pool.get_next_device_client.return_value = None
        orchestrator.connection_pool = mock_pool
        
        input_tensor = mx.array([[1, 2, 3, 4, 5]])
        device_times = {}
        
        result = await orchestrator._distributed_forward(
            input_tensor, "test_request", device_times
        )
        
        # Verify only coordinator processing occurred
        assert result.shape == (1, 5, 512)
        assert 'embedding' in device_times
        assert 'coordinator' in device_times
        assert len(device_times) == 2  # Only embedding and coordinator
        
        mock_processor.process_embedding.assert_called_once()
        mock_processor.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_failure_during_forward(self, mock_cluster_config, 
                                                mock_mlx_model, mock_tokenizer):
        """Test handling worker failure during forward pass."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool
        mock_pool = MagicMock()
        
        # Create mock client that fails
        failing_client = MagicMock()
        failing_client.process_layers.side_effect = Exception("Worker failed")
        
        mock_pool.get_next_device_client.return_value = failing_client
        orchestrator.connection_pool = mock_pool
        
        input_tensor = mx.array([[1, 2, 3, 4, 5]])
        device_times = {}
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Worker failed"):
            await orchestrator._distributed_forward(
                input_tensor, "test_request", device_times
            )
    
    @pytest.mark.asyncio
    async def test_layer_distribution_validation(self, mock_cluster_config, 
                                                mock_mlx_model, mock_tokenizer):
        """Test that layer distribution is properly validated."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor
        mock_processor = MagicMock()
        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
        mock_processor.process.return_value = mx.ones((1, 5, 512))
        orchestrator.layer_processor = mock_processor
        
        # Mock connection pool
        mock_pool = MagicMock()
        
        # Mock worker with empty layer assignment
        worker_client = MagicMock()
        mock_pool.get_next_device_client.return_value = worker_client
        
        # Modify config to have worker with no layers
        mock_cluster_config.model.layer_distribution["worker_1"] = []
        
        input_tensor = mx.array([[1, 2, 3, 4, 5]])
        device_times = {}
        
        # Should handle empty layer assignment gracefully
        result = await orchestrator._distributed_forward(
            input_tensor, "test_request", device_times
        )
        
        # Should still return a result without calling the worker
        assert result.shape == (1, 5, 512)
        worker_client.process_layers.assert_not_called()


class TestConnectionPoolIntegration:
    """Integration tests for ConnectionPool with real device configurations."""
    
    def test_connection_pool_with_multiple_devices(self, mock_cluster_config):
        """Test connection pool manages multiple device connections."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            # Create different clients for different devices
            clients = {}
            
            def create_client(device, timeout):
                client = MagicMock()
                client.target_device = device
                clients[device.device_id] = client
                return client
            
            mock_client_class.side_effect = create_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Verify connections were created for all remote devices
            expected_devices = ["worker_1", "worker_2"]
            assert len(pool.clients) == len(expected_devices)
            
            for device_id in expected_devices:
                assert device_id in pool.clients
                assert pool.clients[device_id] == clients[device_id]
    
    def test_connection_pool_device_ordering(self, mock_cluster_config):
        """Test that connection pool respects device rank ordering."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            mock_client_class.return_value = MagicMock()
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Test getting next device clients in order
            next_client_1 = pool.get_next_device_client("coordinator")  # rank 0 -> rank 1
            next_client_2 = pool.get_next_device_client("worker_1")     # rank 1 -> rank 2
            next_client_3 = pool.get_next_device_client("worker_2")     # rank 2 -> None
            
            assert next_client_1 is not None  # Should get worker_1
            assert next_client_2 is not None  # Should get worker_2
            assert next_client_3 is None      # No more devices
    
    def test_connection_pool_health_monitoring(self, mock_cluster_config):
        """Test connection pool can monitor device health."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            # Create mock clients with different health states
            healthy_client = MagicMock()
            healthy_client.health_check.return_value = {'healthy': True, 'device_id': 'worker_1'}
            
            unhealthy_client = MagicMock()
            unhealthy_client.health_check.return_value = {'healthy': False, 'device_id': 'worker_2'}
            
            def create_client(device, timeout):
                if device.device_id == "worker_1":
                    return healthy_client
                else:
                    return unhealthy_client
            
            mock_client_class.side_effect = create_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Check health of all devices
            worker1_health = pool.get_client("worker_1").health_check()
            worker2_health = pool.get_client("worker_2").health_check()
            
            assert worker1_health['healthy'] is True
            assert worker2_health['healthy'] is False


class TestLayerProcessorIntegration:
    """Integration tests for LayerProcessor with realistic model structures."""
    
    def test_layer_processor_with_model_layers(self, mock_mlx_model):
        """Test layer processor with actual model layer structure."""
        assigned_layers = [3, 4, 5]
        processor = LayerProcessor(mock_mlx_model, "worker_1", assigned_layers)
        
        # Create realistic input
        input_tensor = mx.ones((1, 10, 512))  # [batch, seq_len, hidden_size]
        
        # Process layers
        result = processor.process(input_tensor, assigned_layers, {})
        
        # Verify output shape is preserved
        assert result.shape == input_tensor.shape
        
        # Verify all assigned layers were processed
        for layer_idx in assigned_layers:
            layer = mock_mlx_model.model.layers[layer_idx]
            layer.input_layernorm.assert_called()
            layer.self_attn.assert_called()
            layer.mlp.assert_called()
    
    def test_embedding_to_output_pipeline(self, mock_mlx_model):
        """Test complete pipeline from embedding to output."""
        # Test embedding device
        embedding_processor = LayerProcessor(mock_mlx_model, "coordinator", [])
        
        # Test output device  
        output_processor = LayerProcessor(mock_mlx_model, "worker_2", [])
        
        # Process embedding
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hidden_states = embedding_processor.process_embedding(input_ids)
        
        # Process output
        logits = output_processor.process_output(hidden_states)
        
        # Verify shapes
        assert hidden_states.shape == (1, 10, 512)  # From mock
        assert logits.shape == (1, 10, 32000)       # From mock
        
        # Verify components were called
        mock_mlx_model.model.embed_tokens.assert_called_once()
        mock_mlx_model.model.norm.assert_called_once()
        mock_mlx_model.lm_head.assert_called_once()
    
    def test_memory_tracking_integration(self, mock_mlx_model):
        """Test memory usage tracking during processing."""
        processor = LayerProcessor(mock_mlx_model, "test_device", [0, 1, 2])
        
        # Get initial memory stats
        initial_memory = processor.get_memory_usage()
        
        # Process some layers
        input_tensor = mx.ones((2, 20, 512))  # Larger tensor
        processor.process(input_tensor, [0, 1], {})
        
        # Get memory stats after processing
        final_memory = processor.get_memory_usage()
        
        # Verify memory stats structure
        for memory_type in ['allocated_gb', 'cached_gb', 'reserved_gb']:
            assert memory_type in initial_memory
            assert memory_type in final_memory
            assert isinstance(initial_memory[memory_type], (int, float))
            assert isinstance(final_memory[memory_type], (int, float))


class TestEndToEndRequestFlow:
    """Integration tests for complete request processing flow."""
    
    @pytest.mark.asyncio
    async def test_complete_inference_request_flow(self, mock_cluster_config, 
                                                  mock_mlx_model, mock_tokenizer):
        """Test complete flow from request to response."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        
        # Mock initialization components
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_full_model.return_value = (mock_mlx_model, mock_tokenizer)
            
            with patch('src.coordinator.orchestrator.LayerProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                mock_processor.process.return_value = mx.ones((1, 5, 512))
                mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                mock_processor_class.return_value = mock_processor
                
                with patch('src.coordinator.orchestrator.ConnectionPool') as mock_pool_class:
                    mock_pool = MagicMock()
                    mock_client = MagicMock()
                    mock_client.health_check.return_value = {'healthy': True}
                    mock_pool.get_client.return_value = mock_client
                    mock_pool.get_next_device_client.return_value = None  # Coordinator only
                    mock_pool_class.return_value = mock_pool
                    
                    # Mock text generation
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        # Return EOS token to end generation
                        mock_sampler.return_value = mx.array([[mock_tokenizer.eos_token_id]])
                        mock_make_sampler.return_value = mock_sampler
                        
                        # Create and process request
                        request = InferenceRequest(
                            request_id="integration_test_123",
                            messages=[{"role": "user", "content": "Hello, AI!"}],
                            max_tokens=50,
                            temperature=0.7
                        )
                        
                        response = await orchestrator.process_request(request)
                        
                        # Verify response structure
                        assert response.request_id == request.request_id
                        assert isinstance(response.content, str)
                        assert response.tokens_generated >= 0
                        assert response.total_time_ms > 0
                        assert isinstance(response.device_times, dict)
                        
                        # Verify initialization occurred
                        assert orchestrator.is_initialized
                        
                        # Verify processing pipeline was called
                        mock_processor.process_embedding.assert_called()
                        mock_tokenizer.encode.assert_called()
                        mock_tokenizer.decode.assert_called()
    
    @pytest.mark.asyncio
    async def test_request_error_handling(self, mock_cluster_config, mock_mlx_model, mock_tokenizer):
        """Test error handling in request processing."""
        orchestrator = DistributedOrchestrator(mock_cluster_config)
        orchestrator.is_initialized = True
        orchestrator.model = mock_mlx_model
        orchestrator.tokenizer = mock_tokenizer
        
        # Mock layer processor that fails
        mock_processor = MagicMock()
        mock_processor.process_embedding.side_effect = Exception("Processing failed")
        orchestrator.layer_processor = mock_processor
        
        request = InferenceRequest(
            request_id="error_test_123",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Processing failed"):
            await orchestrator.process_request(request)
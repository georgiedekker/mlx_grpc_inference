"""
End-to-end tests for full distributed inference pipeline.
"""

import pytest
import asyncio
import tempfile
import subprocess
import time
import signal
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.coordinator.api_server import create_app
from src.coordinator.orchestrator import DistributedOrchestrator, InferenceRequest


class TestFullInferencePipeline:
    """End-to-end tests for complete distributed inference system."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_device_inference_pipeline(self, mock_cluster_config, test_model_name):
        """Test inference pipeline on single device (coordinator only)."""
        # Modify config for single device
        mock_cluster_config.devices = [mock_cluster_config.devices[0]]  # Only coordinator
        mock_cluster_config.model.layer_distribution = {
            "coordinator": list(range(9))  # All layers on coordinator
        }
        
        # Mock model loading to avoid downloading
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('mlx_lm.load') as mock_mlx_load:
                # Create mock model and tokenizer
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                mock_tokenizer.decode.return_value = "Test response generated successfully"
                mock_tokenizer.eos_token_id = 2
                
                mock_mlx_load.return_value = (mock_model, mock_tokenizer)
                mock_loader = mock_loader_class.return_value
                mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                
                # Mock layer processing
                with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                    mock_processor = MagicMock()
                    # Mock embedding processing
                    import mlx.core as mx
                    mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                    mock_processor.process.return_value = mx.ones((1, 5, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    # Mock text generation
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])  # EOS token
                        mock_make_sampler.return_value = mock_sampler
                        
                        # Create orchestrator and test
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Test inference request
                        request = InferenceRequest(
                            request_id="e2e_test_001",
                            messages=[
                                {"role": "user", "content": "Hello! Please respond with a greeting."}
                            ],
                            max_tokens=50,
                            temperature=0.1  # Low temperature for deterministic output
                        )
                        
                        response = await orchestrator.process_request(request)
                        
                        # Verify response
                        assert response.request_id == request.request_id
                        assert isinstance(response.content, str)
                        assert len(response.content) > 0
                        assert response.tokens_generated > 0
                        assert response.total_time_ms > 0
                        assert 'coordinator' in response.device_times
                        assert 'embedding' in response.device_times
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multi_device_simulation(self, mock_cluster_config):
        """Test simulated multi-device inference pipeline."""
        # Mock all components for multi-device simulation
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.communication.grpc_client.ConnectionPool') as mock_pool_class:
                with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                    
                    # Setup mock model and tokenizer
                    mock_model = MagicMock()
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer.decode.return_value = "Multi-device response"
                    mock_tokenizer.eos_token_id = 2
                    
                    mock_loader = mock_loader_class.return_value
                    mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                    
                    # Setup mock layer processor
                    import mlx.core as mx
                    mock_processor = MagicMock()
                    mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                    mock_processor.process.return_value = mx.ones((1, 5, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    # Setup mock connection pool and clients
                    from src.communication.grpc_client import ProcessingResult
                    mock_pool = MagicMock()
                    
                    # Mock worker clients
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
                    worker1_client.health_check.return_value = {'healthy': True}
                    worker2_client.health_check.return_value = {'healthy': True}
                    
                    # Setup connection pool to simulate device communication
                    mock_pool.get_client.return_value = worker1_client
                    mock_pool.get_next_device_client.side_effect = [worker1_client, worker2_client, None]
                    mock_pool_class.return_value = mock_pool
                    
                    # Mock text generation
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])  # EOS token
                        mock_make_sampler.return_value = mock_sampler
                        
                        # Test multi-device inference
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        request = InferenceRequest(
                            request_id="e2e_multidevice_001",
                            messages=[
                                {"role": "user", "content": "Test multi-device processing"}
                            ],
                            max_tokens=30
                        )
                        
                        response = await orchestrator.process_request(request)
                        
                        # Verify multi-device processing
                        assert response.request_id == request.request_id
                        assert 'coordinator' in response.device_times
                        assert 'worker_1' in response.device_times
                        assert 'worker_2' in response.device_times
                        assert response.device_times['worker_1'] == 100.0
                        assert response.device_times['worker_2'] == 120.0
                        
                        # Verify worker communication
                        worker1_client.process_layers.assert_called_once()
                        worker2_client.process_layers.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, mock_cluster_config):
        """Test error recovery in end-to-end scenarios."""
        
        # Test 1: Worker failure during processing
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.communication.grpc_client.ConnectionPool') as mock_pool_class:
                # Setup basic mocks
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_tokenizer.encode.return_value = [1, 2, 3]
                
                mock_loader = mock_loader_class.return_value
                mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                
                # Setup failing worker
                mock_pool = MagicMock()
                failing_client = MagicMock()
                failing_client.health_check.return_value = {'healthy': False}
                mock_pool.get_client.return_value = failing_client
                mock_pool_class.return_value = mock_pool
                
                orchestrator = DistributedOrchestrator(mock_cluster_config)
                
                # Should fail during initialization due to unhealthy worker
                with pytest.raises(RuntimeError, match="Worker .* is not healthy"):
                    await orchestrator.initialize()
        
        # Test 2: Model loading failure
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            mock_loader = mock_loader_class.return_value
            mock_loader.load_full_model.side_effect = Exception("Model not found")
            
            orchestrator = DistributedOrchestrator(mock_cluster_config)
            
            with pytest.raises(Exception, match="Model not found"):
                await orchestrator.initialize()
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, temp_dir):
        """Test configuration validation in end-to-end scenarios."""
        
        # Test invalid configuration
        invalid_config = {
            "devices": [
                {
                    "device_id": "coordinator",
                    "hostname": "localhost",
                    "rank": 0,
                    "role": "coordinator",
                    "grpc_port": 50051,
                    "api_port": 8000
                }
                # Missing worker devices
            ],
            "model": {
                "name": "test_model",
                "path": "test/path",
                "layer_distribution": {
                    "coordinator": [0, 1, 2],
                    "worker_1": [3, 4, 5]  # References non-existent worker
                },
                "max_sequence_length": 2048,
                "context_window": 4096
            },
            "performance": {
                "request_timeout_seconds": 30.0,
                "max_concurrent_requests": 10,
                "tensor_compression": True,
                "memory_limit_gb": 8.0
            }
        }
        
        config_file = temp_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Test that configuration loading handles inconsistencies
        from src.core.config import ClusterConfig
        
        # This should handle the validation gracefully
        # (exact behavior depends on implementation)
        try:
            config = ClusterConfig.from_file(config_file)
            # If loading succeeds, verify the configuration is sanitized
            device_ids = [device.device_id for device in config.devices]
            layer_devices = list(config.model.layer_distribution.keys())
            
            # All layer assignment devices should exist
            for layer_device in layer_devices:
                if layer_device not in device_ids:
                    pytest.fail(f"Layer assigned to non-existent device: {layer_device}")
        
        except Exception as e:
            # Configuration validation caught the error - this is also acceptable
            assert "worker_1" in str(e) or "device" in str(e).lower()


class TestAPIServerIntegration:
    """End-to-end tests for API server integration."""
    
    @pytest.mark.asyncio
    async def test_api_server_startup_shutdown(self, mock_cluster_config):
        """Test API server startup and shutdown sequence."""
        
        # Mock orchestrator to avoid model loading
        with patch('src.coordinator.api_server.DistributedOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.initialize = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Create FastAPI app
            app = create_app(mock_cluster_config)
            
            # Test that app was created successfully
            assert app is not None
            assert hasattr(app, 'routes')
            
            # Verify orchestrator was initialized
            mock_orchestrator_class.assert_called_once_with(mock_cluster_config)
    
    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self, mock_cluster_config):
        """Test health endpoint with orchestrator integration."""
        
        with patch('src.coordinator.api_server.DistributedOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.initialize = AsyncMock()
            mock_orchestrator.is_initialized = True
            mock_orchestrator_class.return_value = mock_orchestrator
            
            app = create_app(mock_cluster_config)
            
            # Test health endpoint
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            response = client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert "model_loaded" in health_data
            assert "timestamp" in health_data


class TestPerformanceCharacteristics:
    """End-to-end performance characteristic tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_inference_latency_bounds(self, mock_cluster_config):
        """Test that inference latency stays within reasonable bounds."""
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            # Setup fast mock processing
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "Fast response"
            mock_tokenizer.eos_token_id = 2
            
            mock_loader = mock_loader_class.return_value
            mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
            
            with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                with patch('src.coordinator.orchestrator.ConnectionPool'):
                    import mlx.core as mx
                    mock_processor = MagicMock()
                    mock_processor.process_embedding.return_value = mx.ones((1, 3, 512))
                    mock_processor.process.return_value = mx.ones((1, 3, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])  # Immediate EOS
                        mock_make_sampler.return_value = mock_sampler
                        
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Test multiple requests for timing consistency
                        latencies = []
                        
                        for i in range(5):
                            request = InferenceRequest(
                                request_id=f"perf_test_{i}",
                                messages=[{"role": "user", "content": "Quick test"}],
                                max_tokens=1  # Minimal generation
                            )
                            
                            start_time = time.time()
                            response = await orchestrator.process_request(request)
                            end_time = time.time()
                            
                            latency = (end_time - start_time) * 1000  # Convert to ms
                            latencies.append(latency)
                            
                            # Verify response timing matches
                            assert response.total_time_ms > 0
                            assert abs(response.total_time_ms - latency) < 100  # Within 100ms
                        
                        # Verify latency consistency (should be relatively stable with mocks)
                        avg_latency = sum(latencies) / len(latencies)
                        max_deviation = max(abs(lat - avg_latency) for lat in latencies)
                        
                        # With mocks, latency should be very consistent
                        assert max_deviation < avg_latency * 0.5  # Within 50% of average
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mock_cluster_config):
        """Test handling multiple concurrent requests."""
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                with patch('src.coordinator.orchestrator.ConnectionPool'):
                    
                    # Setup mocks
                    mock_model = MagicMock()
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.encode.return_value = [1, 2, 3]
                    mock_tokenizer.decode.return_value = "Concurrent response"
                    mock_tokenizer.eos_token_id = 2
                    
                    mock_loader = mock_loader_class.return_value
                    mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                    
                    import mlx.core as mx
                    mock_processor = MagicMock()
                    mock_processor.process_embedding.return_value = mx.ones((1, 3, 512))
                    mock_processor.process.return_value = mx.ones((1, 3, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])
                        mock_make_sampler.return_value = mock_sampler
                        
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Create multiple concurrent requests
                        async def process_request(request_id):
                            request = InferenceRequest(
                                request_id=f"concurrent_{request_id}",
                                messages=[{"role": "user", "content": f"Request {request_id}"}],
                                max_tokens=1
                            )
                            return await orchestrator.process_request(request)
                        
                        # Run 3 concurrent requests
                        tasks = [process_request(i) for i in range(3)]
                        responses = await asyncio.gather(*tasks)
                        
                        # Verify all requests completed successfully
                        assert len(responses) == 3
                        
                        for i, response in enumerate(responses):
                            assert response.request_id == f"concurrent_{i}"
                            assert isinstance(response.content, str)
                            assert response.total_time_ms > 0
"""
Unit tests for the gRPC client components.
"""

import pytest
import grpc
from unittest.mock import MagicMock, patch, AsyncMock
import mlx.core as mx

from src.communication.grpc_client import (
    GRPCInferenceClient,
    ConnectionPool,
    ProcessingResult
)
from src.core.config import DeviceConfig, DeviceRole


class TestProcessingResult:
    """Test cases for ProcessingResult dataclass."""
    
    def test_processing_result_creation(self):
        """Test creating a ProcessingResult."""
        output_tensor = mx.ones((1, 5, 512))
        
        result = ProcessingResult(
            output_tensor=output_tensor,
            processing_time_ms=150.5,
            device_id="worker_1"
        )
        
        assert mx.array_equal(result.output_tensor, output_tensor)
        assert result.processing_time_ms == 150.5
        assert result.device_id == "worker_1"


class TestGRPCInferenceClient:
    """Test cases for GRPCInferenceClient."""
    
    @pytest.fixture
    def target_device(self):
        """Create target device configuration."""
        return DeviceConfig(
            device_id="worker_1",
            hostname="localhost",
            rank=1,
            role=DeviceRole.WORKER,
            grpc_port=50052,
            api_port=8001
        )
    
    def test_init(self, target_device):
        """Test GRPCInferenceClient initialization."""
        with patch.object(GRPCInferenceClient, '_connect') as mock_connect:
            client = GRPCInferenceClient(target_device, timeout=30.0)
            
            assert client.target_device == target_device
            assert client.timeout == 30.0
            mock_connect.assert_called_once()
    
    def test_connect(self, target_device):
        """Test establishing gRPC connection."""
        with patch('grpc.insecure_channel') as mock_channel:
            with patch('src.communication.grpc_client.inference_pb2_grpc') as mock_grpc:
                with patch('src.communication.grpc_client.dns_resolver.resolve_grpc_target') as mock_resolve:
                    mock_resolve.return_value = "192.168.1.100:50052"
                    mock_stub = MagicMock()
                    mock_grpc.InferenceServiceStub.return_value = mock_stub
                    
                    client = GRPCInferenceClient(target_device)
                    
                    mock_resolve.assert_called_once_with("localhost", 50052)
                    mock_channel.assert_called_once()
                    mock_grpc.InferenceServiceStub.assert_called_once()
                    assert client.stub == mock_stub
    
    def test_process_layers_success(self, target_device, sample_hidden_states):
        """Test successful layer processing."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            # Mock gRPC components
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            # Mock protobuf modules
            with patch('src.communication.grpc_client.inference_pb2') as mock_pb2:
                with patch('src.communication.grpc_client.serialize_mlx_array') as mock_serialize:
                    with patch('src.communication.grpc_client.deserialize_mlx_array') as mock_deserialize:
                        
                        # Setup mocks
                        mock_serialize.return_value = (b"tensor_data", {
                            'shape': [1, 5, 512],
                            'dtype': 'float32',
                            'compressed': False
                        })
                        mock_deserialize.return_value = sample_hidden_states
                        
                        # Mock response
                        mock_response = MagicMock()
                        mock_response.output_tensor = b"response_data"
                        mock_response.processing_time_ms = 100.0
                        mock_response.device_id = "worker_1"
                        mock_response.metadata.shape = [1, 5, 512]
                        mock_response.metadata.dtype = "float32"
                        mock_response.metadata.compressed = False
                        
                        mock_stub.ProcessLayers.return_value = mock_response
                        
                        # Make the call
                        result = client.process_layers(
                            sample_hidden_states,
                            [3, 4, 5],
                            "test_request_123",
                            {"key": "value"}
                        )
                        
                        # Verify result
                        assert isinstance(result, ProcessingResult)
                        assert mx.array_equal(result.output_tensor, sample_hidden_states)
                        assert result.processing_time_ms == 100.0
                        assert result.device_id == "worker_1"
                        
                        # Verify gRPC call was made
                        mock_stub.ProcessLayers.assert_called_once()
    
    def test_process_layers_grpc_error(self, target_device, sample_hidden_states):
        """Test layer processing with gRPC error."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            # Mock gRPC error
            grpc_error = grpc.RpcError()
            grpc_error.code = MagicMock(return_value=grpc.StatusCode.UNAVAILABLE)
            grpc_error.details = MagicMock(return_value="Service unavailable")
            mock_stub.ProcessLayers.side_effect = grpc_error
            
            with patch('src.communication.grpc_client.serialize_mlx_array'):
                with pytest.raises(grpc.RpcError):
                    client.process_layers(
                        sample_hidden_states,
                        [3, 4, 5],
                        "test_request_123"
                    )
    
    def test_process_layers_with_compression(self, target_device, sample_hidden_states):
        """Test layer processing with tensor compression."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            with patch('src.communication.grpc_client.inference_pb2'):
                with patch('src.communication.grpc_client.serialize_mlx_array') as mock_serialize:
                    with patch('src.communication.grpc_client.deserialize_mlx_array'):
                        
                        mock_serialize.return_value = (b"compressed_data", {
                            'shape': [1, 5, 512],
                            'dtype': 'float32',
                            'compressed': True
                        })
                        
                        mock_response = MagicMock()
                        mock_response.output_tensor = b"response_data"
                        mock_response.processing_time_ms = 100.0
                        mock_response.device_id = "worker_1"
                        mock_response.metadata.shape = [1, 5, 512]
                        mock_response.metadata.dtype = "float32"
                        mock_response.metadata.compressed = True
                        
                        mock_stub.ProcessLayers.return_value = mock_response
                        
                        client.process_layers(
                            sample_hidden_states,
                            [3, 4, 5],
                            "test_request_123",
                            compress=True
                        )
                        
                        # Verify compression was requested
                        mock_serialize.assert_called_once_with(sample_hidden_states, compress=True)
    
    def test_health_check_success(self, target_device):
        """Test successful health check."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            # Mock response
            mock_response = MagicMock()
            mock_response.healthy = True
            mock_response.device_id = "worker_1"
            mock_response.timestamp = "2024-01-01T00:00:00Z"
            mock_response.details = {"status": "ok"}
            
            mock_stub.HealthCheck.return_value = mock_response
            
            with patch('src.communication.grpc_client.inference_pb2'):
                result = client.health_check()
                
                assert result['healthy'] is True
                assert result['device_id'] == "worker_1"
                assert result['timestamp'] == "2024-01-01T00:00:00Z"
                assert isinstance(result['details'], dict)
    
    def test_health_check_grpc_error(self, target_device):
        """Test health check with gRPC error."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            # Mock gRPC error
            mock_stub.HealthCheck.side_effect = grpc.RpcError("Connection failed")
            
            with patch('src.communication.grpc_client.inference_pb2'):
                result = client.health_check()
                
                assert result['healthy'] is False
                assert result['device_id'] == "worker_1"
                assert 'error' in result
    
    def test_get_device_info_success(self, target_device):
        """Test successful device info retrieval."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            # Mock response
            mock_response = MagicMock()
            mock_response.device_id = "worker_1"
            mock_response.hostname = "localhost"
            mock_response.rank = 1
            mock_response.role = "worker"
            mock_response.assigned_layers = [3, 4, 5]
            mock_response.capabilities = {"memory_gb": 8.0}
            mock_response.gpu_utilization = 0.75
            mock_response.memory_usage_gb = 2.5
            
            mock_stub.GetDeviceInfo.return_value = mock_response
            
            with patch('src.communication.grpc_client.inference_pb2'):
                result = client.get_device_info()
                
                assert result['device_id'] == "worker_1"
                assert result['hostname'] == "localhost"
                assert result['rank'] == 1
                assert result['role'] == "worker"
                assert result['assigned_layers'] == [3, 4, 5]
                assert result['gpu_utilization'] == 0.75
                assert result['memory_usage_gb'] == 2.5
    
    def test_get_device_info_grpc_error(self, target_device):
        """Test device info retrieval with gRPC error."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_stub = MagicMock()
            client.stub = mock_stub
            
            mock_stub.GetDeviceInfo.side_effect = grpc.RpcError("Connection failed")
            
            with patch('src.communication.grpc_client.inference_pb2'):
                with pytest.raises(grpc.RpcError):
                    client.get_device_info()
    
    def test_close(self, target_device):
        """Test closing the connection."""
        with patch.object(GRPCInferenceClient, '_connect'):
            client = GRPCInferenceClient(target_device)
            
            mock_channel = MagicMock()
            client.channel = mock_channel
            
            client.close()
            
            mock_channel.close.assert_called_once()


class TestConnectionPool:
    """Test cases for ConnectionPool."""
    
    def test_init(self, mock_cluster_config):
        """Test ConnectionPool initialization."""
        local_device_id = "coordinator"
        
        with patch.object(ConnectionPool, '_initialize_connections') as mock_init:
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            assert pool.config == mock_cluster_config
            assert pool.local_device_id == local_device_id
            assert isinstance(pool.clients, dict)
            mock_init.assert_called_once()
    
    def test_initialize_connections_success(self, mock_cluster_config):
        """Test successful connection initialization."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Should create clients for all non-local devices
            expected_devices = ["worker_1", "worker_2"]
            assert len(pool.clients) == len(expected_devices)
            
            for device_id in expected_devices:
                assert device_id in pool.clients
                assert pool.clients[device_id] == mock_client
    
    def test_initialize_connections_failure(self, mock_cluster_config, caplog):
        """Test connection initialization with some failures."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            # Make connection fail for worker_1
            def side_effect(device, timeout):
                if device.device_id == "worker_1":
                    raise Exception("Connection failed")
                return MagicMock()
            
            mock_client_class.side_effect = side_effect
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Should have connection to worker_2 but not worker_1
            assert "worker_2" in pool.clients
            assert "worker_1" not in pool.clients
            assert "Failed to connect to worker_1" in caplog.text
    
    def test_get_client_existing(self, mock_cluster_config):
        """Test getting existing client."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            result = pool.get_client("worker_1")
            assert result == mock_client
    
    def test_get_client_nonexistent(self, mock_cluster_config):
        """Test getting non-existent client."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient'):
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            result = pool.get_client("nonexistent_device")
            assert result is None
    
    def test_get_next_device_client_success(self, mock_cluster_config):
        """Test getting next device client in pipeline."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Get next device after coordinator (rank 0) should be worker_1 (rank 1)
            result = pool.get_next_device_client("coordinator")
            assert result == mock_client
    
    def test_get_next_device_client_last_device(self, mock_cluster_config):
        """Test getting next device client when at last device."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient'):
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # worker_2 has rank 2, which is the highest
            result = pool.get_next_device_client("worker_2")
            assert result is None
    
    def test_get_next_device_client_invalid_device(self, mock_cluster_config):
        """Test getting next device client with invalid current device."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient'):
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            result = pool.get_next_device_client("nonexistent_device")
            assert result is None
    
    def test_close_all(self, mock_cluster_config):
        """Test closing all connections."""
        local_device_id = "coordinator"
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            pool = ConnectionPool(mock_cluster_config, local_device_id)
            
            # Verify clients exist
            assert len(pool.clients) > 0
            
            pool.close_all()
            
            # Verify all clients were closed and cleared
            mock_client.close.assert_called()
            assert len(pool.clients) == 0
    
    def test_connection_pool_with_timeout_config(self, mock_cluster_config):
        """Test connection pool uses timeout from config."""
        local_device_id = "coordinator"
        mock_cluster_config.performance.request_timeout_seconds = 45.0
        
        with patch('src.communication.grpc_client.GRPCInferenceClient') as mock_client_class:
            ConnectionPool(mock_cluster_config, local_device_id)
            
            # Verify clients were created with config timeout
            for call_args in mock_client_class.call_args_list:
                assert call_args[1]['timeout'] == 45.0
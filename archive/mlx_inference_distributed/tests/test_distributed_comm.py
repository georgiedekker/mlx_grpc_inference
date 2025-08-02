"""
Test suite for distributed communication functionality.
"""

import pytest
import grpc
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_comm import (
    CommunicationBackend, CommunicationType, GRPCCommunicator,
    create_communicator, DistributedCommunicator
)
import distributed_comm_pb2
import distributed_comm_pb2_grpc


class TestCommunicationBackend:
    """Test communication backend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert CommunicationBackend.GRPC.value == "grpc"
        assert CommunicationBackend.NCCL.value == "nccl"
        assert CommunicationBackend.GLOO.value == "gloo"
    
    def test_backend_from_string(self):
        """Test creating backend from string."""
        backend = CommunicationBackend("grpc")
        assert backend == CommunicationBackend.GRPC


class TestCommunicationType:
    """Test communication type enum."""
    
    def test_type_values(self):
        """Test communication type values."""
        assert CommunicationType.TENSOR.value == "tensor"
        assert CommunicationType.STRING.value == "string"
        assert CommunicationType.PICKLE.value == "pickle"


class TestCreateCommunicator:
    """Test communicator factory function."""
    
    def test_create_grpc_communicator(self):
        """Test creating gRPC communicator."""
        comm = create_communicator(CommunicationBackend.GRPC)
        assert isinstance(comm, GRPCCommunicator)
    
    def test_create_unsupported_backend(self):
        """Test creating unsupported backend raises error."""
        with pytest.raises(ValueError, match="backend is not supported"):
            create_communicator(CommunicationBackend.NCCL)


class TestGRPCCommunicator:
    """Test gRPC communicator functionality."""
    
    @pytest.fixture
    def communicator(self):
        """Create a gRPC communicator instance."""
        return GRPCCommunicator()
    
    def test_initialization(self, communicator):
        """Test communicator initialization."""
        assert communicator.rank is None
        assert communicator.world_size is None
        assert communicator._initialized is False
        assert communicator._server is None
        assert communicator._base_port == 50100
    
    def test_init_method(self, communicator):
        """Test init method."""
        with patch('grpc.server') as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance
            
            communicator.init(rank=0, world_size=2)
            
            assert communicator.rank == 0
            assert communicator.world_size == 2
            assert communicator._initialized is True
            mock_server_instance.add_insecure_port.assert_called_once()
            mock_server_instance.start.assert_called_once()
    
    def test_init_with_custom_port(self, communicator):
        """Test init with custom port."""
        with patch('grpc.server') as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value = mock_server_instance
            
            communicator.init(rank=1, world_size=2, port=60000)
            
            # Should use custom port instead of calculated one
            mock_server_instance.add_insecure_port.assert_called_with('[::]:60000')
    
    def test_init_with_device_hostnames(self, communicator):
        """Test init with device hostnames."""
        with patch('grpc.server') as mock_server:
            with patch('grpc.insecure_channel') as mock_channel:
                mock_server.return_value = MagicMock()
                mock_channel.return_value = MagicMock()
                
                hostnames = ["mini1.local", "mini2.local", "master.local"]
                communicator.init(rank=0, world_size=3, device_hostnames=hostnames)
                
                assert communicator.device_hostnames == hostnames
                # Should create channels to other devices
                assert mock_channel.call_count == 2  # world_size - 1
    
    def test_finalize(self, communicator):
        """Test finalize method."""
        # Setup a mock server
        communicator._server = MagicMock()
        communicator._initialized = True
        
        communicator.finalize()
        
        assert communicator._initialized is False
        communicator._server.stop.assert_called_once()
    
    def test_send_tensor(self, communicator):
        """Test sending tensor data."""
        # Setup communicator
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 2
        mock_stub = MagicMock()
        communicator._stubs = {1: mock_stub}
        
        # Create test tensor (using numpy as mock for mx.array)
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        
        # Mock the send response
        mock_stub.Send.return_value = MagicMock(success=True)
        
        # Send tensor
        communicator.send(tensor, dest=1, tag=0, comm_type=CommunicationType.TENSOR)
        
        # Verify send was called
        mock_stub.Send.assert_called_once()
        
        # Check the request
        args = mock_stub.Send.call_args[0]
        request = args[0]
        assert request.dest_rank == 1
        assert request.tag == 0
        assert request.data.comm_type == CommunicationType.TENSOR.value
    
    def test_send_string(self, communicator):
        """Test sending string data."""
        communicator._initialized = True
        communicator.rank = 0
        mock_stub = MagicMock()
        communicator._stubs = {1: mock_stub}
        
        mock_stub.Send.return_value = MagicMock(success=True)
        
        communicator.send("Hello, world!", dest=1, tag=0, comm_type=CommunicationType.STRING)
        
        mock_stub.Send.assert_called_once()
        args = mock_stub.Send.call_args[0]
        request = args[0]
        assert request.data.string_data.data == "Hello, world!"
    
    def test_broadcast(self, communicator):
        """Test broadcast operation."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 3
        
        # Mock stubs for other ranks
        mock_stubs = {1: MagicMock(), 2: MagicMock()}
        communicator._stubs = mock_stubs
        
        for stub in mock_stubs.values():
            stub.Broadcast.return_value = MagicMock(success=True, data=None)
        
        # Broadcast from root
        data = {"message": "broadcast test"}
        result = communicator.broadcast(data, root=0)
        
        assert result == data  # Root returns original data
        
        # Verify broadcast was called on all other ranks
        for stub in mock_stubs.values():
            stub.Broadcast.assert_called_once()
    
    def test_barrier(self, communicator):
        """Test barrier synchronization."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 2
        
        mock_stub = MagicMock()
        communicator._stubs = {1: mock_stub}
        
        # Mock servicer for local receive
        mock_servicer = MagicMock()
        communicator._servicer = mock_servicer
        
        # Mock receive to simulate barrier completion
        mock_servicer.Receive.return_value = [MagicMock()]
        
        communicator.barrier()
        
        # Should send to and receive from other rank
        mock_stub.Send.assert_called()
    
    def test_all_reduce_sum(self, communicator):
        """Test all-reduce with sum operation."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 2
        
        mock_stub = MagicMock()
        communicator._stubs = {1: mock_stub}
        
        # Create test tensor
        local_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Mock response with reduced data
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data.tensor_data.data = np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes()
        mock_response.data.tensor_data.shape = [3]
        mock_response.data.tensor_data.dtype = "float32"
        
        mock_stub.AllReduce.return_value = mock_response
        
        # Mock numpy conversion
        with patch('distributed_comm.np.frombuffer') as mock_frombuffer:
            mock_frombuffer.return_value = np.array([4.0, 5.0, 6.0], dtype=np.float32)
            
            result = communicator.all_reduce(local_tensor, op='sum')
            
            # Verify all-reduce was called
            mock_stub.AllReduce.assert_called_once()
    
    def test_gather(self, communicator):
        """Test gather operation."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 2
        
        # Mock servicer for receiving
        mock_servicer = MagicMock()
        communicator._servicer = mock_servicer
        
        # Mock receive responses
        mock_response = MagicMock()
        mock_response.data.string_data.data = "data from rank 1"
        mock_servicer.Receive.return_value = [mock_response]
        
        # Gather at root
        local_data = "data from rank 0"
        gathered = communicator.gather(local_data, root=0)
        
        assert len(gathered) == 2
        assert gathered[0] == local_data
    
    def test_error_handling_send_not_initialized(self, communicator):
        """Test error when sending without initialization."""
        with pytest.raises(RuntimeError, match="Communicator not initialized"):
            communicator.send("test", dest=1)
    
    def test_error_handling_invalid_rank(self, communicator):
        """Test error with invalid rank."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 2
        
        with pytest.raises(ValueError, match="Invalid destination rank"):
            communicator.send("test", dest=5)
    
    def test_channel_options(self, communicator):
        """Test gRPC channel options."""
        options = communicator._channel_options
        
        # Check max message size options
        assert ('grpc.max_send_message_length', 100 * 1024 * 1024) in options
        assert ('grpc.max_receive_message_length', 100 * 1024 * 1024) in options
        assert ('grpc.dns_resolver', 'native') in options
    
    @patch('time.time')
    def test_timeout_handling(self, mock_time, communicator):
        """Test timeout handling in operations."""
        # Simulate timeout by advancing time
        mock_time.side_effect = [0, 0, 35]  # Start, check, timeout
        
        communicator._initialized = True
        communicator.rank = 0
        communicator._servicer = MagicMock()
        communicator._servicer.Receive.return_value = []  # No data received
        
        # This should timeout
        with pytest.raises(Exception):
            communicator.receive(src=1, comm_type=CommunicationType.TENSOR)
    
    def test_concurrent_operations(self, communicator):
        """Test concurrent send/receive operations."""
        communicator._initialized = True
        communicator.rank = 0
        communicator.world_size = 3
        
        # Mock multiple stubs
        mock_stubs = {1: MagicMock(), 2: MagicMock()}
        communicator._stubs = mock_stubs
        
        for stub in mock_stubs.values():
            stub.Send.return_value = MagicMock(success=True)
        
        # Send to multiple destinations
        for dest in [1, 2]:
            communicator.send(f"message to {dest}", dest=dest)
        
        # Verify all sends completed
        for dest, stub in mock_stubs.items():
            stub.Send.assert_called_once()


class TestDistributedCommunicatorInterface:
    """Test the abstract base class interface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            DistributedCommunicator()
    
    def test_interface_methods(self):
        """Test that interface defines all required methods."""
        required_methods = [
            'init', 'finalize', 'send', 'receive', 'broadcast',
            'all_reduce', 'reduce', 'all_gather', 'gather', 'scatter',
            'barrier', 'send_tensor', 'receive_tensor'
        ]
        
        for method in required_methods:
            assert hasattr(DistributedCommunicator, method)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
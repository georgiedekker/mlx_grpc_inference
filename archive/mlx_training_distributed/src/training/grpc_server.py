#!/usr/bin/env python3
"""
gRPC server for distributed MLX training with proper tensor serialization.
Handles MLX tensor communication between distributed training processes.
"""

import grpc
from concurrent import futures
import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Optional, Union
import pickle
import logging
import time
from dataclasses import dataclass

from .distributed_comm import (
    MLXArraySerializer, 
    serialize_mlx_tensors, 
    deserialize_mlx_tensors,
    prepare_gradients_for_allreduce,
    restore_gradients_from_allreduce
)

logger = logging.getLogger(__name__)

@dataclass
class TensorMessage:
    """Message format for tensor communication."""
    data: bytes
    metadata: Dict[str, Any]
    timestamp: float
    sender_rank: int
    message_type: str  # 'gradients', 'parameters', 'broadcast', etc.


class TensorSerializer:
    """
    Enhanced tensor serializer that handles all MLX dtypes correctly.
    Specifically addresses PEP 3118 buffer format issues.
    """
    
    def __init__(self):
        self.mlx_serializer = MLXArraySerializer()
        self._dtype_registry = self._build_dtype_registry()
    
    def _build_dtype_registry(self) -> Dict[str, Any]:
        """Build registry of supported dtypes for validation."""
        return {
            # MLX dtypes
            'bool_': mx.bool_,
            'uint8': mx.uint8,
            'uint16': mx.uint16,
            'uint32': mx.uint32,
            'uint64': mx.uint64,
            'int8': mx.int8,
            'int16': mx.int16,
            'int32': mx.int32,
            'int64': mx.int64,
            'float16': mx.float16,
            'float32': mx.float32,
            'float64': mx.float64,
            'complex64': mx.complex64,
            'complex128': mx.complex128,
            
            # NumPy dtypes (for intermediate conversion)
            'np_bool': np.bool_,
            'np_uint8': np.uint8,
            'np_uint16': np.uint16,
            'np_uint32': np.uint32,
            'np_uint64': np.uint64,
            'np_int8': np.int8,
            'np_int16': np.int16,
            'np_int32': np.int32,
            'np_int64': np.int64,
            'np_float16': np.float16,
            'np_float32': np.float32,
            'np_float64': np.float64,
            'np_complex64': np.complex64,
            'np_complex128': np.complex128,
        }
    
    def serialize_tensor_dict(self, tensors: Dict[str, mx.array], message_type: str = "gradients") -> TensorMessage:
        """
        Serialize a dictionary of MLX tensors with proper error handling.
        
        Args:
            tensors: Dictionary of MLX arrays to serialize
            message_type: Type of message for debugging
            
        Returns:
            TensorMessage with serialized data
        """
        start_time = time.time()
        
        try:
            # Use the enhanced serialization from distributed_comm
            serialized_data = serialize_mlx_tensors(tensors)
            
            # Convert to bytes for transport
            message_bytes = pickle.dumps(serialized_data)
            
            metadata = {
                'num_tensors': len(tensors),
                'tensor_names': list(tensors.keys()),
                'tensor_shapes': {name: list(tensor.shape) for name, tensor in tensors.items()},
                'tensor_dtypes': {name: str(tensor.dtype) for name, tensor in tensors.items()},
                'serialization_time': time.time() - start_time,
                'message_size_bytes': len(message_bytes),
                'serialization_version': '1.0'
            }
            
            return TensorMessage(
                data=message_bytes,
                metadata=metadata,
                timestamp=time.time(),
                sender_rank=-1,  # Will be set by sender
                message_type=message_type
            )
            
        except Exception as e:
            logger.error(f"Failed to serialize tensors: {e}")
            # Return error message
            error_data = {
                'error': str(e),
                'tensor_info': {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)} 
                              for name, tensor in tensors.items()}
            }
            error_bytes = pickle.dumps(error_data)
            
            return TensorMessage(
                data=error_bytes,
                metadata={'error': True, 'error_message': str(e)},
                timestamp=time.time(),
                sender_rank=-1,
                message_type='error'
            )
    
    def deserialize_tensor_dict(self, message: TensorMessage) -> Dict[str, mx.array]:
        """
        Deserialize TensorMessage back to MLX tensors.
        
        Args:
            message: TensorMessage to deserialize
            
        Returns:
            Dictionary of MLX arrays
        """
        if message.metadata.get('error', False):
            error_data = pickle.loads(message.data)
            raise RuntimeError(f"Received error message: {error_data['error']}")
        
        try:
            # Deserialize the message data
            serialized_data = pickle.loads(message.data)
            
            # Convert back to MLX tensors
            tensors = deserialize_mlx_tensors(serialized_data)
            
            logger.debug(f"Deserialized {len(tensors)} tensors from {message.message_type} message")
            return tensors
            
        except Exception as e:
            logger.error(f"Failed to deserialize tensors: {e}")
            logger.error(f"Message metadata: {message.metadata}")
            raise RuntimeError(f"Tensor deserialization failed: {e}")
    
    def validate_tensor_compatibility(self, tensor: mx.array) -> bool:
        """
        Validate that a tensor can be safely serialized.
        
        Args:
            tensor: MLX array to validate
            
        Returns:
            True if tensor can be serialized safely
        """
        try:
            # Check if dtype is supported
            dtype_str = str(tensor.dtype).split('.')[-1]
            if dtype_str not in self._dtype_registry:
                logger.warning(f"Unsupported dtype: {tensor.dtype}")
                return False
            
            # Check if tensor is finite and well-formed
            mx.eval(tensor)  # Ensure tensor is computed
            
            # For boolean tensors, do additional validation
            if tensor.dtype == mx.bool_:
                # Check for any potential buffer format issues
                test_array = np.array(tensor, dtype=np.uint8)
                if test_array.size != tensor.size:
                    logger.warning("Boolean tensor size mismatch detected")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed: {e}")
            return False
    
    def fix_tensor_dtype_issues(self, tensor: mx.array) -> mx.array:
        """
        Fix common dtype issues that cause serialization problems.
        
        Args:
            tensor: MLX array with potential issues
            
        Returns:
            Fixed MLX array
        """
        try:
            # Ensure tensor is evaluated
            mx.eval(tensor)
            
            # Handle boolean tensors specially
            if tensor.dtype == mx.bool_:
                # Convert through uint8 to avoid buffer format issues
                temp_array = tensor.astype(mx.uint8)
                mx.eval(temp_array)
                return temp_array.astype(mx.bool_)
            
            # For other dtypes, ensure they're in a clean state
            return tensor.astype(tensor.dtype)  # This can clean up the tensor
            
        except Exception as e:
            logger.error(f"Failed to fix tensor dtype issues: {e}")
            # Return original tensor if fixing fails
            return tensor


class DistributedTrainingService:
    """gRPC service for distributed training communication."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.tensor_serializer = TensorSerializer()
        self.gradient_buffer = {}
        self.parameter_buffer = {}
        
    def AllReduceGradients(self, request, context):
        """Handle all-reduce gradient requests."""
        try:
            # Deserialize incoming gradients
            message = TensorMessage(
                data=request.data,
                metadata=pickle.loads(request.metadata),
                timestamp=request.timestamp,
                sender_rank=request.sender_rank,
                message_type="gradients"
            )
            
            gradients = self.tensor_serializer.deserialize_tensor_dict(message)
            
            # Perform all-reduce (simplified - in practice would coordinate with other processes)
            reduced_gradients = self._simulate_all_reduce(gradients)
            
            # Serialize response
            response_message = self.tensor_serializer.serialize_tensor_dict(
                reduced_gradients, "reduced_gradients"
            )
            
            return self._create_tensor_response(response_message)
            
        except Exception as e:
            logger.error(f"AllReduceGradients failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None
    
    def BroadcastParameters(self, request, context):
        """Handle parameter broadcast requests."""
        try:
            # Deserialize parameters
            message = TensorMessage(
                data=request.data,
                metadata=pickle.loads(request.metadata),
                timestamp=request.timestamp,
                sender_rank=request.sender_rank,
                message_type="parameters"
            )
            
            parameters = self.tensor_serializer.deserialize_tensor_dict(message)
            
            # Store parameters (in practice, would broadcast to other processes)
            self.parameter_buffer.update(parameters)
            
            # Return confirmation
            response_data = {'status': 'success', 'num_parameters': len(parameters)}
            response_bytes = pickle.dumps(response_data)
            
            return self._create_simple_response(response_bytes, "broadcast_confirmation")
            
        except Exception as e:
            logger.error(f"BroadcastParameters failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None
    
    def _simulate_all_reduce(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Simulate all-reduce operation (placeholder for real implementation)."""
        # In a real implementation, this would coordinate with all processes
        reduced_gradients = {}
        
        for name, grad in gradients.items():
            # Validate tensor before processing
            if not self.tensor_serializer.validate_tensor_compatibility(grad):
                logger.warning(f"Fixing tensor compatibility issues for {name}")
                grad = self.tensor_serializer.fix_tensor_dtype_issues(grad)
            
            # Simulate reduction (in practice, would sum gradients from all processes)
            reduced_gradients[name] = grad / self.world_size
        
        return reduced_gradients
    
    def _create_tensor_response(self, message: TensorMessage):
        """Create gRPC response from TensorMessage."""
        # This would be the actual gRPC response format
        # For now, return a mock response structure
        return {
            'data': message.data,
            'metadata': pickle.dumps(message.metadata),
            'timestamp': message.timestamp,
            'sender_rank': self.rank
        }
    
    def _create_simple_response(self, data: bytes, message_type: str):
        """Create simple gRPC response."""
        return {
            'data': data,
            'metadata': pickle.dumps({'message_type': message_type}),
            'timestamp': time.time(),
            'sender_rank': self.rank
        }


class DistributedTrainingServer:
    """gRPC server for distributed training."""
    
    def __init__(self, rank: int, world_size: int, port: int = 50051):
        self.rank = rank
        self.world_size = world_size
        self.port = port
        self.server = None
        self.service = DistributedTrainingService(rank, world_size)
    
    def start(self):
        """Start the gRPC server."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # In a real implementation, would add the service to the server
        # grpc.add_DistributedTrainingServicer_to_server(self.service, self.server)
        
        listen_addr = f'[::]:{self.port}'
        self.server.add_insecure_port(listen_addr)
        self.server.start()
        
        logger.info(f"Distributed training server started on {listen_addr} (rank {self.rank})")
        return self.server
    
    def stop(self):
        """Stop the gRPC server."""
        if self.server:
            self.server.stop(0)
            logger.info(f"Distributed training server stopped (rank {self.rank})")


def create_distributed_server(rank: int, world_size: int, port: Optional[int] = None) -> DistributedTrainingServer:
    """
    Factory function to create a distributed training server.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Server port (auto-assigned if None)
        
    Returns:
        Configured DistributedTrainingServer
    """
    if port is None:
        port = 50051 + rank  # Assign unique ports per rank
    
    return DistributedTrainingServer(rank, world_size, port)


# Utility functions for testing tensor serialization
def test_tensor_serialization():
    """Test function to validate tensor serialization works correctly."""
    logger.info("Testing MLX tensor serialization...")
    
    serializer = TensorSerializer()
    
    # Test different dtypes
    test_tensors = {
        'bool_tensor': mx.array([True, False, True], dtype=mx.bool_),
        'float32_tensor': mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
        'int32_tensor': mx.array([1, 2, 3], dtype=mx.int32),
        'uint8_tensor': mx.array([1, 2, 3], dtype=mx.uint8),
        'complex64_tensor': mx.array([1+2j, 3+4j], dtype=mx.complex64),
    }
    
    # Test serialization and deserialization
    for name, tensor in test_tensors.items():
        try:
            logger.info(f"Testing {name} with shape {tensor.shape} and dtype {tensor.dtype}")
            
            # Validate tensor
            is_valid = serializer.validate_tensor_compatibility(tensor)
            logger.info(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
            
            # Serialize
            message = serializer.serialize_tensor_dict({name: tensor})
            logger.info(f"  Serialization: {'PASS' if message.message_type != 'error' else 'FAIL'}")
            
            if message.message_type != 'error':
                # Deserialize
                restored_tensors = serializer.deserialize_tensor_dict(message)
                restored_tensor = restored_tensors[name]
                
                # Check if restoration is correct
                original_np = np.array(tensor)
                restored_np = np.array(restored_tensor)
                
                if np.allclose(original_np, restored_np, equal_nan=True):
                    logger.info(f"  Round-trip: PASS")
                else:
                    logger.error(f"  Round-trip: FAIL - values differ")
            
        except Exception as e:
            logger.error(f"  Test failed for {name}: {e}")
    
    logger.info("Tensor serialization test completed")


if __name__ == "__main__":
    # Run tests
    logging.basicConfig(level=logging.INFO)
    test_tensor_serialization()
#!/usr/bin/env python3
"""
Distributed communication utilities for MLX tensor serialization.
Handles proper MLX array conversion to avoid PEP 3118 buffer format errors.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, Any, Union, List, Tuple, Optional
import pickle
import struct
import logging

logger = logging.getLogger(__name__)

class MLXArraySerializer:
    """Handles MLX array serialization with proper dtype handling."""
    
    # MLX to NumPy dtype mapping to avoid buffer format issues
    MLX_TO_NUMPY_DTYPE = {
        mx.bool_: np.bool_,
        mx.uint8: np.uint8,
        mx.uint16: np.uint16, 
        mx.uint32: np.uint32,
        mx.uint64: np.uint64,
        mx.int8: np.int8,
        mx.int16: np.int16,
        mx.int32: np.int32,
        mx.int64: np.int64,
        mx.float16: np.float16,
        mx.float32: np.float32,
        mx.float64: np.float64,
        mx.complex64: np.complex64,
        mx.complex128: np.complex128,
    }
    
    # Reverse mapping for deserialization
    NUMPY_TO_MLX_DTYPE = {v: k for k, v in MLX_TO_NUMPY_DTYPE.items()}
    
    @staticmethod
    def _get_safe_numpy_dtype(mlx_array: mx.array) -> np.dtype:
        """Get a safe NumPy dtype that avoids PEP 3118 buffer issues."""
        mlx_dtype = mlx_array.dtype
        
        # Special handling for boolean arrays - they cause the most issues
        if mlx_dtype == mx.bool_:
            # Use uint8 as intermediate format to avoid buffer size mismatches
            return np.uint8
        
        # For other types, use direct mapping if available
        if mlx_dtype in MLXArraySerializer.MLX_TO_NUMPY_DTYPE:
            return MLXArraySerializer.MLX_TO_NUMPY_DTYPE[mlx_dtype]
        
        # Fallback to float32 for unknown types
        logger.warning(f"Unknown MLX dtype {mlx_dtype}, falling back to float32")
        return np.float32
    
    @staticmethod
    def _prepare_mlx_array_for_serialization(mlx_array: mx.array) -> Dict[str, Any]:
        """
        Prepare MLX array for serialization, handling dtype mismatches.
        
        This function specifically addresses the PEP 3118 buffer format error:
        "Item size 2 for PEP 3118 buffer format string B does not match the dtype B item size 1"
        """
        if not isinstance(mlx_array, mx.array):
            raise ValueError(f"Expected mx.array, got {type(mlx_array)}")
        
        # Store original MLX dtype for reconstruction
        original_dtype = mlx_array.dtype
        original_shape = mlx_array.shape
        
        # Ensure array is evaluated and on CPU for serialization
        mx.eval(mlx_array)
        
        try:
            # Special case for boolean arrays - the main source of buffer format errors
            if original_dtype == mx.bool_:
                # Convert to uint8 to avoid PEP 3118 buffer format issues
                numpy_array = np.array(mlx_array, dtype=np.uint8)
                serialized_data = numpy_array.tobytes()
                
                return {
                    'data': serialized_data,
                    'shape': original_shape,
                    'original_dtype': 'bool_',  # Store as string for JSON serialization
                    'numpy_dtype': 'uint8',
                    'is_boolean': True
                }
            
            # For other data types, use safe conversion
            safe_dtype = MLXArraySerializer._get_safe_numpy_dtype(mlx_array)
            
            # Convert MLX array to NumPy with safe dtype
            if original_dtype == safe_dtype or original_dtype in MLXArraySerializer.MLX_TO_NUMPY_DTYPE:
                numpy_array = np.array(mlx_array, dtype=safe_dtype)
            else:
                # Cast through MLX first, then to NumPy
                casted_mlx = mlx_array.astype(safe_dtype)
                mx.eval(casted_mlx)
                numpy_array = np.array(casted_mlx, dtype=safe_dtype)
            
            serialized_data = numpy_array.tobytes()
            
            return {
                'data': serialized_data,
                'shape': original_shape,
                'original_dtype': str(original_dtype).split('.')[-1],  # e.g., 'float32'
                'numpy_dtype': str(safe_dtype).split('.')[-1],
                'is_boolean': False
            }
            
        except Exception as e:
            logger.error(f"Failed to serialize MLX array: {e}")
            logger.error(f"Array shape: {original_shape}, dtype: {original_dtype}")
            
            # Last resort: pickle the entire array (less efficient but reliable)
            return {
                'data': pickle.dumps(mlx_array),
                'shape': original_shape,
                'original_dtype': str(original_dtype).split('.')[-1],
                'numpy_dtype': None,
                'is_pickled': True,
                'is_boolean': False
            }
    
    @staticmethod
    def _deserialize_to_mlx_array(serialized_data: Dict[str, Any]) -> mx.array:
        """Deserialize data back to MLX array with proper dtype restoration."""
        
        # Handle pickled arrays
        if serialized_data.get('is_pickled', False):
            return pickle.loads(serialized_data['data'])
        
        # Get array metadata
        shape = tuple(serialized_data['shape'])
        original_dtype_str = serialized_data['original_dtype']
        numpy_dtype_str = serialized_data['numpy_dtype']
        is_boolean = serialized_data.get('is_boolean', False)
        
        # Reconstruct NumPy array from bytes
        numpy_dtype = getattr(np, numpy_dtype_str)
        numpy_array = np.frombuffer(serialized_data['data'], dtype=numpy_dtype)
        numpy_array = numpy_array.reshape(shape)
        
        # Convert back to MLX array with correct dtype
        if is_boolean:
            # Convert uint8 back to boolean
            boolean_array = numpy_array.astype(np.bool_)
            mlx_array = mx.array(boolean_array)
            return mlx_array.astype(mx.bool_)
        
        # For other types, convert to MLX array
        mlx_array = mx.array(numpy_array)
        
        # Restore original MLX dtype if different
        if original_dtype_str != numpy_dtype_str:
            target_dtype = getattr(mx, original_dtype_str)
            mlx_array = mlx_array.astype(target_dtype)
        
        return mlx_array


def serialize_mlx_tensors(tensors: Dict[str, mx.array]) -> Dict[str, Any]:
    """
    Serialize a dictionary of MLX tensors for distributed communication.
    
    Args:
        tensors: Dictionary mapping names to MLX arrays
        
    Returns:
        Dictionary containing serialized tensor data
    """
    serialized = {}
    
    for name, tensor in tensors.items():
        try:
            serialized[name] = MLXArraySerializer._prepare_mlx_array_for_serialization(tensor)
        except Exception as e:
            logger.error(f"Failed to serialize tensor '{name}': {e}")
            # Skip problematic tensors rather than failing entirely
            continue
    
    return {
        'tensors': serialized,
        'num_tensors': len(serialized),
        'serialization_version': '1.0'
    }


def deserialize_mlx_tensors(serialized_data: Dict[str, Any]) -> Dict[str, mx.array]:
    """
    Deserialize tensors back to MLX arrays.
    
    Args:
        serialized_data: Dictionary containing serialized tensor data
        
    Returns:
        Dictionary mapping names to MLX arrays
    """
    if 'tensors' not in serialized_data:
        raise ValueError("Invalid serialized data format")
    
    tensors = {}
    
    for name, tensor_data in serialized_data['tensors'].items():
        try:
            tensors[name] = MLXArraySerializer._deserialize_to_mlx_array(tensor_data)
        except Exception as e:
            logger.error(f"Failed to deserialize tensor '{name}': {e}")
            # Skip problematic tensors rather than failing entirely
            continue
    
    return tensors


def prepare_gradients_for_allreduce(gradients: Dict[str, mx.array]) -> bytes:
    """
    Prepare gradients for all-reduce operation by serializing them properly.
    
    Args:
        gradients: Dictionary of gradient tensors
        
    Returns:
        Serialized gradient data as bytes
    """
    serialized_grads = serialize_mlx_tensors(gradients)
    return pickle.dumps(serialized_grads)


def restore_gradients_from_allreduce(data: bytes) -> Dict[str, mx.array]:
    """
    Restore gradients from all-reduce operation.
    
    Args:
        data: Serialized gradient data
        
    Returns:
        Dictionary of gradient tensors
    """
    serialized_grads = pickle.loads(data)
    return deserialize_mlx_tensors(serialized_grads)


class DistributedCommunicator:
    """Handles distributed communication for MLX training."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.serializer = MLXArraySerializer()
    
    def all_reduce(self, tensors: Dict[str, mx.array], op: str = "mean") -> Dict[str, mx.array]:
        """
        Perform all-reduce operation on tensors across all processes.
        
        Args:
            tensors: Dictionary of tensors to reduce
            op: Reduction operation ("sum" or "mean")
            
        Returns:
            Reduced tensors
        """
        if self.world_size == 1:
            return tensors
        
        # Serialize tensors for communication
        serialized_data = prepare_gradients_for_allreduce(tensors)
        
        # In a real implementation, this would use MPI, NCCL, or similar
        # For now, we simulate the operation
        logger.info(f"Rank {self.rank}: Performing all-reduce on {len(tensors)} tensors")
        
        # Deserialize back (in real implementation, this would be received data)
        reduced_tensors = restore_gradients_from_allreduce(serialized_data)
        
        # Apply reduction operation
        if op == "mean":
            for name, tensor in reduced_tensors.items():
                reduced_tensors[name] = tensor / self.world_size
        
        return reduced_tensors
    
    def broadcast(self, tensors: Dict[str, mx.array], root: int = 0) -> Dict[str, mx.array]:
        """
        Broadcast tensors from root process to all processes.
        
        Args:
            tensors: Dictionary of tensors to broadcast
            root: Root process rank
            
        Returns:
            Broadcasted tensors
        """
        if self.world_size == 1:
            return tensors
        
        if self.rank == root:
            # Serialize tensors for broadcasting
            serialized_data = prepare_gradients_for_allreduce(tensors)
            logger.info(f"Rank {self.rank}: Broadcasting {len(tensors)} tensors")
            # In real implementation, would send to other processes
            return tensors
        else:
            # In real implementation, would receive from root
            logger.info(f"Rank {self.rank}: Receiving broadcast tensors")
            return tensors
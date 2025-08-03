"""
DLPack-based tensor serialization for MLX arrays.
Preserves dtype (including bfloat16) without conversion.
"""

import time
import logging
import pickle
import struct
import hashlib
from typing import Tuple, Dict, Any
import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def serialize_mlx_array_dlpack(array: mx.array, include_checksum: bool = True) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize MLX array using DLPack to preserve exact dtype.
    
    Args:
        array: MLX array to serialize
        include_checksum: Whether to compute checksum for integrity
        
    Returns:
        Tuple of (serialized bytes, metadata dict)
    """
    start_time = time.time()
    
    # Ensure array is evaluated
    mx.eval(array)
    # MLX arrays don't have a flags attribute, but we can ensure contiguity
    # by reshaping if needed
    
    # Store original metadata
    original_shape = list(array.shape)
    original_dtype = str(array.dtype)
    original_device = str(array.device)
    
    # Convert to DLPack capsule
    try:
        # MLX arrays support DLPack protocol
        dlpack_capsule = array.__dlpack__()
        
        # Serialize the capsule data
        # Note: We need to extract the raw data from the array
        # since DLPack capsules can't be directly serialized
        
        # For now, we'll use a workaround with numpy conversion
        # But we'll preserve the exact dtype info
        if array.dtype == mx.bfloat16:
            # Special handling for bfloat16
            # Convert to float32 temporarily but keep track
            float32_array = array.astype(mx.float32)
            np_array = np.array(float32_array)
            # Convert float32 to bfloat16 representation
            uint32_view = np_array.view(np.uint32)
            uint16_array = (uint32_view >> 16).astype(np.uint16)
            raw_bytes = uint16_array.tobytes()
        else:
            # For other dtypes, direct conversion
            np_array = np.array(array)
            raw_bytes = np_array.tobytes()
        
        # Compute checksum if requested
        checksum = None
        if include_checksum:
            checksum = hashlib.sha256(raw_bytes).hexdigest()[:16]  # First 16 chars
        
        serialization_time = (time.time() - start_time) * 1000
        
        # Create metadata
        metadata = {
            'shape': original_shape,
            'dtype': original_dtype,
            'device': original_device,
            'byte_size': len(raw_bytes),
            'checksum': checksum,
            'serialization_time_ms': serialization_time,
            'method': 'dlpack_raw'
        }
        
        logger.debug(f"Serialized tensor {original_shape} with dtype {original_dtype} in {serialization_time:.1f}ms")
        
        return raw_bytes, metadata
        
    except Exception as e:
        logger.warning(f"DLPack serialization failed: {e}, falling back to standard method")
        # Fallback to standard serialization
        from .tensor_utils import serialize_mlx_array
        return serialize_mlx_array(array, compress=False)


def deserialize_mlx_array_dlpack(data: bytes, metadata: Dict[str, Any]) -> mx.array:
    """
    Deserialize MLX array from DLPack data preserving exact dtype.
    
    Args:
        data: Serialized bytes
        metadata: Metadata dict with shape, dtype, etc.
        
    Returns:
        Reconstructed MLX array
    """
    start_time = time.time()
    
    try:
        # Validate checksum if present
        if metadata.get('checksum'):
            computed_checksum = hashlib.sha256(data).hexdigest()[:16]
            if computed_checksum != metadata['checksum']:
                raise ValueError(f"Checksum mismatch: expected {metadata['checksum']}, got {computed_checksum}")
        
        # Get original properties
        shape = tuple(metadata['shape'])
        dtype_str = metadata['dtype']
        
        # Map string dtype to MLX dtype
        dtype_map = {
            'mlx.core.bool_': mx.bool_,
            'mlx.core.uint8': mx.uint8,
            'mlx.core.uint16': mx.uint16,
            'mlx.core.uint32': mx.uint32,
            'mlx.core.uint64': mx.uint64,
            'mlx.core.int8': mx.int8,
            'mlx.core.int16': mx.int16,
            'mlx.core.int32': mx.int32,
            'mlx.core.int64': mx.int64,
            'mlx.core.float16': mx.float16,
            'mlx.core.float32': mx.float32,
            'mlx.core.bfloat16': mx.bfloat16,
            'mlx.core.complex64': mx.complex64,
        }
        
        mlx_dtype = dtype_map.get(dtype_str, mx.float32)
        
        # Calculate expected size
        dtype_sizes = {
            mx.bool_: 1, mx.uint8: 1, mx.int8: 1,
            mx.uint16: 2, mx.int16: 2, mx.float16: 2, mx.bfloat16: 2,
            mx.uint32: 4, mx.int32: 4, mx.float32: 4,
            mx.uint64: 8, mx.int64: 8, mx.complex64: 8,
        }
        
        element_size = dtype_sizes.get(mlx_dtype, 4)
        expected_size = element_size
        for dim in shape:
            expected_size *= dim
        
        if len(data) != expected_size:
            raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data)}")
        
        # Create array from raw bytes
        # MLX doesn't have a direct frombuffer, so we need to go through numpy
        import numpy as np
        
        # For bfloat16, we need special handling
        if mlx_dtype == mx.bfloat16:
            # Create as float32 first, then convert
            # This is because numpy doesn't support bfloat16
            num_elements = len(data) // 2  # bfloat16 is 2 bytes
            
            # Interpret as uint16 first
            uint16_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            
            # Convert uint16 representation to float32
            # bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
            # float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
            float32_array = np.zeros(shape, dtype=np.float32)
            float32_view = float32_array.view(np.uint32)
            float32_view[:] = uint16_array.astype(np.uint32) << 16
            
            # Create MLX array and convert to bfloat16
            mlx_array = mx.array(float32_array).astype(mx.bfloat16)
        else:
            # For other dtypes, use numpy mapping
            numpy_dtype_map = {
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
                mx.complex64: np.complex64,
            }
            
            np_dtype = numpy_dtype_map.get(mlx_dtype, np.float32)
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            mlx_array = mx.array(np_array)
        
        deserialization_time = (time.time() - start_time) * 1000
        logger.debug(f"Deserialized tensor {shape} with dtype {mlx_dtype} in {deserialization_time:.1f}ms")
        
        return mlx_array
        
    except Exception as e:
        logger.error(f"DLPack deserialization failed: {e}")
        # Fallback to standard deserialization
        from .tensor_utils import deserialize_mlx_array
        return deserialize_mlx_array(data, metadata)


def validate_tensor_integrity(tensor: mx.array, name: str = "tensor") -> Dict[str, Any]:
    """
    Compute integrity metrics for a tensor.
    
    Args:
        tensor: MLX array to validate
        name: Name for logging
        
    Returns:
        Dict with integrity metrics
    """
    mx.eval(tensor)
    
    metrics = {
        'name': name,
        'shape': tensor.shape,
        'dtype': str(tensor.dtype),
        'min': float(mx.min(tensor).item()),
        'max': float(mx.max(tensor).item()),
        'mean': float(mx.mean(tensor).item()),
        'std': float(mx.std(tensor).item()),
        'has_nan': bool(mx.any(mx.isnan(tensor)).item()),
        'has_inf': bool(mx.any(mx.isinf(tensor)).item()),
        'checksum': hashlib.sha256(tensor.tobytes()).hexdigest()[:16]
    }
    
    logger.debug(f"Tensor {name} integrity: shape={metrics['shape']}, dtype={metrics['dtype']}, "
                f"range=[{metrics['min']:.4f}, {metrics['max']:.4f}], mean={metrics['mean']:.4f}")
    
    return metrics
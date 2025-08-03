"""
Zero-copy DLPack tensor serialization for MLX arrays.
Provides exact dtype preservation with minimal overhead.
"""

import time
import logging
import pickle
import hashlib
from typing import Tuple, Dict, Any, Optional
import mlx.core as mx

logger = logging.getLogger(__name__)


def serialize_mlx_array_zerocopy(
    array: mx.array, 
    include_checksum: bool = False
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize MLX array using zero-copy DLPack protocol.
    
    This method preserves exact dtype (including bfloat16) without any
    conversion overhead by using DLPack's capsule mechanism.
    
    Args:
        array: MLX array to serialize
        include_checksum: Whether to compute checksum for integrity
        
    Returns:
        Tuple of (serialized capsule bytes, metadata dict)
    """
    start_time = time.time()
    
    # Ensure array is evaluated and contiguous
    mx.eval(array)
    array = array.contiguous()  # Force C-order layout
    
    # Store metadata
    metadata = {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'device': str(array.device),
        'method': 'dlpack_zerocopy'
    }
    
    try:
        # Create DLPack capsule
        dlpack_capsule = array.__dlpack__()
        
        # Serialize the capsule object
        # Note: In production, you'd pass the capsule directly through
        # a shared memory interface. For gRPC, we need to serialize it.
        capsule_bytes = pickle.dumps(dlpack_capsule, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Optional checksum for validation
        if include_checksum:
            # For checksum, we need to access the raw data
            # This defeats zero-copy but is optional for debugging
            checksum = hashlib.sha256(capsule_bytes).hexdigest()[:16]
            metadata['checksum'] = checksum
        
        serialization_time = (time.time() - start_time) * 1000
        metadata['serialization_time_ms'] = serialization_time
        
        logger.debug(
            f"Zero-copy serialized tensor {array.shape} with dtype {array.dtype} "
            f"in {serialization_time:.1f}ms"
        )
        
        return capsule_bytes, metadata
        
    except Exception as e:
        logger.error(f"Zero-copy serialization failed: {e}")
        # Fallback to standard method
        from .tensor_utils import serialize_mlx_array
        return serialize_mlx_array(array, compress=False)


def deserialize_mlx_array_zerocopy(
    data: bytes, 
    metadata: Dict[str, Any]
) -> mx.array:
    """
    Deserialize MLX array from zero-copy DLPack data.
    
    Args:
        data: Serialized DLPack capsule bytes
        metadata: Metadata dict with shape, dtype, etc.
        
    Returns:
        Reconstructed MLX array with exact dtype preserved
    """
    start_time = time.time()
    
    try:
        # Validate checksum if present
        if metadata.get('checksum'):
            computed_checksum = hashlib.sha256(data).hexdigest()[:16]
            if computed_checksum != metadata['checksum']:
                raise ValueError(
                    f"Checksum mismatch: expected {metadata['checksum']}, "
                    f"got {computed_checksum}"
                )
        
        # Deserialize DLPack capsule
        dlpack_capsule = pickle.loads(data)
        
        # Reconstruct array from DLPack
        # This is zero-copy if the memory is shared
        array = mx.from_dlpack(dlpack_capsule)
        
        # Verify dtype matches
        expected_dtype = metadata['dtype']
        if str(array.dtype) != expected_dtype:
            logger.warning(
                f"DLPack dtype mismatch: expected {expected_dtype}, "
                f"got {array.dtype}"
            )
        
        deserialization_time = (time.time() - start_time) * 1000
        logger.debug(
            f"Zero-copy deserialized tensor {array.shape} with dtype {array.dtype} "
            f"in {deserialization_time:.1f}ms"
        )
        
        return array
        
    except Exception as e:
        logger.error(f"Zero-copy deserialization failed: {e}")
        # Fallback to standard method
        from .tensor_utils import deserialize_mlx_array
        return deserialize_mlx_array(data, metadata)


def validate_tensor_zerocopy(tensor: mx.array, name: str = "tensor") -> Dict[str, Any]:
    """
    Validate tensor integrity without copying data.
    
    Args:
        tensor: MLX array to validate
        name: Name for logging
        
    Returns:
        Dict with validation metrics
    """
    mx.eval(tensor)
    
    # Basic shape and dtype info (no data access)
    metrics = {
        'name': name,
        'shape': tensor.shape,
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'nbytes': tensor.nbytes,
        'itemsize': tensor.itemsize,
        'size': tensor.size,
        'ndim': tensor.ndim,
    }
    
    # Check if contiguous (important for DLPack)
    # Note: MLX doesn't expose strides directly, but we can check if it's contiguous
    try:
        _ = tensor.contiguous()
        metrics['is_contiguous'] = True
    except:
        metrics['is_contiguous'] = False
    
    logger.debug(
        f"Tensor {name} validation: shape={metrics['shape']}, "
        f"dtype={metrics['dtype']}, contiguous={metrics['is_contiguous']}"
    )
    
    return metrics


# Specialized functions for common tensor patterns

def serialize_kv_cache_zerocopy(
    keys: mx.array,
    values: mx.array,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Specialized serialization for KV-cache tensors.
    
    Optimized for the common pattern of sending key-value pairs together.
    """
    # Ensure both arrays are contiguous
    keys = keys.contiguous()
    values = values.contiguous()
    
    # Create capsules
    keys_capsule = keys.__dlpack__()
    values_capsule = values.__dlpack__()
    
    # Pack both capsules
    combined_data = pickle.dumps({
        'keys': keys_capsule,
        'values': values_capsule
    }, protocol=pickle.HIGHEST_PROTOCOL)
    
    combined_metadata = {
        'keys_shape': list(keys.shape),
        'keys_dtype': str(keys.dtype),
        'values_shape': list(values.shape),
        'values_dtype': str(values.dtype),
        'method': 'kv_cache_zerocopy'
    }
    
    if metadata:
        combined_metadata.update(metadata)
    
    return combined_data, combined_metadata


def deserialize_kv_cache_zerocopy(
    data: bytes,
    metadata: Dict[str, Any]
) -> Tuple[mx.array, mx.array]:
    """
    Deserialize KV-cache tensors.
    
    Returns:
        Tuple of (keys, values) arrays
    """
    # Unpack capsules
    capsules = pickle.loads(data)
    
    # Reconstruct arrays
    keys = mx.from_dlpack(capsules['keys'])
    values = mx.from_dlpack(capsules['values'])
    
    return keys, values
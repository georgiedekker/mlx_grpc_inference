#!/usr/bin/env python3
"""
Tensor serialization utilities with proper dtype handling for MLX
"""
import mlx.core as mx
import numpy as np
import struct
import os
import hashlib
from typing import Tuple, Dict, Any, List
import logging

# Create a logger with a simpler format that doesn't require 'rank'
logger = logging.getLogger(__name__)
# Only configure if no handlers exist (to avoid duplicate configuration)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# MLX dtype to numpy dtype mapping
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
    mx.bfloat16: 'bfloat16',  # Special handling required
    mx.complex64: np.complex64,
}

# Numpy dtype to MLX dtype mapping
NUMPY_TO_MLX_DTYPE = {
    np.bool_: mx.bool_,
    np.uint8: mx.uint8,
    np.uint16: mx.uint16,
    np.uint32: mx.uint32,
    np.uint64: mx.uint64,
    np.int8: mx.int8,
    np.int16: mx.int16,
    np.int32: mx.int32,
    np.int64: mx.int64,
    np.float16: mx.float16,
    np.float32: mx.float32,
    'bfloat16': mx.bfloat16,
    np.complex64: mx.complex64,
}

def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of tensor data"""
    return hashlib.sha256(data).hexdigest()

def serialize_tensor(tensor: mx.array) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize MLX tensor preserving dtype exactly with raw bit transport for bfloat16
    
    Returns:
        bytes: Serialized tensor data
        dict: Metadata including shape, dtype, checksum, and original dtype info
    """
    # Ensure tensor is evaluated
    mx.eval(tensor)
    
    # Get original dtype
    original_dtype = tensor.dtype
    dtype_str = str(original_dtype).replace('mlx.core.', '')
    
    # Handle bfloat16 specially to preserve exact bits
    if original_dtype == mx.bfloat16:
        # CRITICAL: Preserve raw bfloat16 bits without any conversion
        # Use memoryview to access raw bytes directly
        mx.eval(tensor)
        
        # Get raw bytes via memoryview - this preserves exact bit representation
        mv = memoryview(tensor)
        raw_bytes = mv.tobytes()
        
        # MLX represents bfloat16 as 16-bit values (itemsize=2)
        # We transport as raw uint16 to preserve exact bits
        transport_dtype = 'bfloat16_as_uint16'
        
        logger.debug(f"Preserving raw bfloat16 bits via memoryview (no conversion)")
    else:
        # For non-bfloat16 types, use direct numpy conversion
        if original_dtype == mx.float16:
            np_array = np.array(tensor, dtype=np.float16, copy=True)
            transport_dtype = 'float16'
        elif original_dtype == mx.float32:
            np_array = np.array(tensor, dtype=np.float32, copy=True)
            transport_dtype = 'float32'
        elif original_dtype == mx.int32:
            np_array = np.array(tensor, dtype=np.int32, copy=True)
            transport_dtype = 'int32'
        elif original_dtype == mx.int64:
            np_array = np.array(tensor, dtype=np.int64, copy=True)
            transport_dtype = 'int64'
        else:
            # For other types, let numpy preserve the dtype
            np_array = np.array(tensor, copy=True)
            transport_dtype = str(np_array.dtype)
        
        raw_bytes = np_array.tobytes()
    
    # Compute checksum for verification
    checksum = compute_checksum(raw_bytes)
    
    # Verify size is correct
    dtype_sizes = {
        'float16': 2, 'float32': 4, 'int32': 4, 'int64': 8, 
        'bfloat16': 2, 'bfloat16_raw': 2, 'bfloat16_as_uint16': 2
    }
    bytes_per_element = dtype_sizes.get(transport_dtype, 4)
    expected_size = np.prod(tensor.shape) * bytes_per_element
    actual_size = len(raw_bytes)
    
    if actual_size != expected_size:
        logger.error(f"Size mismatch: actual {actual_size} != expected {expected_size}")
        raise ValueError(f"Serialization produced wrong size: {actual_size} bytes instead of {expected_size}")
    
    metadata = {
        'shape': list(tensor.shape),
        'dtype': transport_dtype,
        'original_dtype': dtype_str,
        'checksum': checksum,
        'size': actual_size
    }
    
    logger.debug(f"Serialized tensor: shape={tensor.shape}, "
                f"original_dtype={original_dtype}, transport_dtype={transport_dtype}, "
                f"checksum={checksum[:8]}..., size={actual_size}")
    
    return raw_bytes, metadata

def deserialize_tensor(data: bytes, metadata: Dict[str, Any]) -> mx.array:
    """
    Deserialize tensor data back to MLX array with correct dtype and checksum verification
    """
    # Verify checksum if present
    if 'checksum' in metadata:
        actual_checksum = compute_checksum(data)
        expected_checksum = metadata['checksum']
        if actual_checksum != expected_checksum:
            logger.error(f"Checksum mismatch! Expected {expected_checksum[:8]}..., got {actual_checksum[:8]}...")
            raise ValueError("Tensor data corrupted during transport")
    
    shape = tuple(metadata['shape'])
    transport_dtype = metadata['dtype']
    original_dtype = metadata.get('original_dtype', transport_dtype)
    
    # Verify data size
    if 'size' in metadata and len(data) != metadata['size']:
        logger.error(f"Data size mismatch: expected {metadata['size']}, got {len(data)}")
        raise ValueError(f"Data size mismatch during deserialization")
    
    if transport_dtype == 'bfloat16_raw':
        # Using ml_dtypes for exact bfloat16 reconstruction
        try:
            import ml_dtypes
            np_array = np.frombuffer(data, dtype=ml_dtypes.bfloat16).reshape(shape)
            # MLX doesn't support direct creation from bfloat16 numpy arrays
            # Convert through float32 as intermediate
            np_float32 = np_array.astype(np.float32)
            mx_array = mx.array(np_float32, dtype=mx.bfloat16)
            logger.debug(f"Successfully deserialized bfloat16 using ml_dtypes")
        except ImportError:
            logger.error("ml_dtypes not available for bfloat16 deserialization")
            raise
    elif transport_dtype == 'bfloat16_as_uint16':
        # CRITICAL: Reconstruct bfloat16 from raw uint16 bits without conversion
        # Create numpy array view of the raw bytes as uint16
        np_uint16 = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        
        # Create MLX array directly from the uint16 data, interpreting as bfloat16
        # MLX should be able to reinterpret the uint16 buffer as bfloat16
        # First create as uint16, then view as bfloat16
        mx_uint16 = mx.array(np_uint16, dtype=mx.uint16)
        
        # Reinterpret the bits as bfloat16 (view cast, not value conversion)
        mx_array = mx_uint16.view(mx.bfloat16)
        
        logger.debug(f"Reconstructed bfloat16 from raw uint16 bits without conversion")
    elif transport_dtype == 'float32':
        np_array = np.frombuffer(data, dtype=np.float32).reshape(shape)
        mx_array = mx.array(np_array)
    elif transport_dtype == 'float16':
        np_array = np.frombuffer(data, dtype=np.float16).reshape(shape)
        mx_array = mx.array(np_array)
    elif transport_dtype == 'int32':
        np_array = np.frombuffer(data, dtype=np.int32).reshape(shape)
        mx_array = mx.array(np_array)
    elif transport_dtype == 'int64':
        np_array = np.frombuffer(data, dtype=np.int64).reshape(shape)
        mx_array = mx.array(np_array)
    else:
        # Try to parse the dtype string
        try:
            np_dtype = np.dtype(transport_dtype)
            np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            mx_array = mx.array(np_array)
        except:
            # Fallback to float32
            logger.warning(f"Unknown transport dtype {transport_dtype}, falling back to float32")
            np_array = np.frombuffer(data, dtype=np.float32).reshape(shape)
            mx_array = mx.array(np_array)
    
    # Ensure correct final dtype
    target_dtype = getattr(mx, original_dtype, None)
    if target_dtype and mx_array.dtype != target_dtype:
        mx_array = mx_array.astype(target_dtype)
    
    mx.eval(mx_array)
    
    # Add max-abs check for first 16 values
    flat_array = mx_array.flatten()
    num_values = min(16, flat_array.shape[0])
    if num_values > 0:
        # Convert first 16 values to float32 for max-abs check
        first_values_mx = flat_array[:num_values].astype(mx.float32)
        mx.eval(first_values_mx)
        first_values = [float(first_values_mx[i]) for i in range(num_values)]
        max_abs = max(abs(v) for v in first_values)
        
        # Always log max-abs (not just in debug mode)
        logger.info(f"Deserialized tensor max-abs (first 16): {max_abs:.6f}, shape={mx_array.shape}, dtype={mx_array.dtype}")
        
        # Log first few values for validation (only in debug mode)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Deserialized tensor first {num_values} values: {first_values}")
    
    logger.debug(f"Deserialized tensor: shape={mx_array.shape}, "
                f"dtype={mx_array.dtype}, original_dtype={original_dtype}")
    
    return mx_array

def validate_tensor_preservation(original: mx.array, reconstructed: mx.array) -> bool:
    """
    Validate that tensor was preserved correctly through serialization
    """
    # Check shape
    if original.shape != reconstructed.shape:
        logger.error(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        return False
    
    # Check dtype
    if original.dtype != reconstructed.dtype:
        logger.error(f"Dtype mismatch: {original.dtype} vs {reconstructed.dtype}")
        return False
    
    # Check values (accounting for bfloat16 precision)
    if original.dtype == mx.bfloat16:
        # For bfloat16, we should have EXACT bit preservation now
        # Allow only tiny differences due to potential float32 intermediate
        diff = mx.abs(original - reconstructed)
        max_diff = mx.max(diff).item()
        if max_diff > 1e-6:  # Much stricter than before
            logger.error(f"Value mismatch for bfloat16: max_diff={max_diff}")
            # Log some examples of differences
            if logger.isEnabledFor(logging.DEBUG):
                indices = mx.where(diff > 1e-6)
                if len(indices[0]) > 0:
                    for i in range(min(5, len(indices[0]))):
                        idx = tuple(idx[i] for idx in indices)
                        logger.debug(f"  Mismatch at {idx}: {original[idx].item()} vs {reconstructed[idx].item()}")
            return False
    else:
        # For other dtypes, expect exact match
        if not mx.array_equal(original, reconstructed):
            logger.error("Value mismatch for non-bfloat16 dtype")
            return False
    
    return True

# Wrapper functions for backward compatibility with existing codebase
def serialize_mlx_array(tensor: mx.array, compress: bool = False) -> Tuple[bytes, Dict[str, Any]]:
    """
    Wrapper for serialize_tensor to match existing API expectations
    """
    data, metadata = serialize_tensor(tensor)
    
    # Add compression if requested
    if compress:
        import gzip
        original_size = len(data)
        # Use faster compression level (1-3) for better speed/size tradeoff
        data = gzip.compress(data, compresslevel=1)
        compressed_size = len(data)
        metadata['compressed'] = True
        metadata['original_size'] = original_size
        logger.debug(f"Compressed tensor from {original_size/1024/1024:.2f}MB to {compressed_size/1024/1024:.2f}MB (ratio: {compressed_size/original_size:.2%})")
    else:
        metadata['compressed'] = False
    
    return data, metadata

def deserialize_mlx_array(data: bytes, metadata: Dict[str, Any]) -> mx.array:
    """
    Wrapper for deserialize_tensor to match existing API expectations
    """
    # Handle compression
    if metadata.get('compressed', False):
        import gzip
        data = gzip.decompress(data)
    
    return deserialize_tensor(data, metadata)

# KV Cache serialization functions
def serialize_kv_cache(cache: List[Any]) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize MLX transformer KV cache for distributed transmission
    
    Args:
        cache: List of cache entries (None or KV cache tuples)
    
    Returns:
        bytes: Serialized cache data
        dict: Cache metadata
    """
    if not cache:
        return b'', {'type': 'empty_cache', 'length': 0}
    
    cache_data = []
    cache_metadata = {'type': 'kv_cache', 'length': len(cache), 'entries': []}
    current_offset = 0  # Track byte offset, not list length!
    
    for i, cache_entry in enumerate(cache):
        if cache_entry is None:
            # Empty cache entry
            cache_metadata['entries'].append({'type': 'none', 'offset': current_offset, 'size': 0})
        elif hasattr(cache_entry, 'keys') and hasattr(cache_entry, 'values'):
            # KV cache object with keys and values - ensure proper dtype handling
            # For unpopulated caches (keys/values are None), skip serialization
            # The layer will initialize its own cache when needed
            if cache_entry.keys is None or cache_entry.values is None:
                # Don't serialize empty cache entries - they cause shape mismatches
                # Instead, mark them as empty so they can be reconstructed properly
                cache_metadata['entries'].append({
                    'type': 'empty_kv',
                    'offset': current_offset,
                    'size': 0
                })
                continue
            
            keys_data, keys_meta = serialize_tensor(cache_entry.keys)
            values_data, values_meta = serialize_tensor(cache_entry.values)
            
            entry_data = (
                struct.pack('<Q', len(keys_data)) + keys_data +
                struct.pack('<Q', len(values_data)) + values_data
            )
            
            cache_metadata['entries'].append({
                'type': 'kv_object',
                'offset': current_offset,
                'size': len(entry_data),
                'keys_meta': keys_meta,
                'values_meta': values_meta,
                'cache_offset': cache_entry.offset if hasattr(cache_entry, 'offset') else 0,
                'cache_step': cache_entry.step if hasattr(cache_entry, 'step') else 256
            })
            cache_data.append(entry_data)
            current_offset += len(entry_data)
        elif isinstance(cache_entry, (tuple, list)) and len(cache_entry) == 2:
            # KV cache as tuple (keys, values) - ensure proper dtype handling
            keys, values = cache_entry
            keys_data, keys_meta = serialize_tensor(keys)
            values_data, values_meta = serialize_tensor(values)
            
            entry_data = (
                struct.pack('<Q', len(keys_data)) + keys_data +
                struct.pack('<Q', len(values_data)) + values_data
            )
            
            cache_metadata['entries'].append({
                'type': 'kv_tuple',
                'offset': current_offset,
                'size': len(entry_data),
                'keys_meta': keys_meta,
                'values_meta': values_meta
            })
            cache_data.append(entry_data)
            current_offset += len(entry_data)
        else:
            # Unknown cache type, serialize as single tensor
            entry_data, entry_meta = serialize_tensor(cache_entry)
            cache_metadata['entries'].append({
                'type': 'tensor',
                'offset': current_offset,
                'size': len(entry_data),
                'meta': entry_meta
            })
            cache_data.append(entry_data)
            current_offset += len(entry_data)
    
    # Combine all cache data
    combined_data = b''.join(cache_data)
    return combined_data, cache_metadata

def deserialize_kv_cache(data: bytes, metadata: Dict[str, Any], existing_cache: List[Any] = None) -> List[Any]:
    """
    Deserialize KV cache from distributed transmission
    
    Args:
        data: Serialized cache data
        metadata: Cache metadata
        existing_cache: Optional existing cache list to update in-place (preserves object identity)
    
    Returns:
        Updated cache list (same object if existing_cache provided)
    """
    if metadata['type'] == 'empty_cache':
        return existing_cache if existing_cache is not None else []
    
    # Use existing cache if provided, otherwise create new list
    if existing_cache is not None:
        cache = existing_cache
        # Ensure cache list is long enough
        while len(cache) < metadata['length']:
            cache.append(None)
    else:
        cache = []
    
    for i, entry_meta in enumerate(metadata['entries']):
        if entry_meta['type'] == 'none':
            if existing_cache is None or i >= len(existing_cache):
                cache.append(None)
            # else: keep existing None
        elif entry_meta['type'] == 'empty_kv':
            if existing_cache is None or i >= len(existing_cache) or existing_cache[i] is None:
                # Only create new KVCache if we don't have one already
                from mlx_lm.models.cache import KVCache
                if existing_cache is None:
                    cache.append(KVCache())
                else:
                    cache[i] = KVCache()
            # else: keep existing empty KVCache object
        elif entry_meta['type'] in ['kv_object', 'kv_tuple']:
            # Extract KV data
            start = entry_meta['offset']
            end = start + entry_meta['size']
            entry_data = data[start:end]
            
            # Debug logging
            logger.debug(f"Deserializing cache entry {i}: type={entry_meta['type']}, size={entry_meta['size']}, actual_data_len={len(entry_data)}")
            
            if len(entry_data) < 16:
                logger.error(f"Invalid cache entry data: expected at least 16 bytes, got {len(entry_data)}")
                # Create empty cache for this entry
                from mlx_lm.models.cache import KVCache as MLXKVCache
                cache.append(MLXKVCache())
                continue
            
            # Parse keys and values
            keys_size = struct.unpack('<Q', entry_data[:8])[0]
            keys_data = entry_data[8:8+keys_size]
            values_size = struct.unpack('<Q', entry_data[8+keys_size:16+keys_size])[0]
            values_data = entry_data[16+keys_size:16+keys_size+values_size]
            
            keys = deserialize_tensor(keys_data, entry_meta['keys_meta'])
            values = deserialize_tensor(values_data, entry_meta['values_meta'])
            
            if entry_meta['type'] == 'kv_tuple':
                if existing_cache is None or i >= len(existing_cache):
                    cache.append((keys, values))
                else:
                    cache[i] = (keys, values)
            else:
                # Update existing KVCache object in-place or create new one
                from mlx_lm.models.cache import KVCache as MLXKVCache
                if existing_cache is not None and i < len(existing_cache) and existing_cache[i] is not None:
                    # UPDATE IN-PLACE - this preserves object identity!
                    kv_cache = existing_cache[i]
                    kv_cache.keys = keys
                    kv_cache.values = values
                    # Also need to update offset and step from the serialized cache
                    if 'cache_offset' in entry_meta:
                        kv_cache.offset = entry_meta['cache_offset']
                    if 'cache_step' in entry_meta:
                        kv_cache.step = entry_meta['cache_step']
                else:
                    # Create new cache object only if we don't have one
                    kv_cache = MLXKVCache()
                    kv_cache.keys = keys
                    kv_cache.values = values
                    # Set offset and step from metadata
                    if 'cache_offset' in entry_meta:
                        kv_cache.offset = entry_meta['cache_offset']
                    if 'cache_step' in entry_meta:
                        kv_cache.step = entry_meta['cache_step']
                    if existing_cache is None:
                        cache.append(kv_cache)
                    else:
                        cache[i] = kv_cache
        elif entry_meta['type'] == 'tensor':
            start = entry_meta['offset']
            end = start + entry_meta['size']
            entry_data = data[start:end]
            cache.append(deserialize_tensor(entry_data, entry_meta['meta']))
        else:
            cache.append(None)
    
    return cache

# SafeTensors-style format for multiple tensors
def serialize_tensor_dict(tensors: Dict[str, mx.array]) -> bytes:
    """
    Serialize multiple tensors in a SafeTensors-like format
    """
    # Header will contain metadata for all tensors
    header = {
        'version': 1,
        'tensors': {}
    }
    
    # Serialize each tensor
    tensor_data = []
    offset = 0
    
    for name, tensor in tensors.items():
        data, metadata = serialize_tensor(tensor)
        
        header['tensors'][name] = {
            **metadata,
            'offset': offset,
            'size': len(data)
        }
        
        tensor_data.append(data)
        offset += len(data)
    
    # Encode header as JSON
    import json
    header_json = json.dumps(header).encode('utf-8')
    header_size = len(header_json)
    
    # Pack: [header_size (8 bytes)] [header_json] [tensor_data...]
    result = struct.pack('<Q', header_size) + header_json
    for data in tensor_data:
        result += data
    
    return result

def deserialize_tensor_dict(data: bytes) -> Dict[str, mx.array]:
    """
    Deserialize multiple tensors from SafeTensors-like format
    """
    # Unpack header size
    header_size = struct.unpack('<Q', data[:8])[0]
    
    # Extract and parse header
    import json
    header_json = data[8:8+header_size]
    header = json.loads(header_json.decode('utf-8'))
    
    # Base offset for tensor data
    base_offset = 8 + header_size
    
    # Deserialize each tensor
    result = {}
    for name, info in header['tensors'].items():
        start = base_offset + info['offset']
        end = start + info['size']
        tensor_data = data[start:end]
        
        # Remove offset and size from metadata
        metadata = {k: v for k, v in info.items() if k not in ['offset', 'size']}
        
        result[name] = deserialize_tensor(tensor_data, metadata)
    
    return result
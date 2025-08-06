"""
Utilities for serializing and deserializing MLX tensors.
"""

import mlx.core as mx
import numpy as np
import pickle
import gzip
import lz4.frame
import zstandard as zstd
from typing import Tuple, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


def serialize_mlx_array(array: mx.array, 
                       compress: bool = False, 
                       compression_algorithm: str = "auto") -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize an MLX array to bytes with optimized compression.
    
    Args:
        array: MLX array to serialize
        compress: Whether to compress the data
        compression_algorithm: Compression algorithm ('auto', 'gzip', 'lz4', 'zstd', 'none')
        
    Returns:
        Tuple of (serialized bytes, metadata dict)
    """
    start_time = time.time()
    
    # Convert to numpy for serialization
    # First ensure the array is evaluated and synchronized
    mx.eval(array)
    mx.synchronize()  # CRITICAL: Ensure all operations complete before serialization
    
    # Store original dtype for proper reconstruction
    original_dtype = str(array.dtype)
    logger.debug(f"Serializing MLX array with dtype: {array.dtype}, shape: {array.shape}")
    
    # CRITICAL: Handle bfloat16 specially - NumPy doesn't support it natively
    if array.dtype == mx.bfloat16:
        # Convert to float32 for serialization (NumPy doesn't support bfloat16)
        logger.debug("Converting bfloat16 to float32 for serialization")
        array = array.astype(mx.float32)
        np_array = np.array(array, copy=True)
    else:
        # For other dtypes, preserve exact representation
        np_array = np.array(array, copy=True)  # Force copy to ensure data is materialized
    
    logger.debug(f"NumPy array dtype: {np_array.dtype}, shape: {np_array.shape}")
    
    # Serialize to bytes first
    raw_data = pickle.dumps(np_array)
    original_size = len(raw_data)
    
    # Initialize compression metadata
    compression_info = {
        'algorithm': 'none',
        'ratio': 1.0,
        'time_ms': 0.0
    }
    
    # Apply compression if requested
    compressed_data = raw_data
    if compress:
        if compression_algorithm == "auto":
            # Choose best compression algorithm based on data characteristics
            compression_algorithm = _select_optimal_compression(np_array, original_size)
        
        if compression_algorithm != "none":
            compressed_data, compression_info = _compress_data(raw_data, compression_algorithm)
    
    serialization_time = (time.time() - start_time) * 1000
    
    # Create metadata
    metadata = {
        'shape': list(np_array.shape),
        'dtype': str(np_array.dtype),
        'mlx_dtype': original_dtype,  # CRITICAL: Store MLX dtype for exact reconstruction
        'compressed': compress,
        'original_size': original_size,
        'compressed_size': len(compressed_data),
        'compression_info': compression_info,
        'serialization_time_ms': serialization_time
    }
    
    logger.debug(f"Serialized tensor {array.shape} ({original_size} -> {len(compressed_data)} bytes, "
                f"{compression_info['ratio']:.2f}x ratio, {serialization_time:.1f}ms)")
    
    return compressed_data, metadata


def deserialize_mlx_array(data: bytes, metadata: Dict[str, Any]) -> mx.array:
    """
    Deserialize bytes to an MLX array with optimized decompression.
    
    Args:
        data: Serialized array data
        metadata: Metadata about the array
        
    Returns:
        MLX array
    """
    start_time = time.time()
    
    # Decompress if needed
    if metadata.get('compressed', False):
        compression_info = metadata.get('compression_info', {})
        algorithm = compression_info.get('algorithm', 'gzip')
        
        if algorithm == 'gzip':
            data = gzip.decompress(data)
        elif algorithm == 'lz4':
            data = lz4.frame.decompress(data)
        elif algorithm == 'zstd':
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(data)
        else:
            logger.warning(f"Unknown compression algorithm: {algorithm}, trying gzip")
            data = gzip.decompress(data)
    
    # Deserialize numpy array
    np_array = pickle.loads(data)
    
    # Convert back to MLX with the ORIGINAL dtype
    mlx_array = mx.array(np_array)
    
    # CRITICAL: Restore original MLX dtype if needed
    if 'mlx_dtype' in metadata:
        mlx_dtype_str = metadata['mlx_dtype']
        
        # Special handling for bfloat16 (was converted to float32 for serialization)
        if mlx_dtype_str == 'bfloat16':
            logger.debug(f"Restoring bfloat16 from {mlx_array.dtype}")
            mlx_array = mlx_array.astype(mx.bfloat16)
        elif str(mlx_array.dtype) != mlx_dtype_str:
            logger.warning(f"Dtype mismatch after deserialization: expected {mlx_dtype_str}, got {mlx_array.dtype}")
            # For other types, preserve what we got to avoid corruption
    
    # Verify shape
    expected_shape = tuple(metadata['shape'])
    if mlx_array.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {mlx_array.shape}")
    
    deserialization_time = (time.time() - start_time) * 1000
    logger.debug(f"Deserialized tensor {expected_shape} in {deserialization_time:.1f}ms")
    
    return mlx_array


def estimate_tensor_size(shape: Tuple[int, ...], dtype: str = 'float32') -> int:
    """
    Estimate the size of a tensor in bytes.
    
    Args:
        shape: Tensor shape
        dtype: Data type string
        
    Returns:
        Estimated size in bytes
    """
    # Map dtype strings to bytes per element
    dtype_sizes = {
        'float32': 4,
        'float16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'bool': 1
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    return num_elements * bytes_per_element


def split_large_tensor(array: mx.array, max_chunk_size: int = 50 * 1024 * 1024) -> list:
    """
    Split a large tensor into smaller chunks for transmission.
    
    Args:
        array: MLX array to split
        max_chunk_size: Maximum size per chunk in bytes
        
    Returns:
        List of array chunks
    """
    # Estimate total size
    total_size = estimate_tensor_size(array.shape, str(array.dtype))
    
    if total_size <= max_chunk_size:
        return [array]
    
    # Calculate how to split
    num_chunks = (total_size + max_chunk_size - 1) // max_chunk_size
    
    # Split along first dimension
    if len(array.shape) == 0:
        return [array]
    
    chunk_size = max(1, array.shape[0] // num_chunks)
    chunks = []
    
    for i in range(0, array.shape[0], chunk_size):
        end = min(i + chunk_size, array.shape[0])
        chunk = array[i:end]
        chunks.append(chunk)
    
    logger.debug(f"Split tensor {array.shape} into {len(chunks)} chunks")
    return chunks


def merge_tensor_chunks(chunks: list, axis: int = 0) -> mx.array:
    """
    Merge tensor chunks back into a single array.
    
    Args:
        chunks: List of array chunks
        axis: Axis along which to concatenate
        
    Returns:
        Merged MLX array
    """
    if len(chunks) == 1:
        return chunks[0]
    
    return mx.concatenate(chunks, axis=axis)


def _select_optimal_compression(np_array: np.ndarray, data_size: int) -> str:
    """
    Select optimal compression algorithm based on data characteristics.
    
    Args:
        np_array: NumPy array to analyze
        data_size: Size of serialized data in bytes
        
    Returns:
        Optimal compression algorithm name
    """
    # For very small tensors, compression overhead may not be worth it
    if data_size < 1024:  # 1KB
        return "none"
    
    # Analyze data characteristics
    dtype = np_array.dtype
    shape = np_array.shape
    
    # For floating point data, check for sparsity
    if dtype.kind == 'f':  # floating point
        # Calculate sparsity (percentage of near-zero values)
        sparsity = np.sum(np.abs(np_array) < 1e-6) / np_array.size
        
        if sparsity > 0.9:  # Very sparse
            return "zstd"  # Best compression for sparse data
        elif sparsity > 0.5:  # Moderately sparse
            return "gzip"  # Good compression with reasonable speed
        else:  # Dense data
            return "lz4"  # Fastest for dense data
    
    # For integer data
    elif dtype.kind in ['i', 'u']:  # signed or unsigned integer
        # Integer data often compresses well
        return "gzip"
    
    # For boolean data
    elif dtype.kind == 'b':
        return "zstd"  # Excellent compression for boolean arrays
    
    # For large tensors, prioritize speed
    if data_size > 10 * 1024 * 1024:  # 10MB
        return "lz4"
    
    # Default to balanced option
    return "gzip"


def _compress_data(data: bytes, algorithm: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress data using specified algorithm.
    
    Args:
        data: Raw data to compress
        algorithm: Compression algorithm
        
    Returns:
        Tuple of (compressed data, compression info)
    """
    start_time = time.time()
    original_size = len(data)
    
    try:
        if algorithm == "gzip":
            compressed = gzip.compress(data, compresslevel=6)  # Balanced speed/compression
        elif algorithm == "lz4":
            compressed = lz4.frame.compress(data, compression_level=4)  # Fast compression
        elif algorithm == "zstd":
            cctx = zstd.ZstdCompressor(level=3, threads=2)  # Balanced with threading
            compressed = cctx.compress(data)
        else:
            # Fallback to gzip
            compressed = gzip.compress(data, compresslevel=6)
            algorithm = "gzip"
        
        compression_time = (time.time() - start_time) * 1000
        compression_ratio = original_size / len(compressed) if len(compressed) > 0 else 1.0
        
        compression_info = {
            'algorithm': algorithm,
            'ratio': compression_ratio,
            'time_ms': compression_time,
            'original_size': original_size,
            'compressed_size': len(compressed)
        }
        
        logger.debug(f"Compressed {original_size} -> {len(compressed)} bytes using {algorithm} "
                    f"({compression_ratio:.2f}x, {compression_time:.1f}ms)")
        
        return compressed, compression_info
        
    except Exception as e:
        logger.warning(f"Compression with {algorithm} failed: {e}, using uncompressed data")
        return data, {
            'algorithm': 'none',
            'ratio': 1.0,
            'time_ms': 0.0,
            'original_size': original_size,
            'compressed_size': original_size,
            'error': str(e)
        }


def adaptive_serialize_mlx_array(array: mx.array, 
                                target_bandwidth_mbps: Optional[float] = None,
                                latency_budget_ms: Optional[float] = None) -> Tuple[bytes, Dict[str, Any]]:
    """
    Adaptively serialize MLX array based on network conditions.
    
    Args:
        array: MLX array to serialize
        target_bandwidth_mbps: Target bandwidth utilization
        latency_budget_ms: Available latency budget for compression
        
    Returns:
        Tuple of (serialized bytes, metadata dict)
    """
    # Analyze tensor characteristics
    np_array = np.array(array)
    raw_size = estimate_tensor_size(array.shape, str(array.dtype))
    
    # Decide whether to compress based on conditions
    should_compress = True
    compression_algorithm = "auto"
    
    # If latency budget is very tight, skip compression for small tensors
    if latency_budget_ms is not None and latency_budget_ms < 5.0 and raw_size < 1024 * 1024:
        should_compress = False
    
    # If bandwidth is abundant, prioritize speed over compression
    if target_bandwidth_mbps is not None and target_bandwidth_mbps > 100:
        if raw_size < 5 * 1024 * 1024:  # < 5MB
            should_compress = False
        else:
            compression_algorithm = "lz4"  # Fastest compression
    
    # For very large tensors, always compress
    if raw_size > 50 * 1024 * 1024:  # > 50MB
        should_compress = True
        compression_algorithm = "zstd"  # Best compression ratio
    
    return serialize_mlx_array(array, compress=should_compress, compression_algorithm=compression_algorithm)


def benchmark_compression_algorithms(array: mx.array) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different compression algorithms for a given tensor.
    
    Args:
        array: MLX array to benchmark
        
    Returns:
        Dictionary with benchmark results for each algorithm
    """
    results = {}
    algorithms = ["none", "gzip", "lz4", "zstd"]
    
    for algorithm in algorithms:
        try:
            if algorithm == "none":
                data, metadata = serialize_mlx_array(array, compress=False)
            else:
                data, metadata = serialize_mlx_array(array, compress=True, compression_algorithm=algorithm)
            
            # Test decompression time
            start_time = time.time()
            reconstructed = deserialize_mlx_array(data, metadata)
            decompression_time = (time.time() - start_time) * 1000
            
            results[algorithm] = {
                'compression_ratio': metadata.get('compression_info', {}).get('ratio', 1.0),
                'compression_time_ms': metadata.get('compression_info', {}).get('time_ms', 0.0),
                'decompression_time_ms': decompression_time,
                'total_time_ms': metadata.get('compression_info', {}).get('time_ms', 0.0) + decompression_time,
                'compressed_size': len(data),
                'original_size': metadata.get('original_size', 0),
                'accuracy_preserved': mx.allclose(array, reconstructed, rtol=1e-6)
            }
            
        except Exception as e:
            results[algorithm] = {
                'error': str(e),
                'compression_ratio': 1.0,
                'compression_time_ms': 0.0,
                'decompression_time_ms': 0.0,
                'total_time_ms': 0.0,
                'compressed_size': 0,
                'original_size': 0,
                'accuracy_preserved': False
            }
    
    return results
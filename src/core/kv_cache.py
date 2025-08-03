#!/usr/bin/env python3
"""
KV-Cache implementation for MLX distributed inference.

This module implements key-value caching to avoid recomputing attention states
for already processed tokens, dramatically improving generation speed.
"""

import mlx.core as mx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KVCache:
    """
    Key-Value cache for transformer layers.
    
    Stores attention key and value states to avoid recomputation during generation.
    """
    
    def __init__(self, max_sequence_length: int = 2048):
        self.max_sequence_length = max_sequence_length
        self.cache: Dict[int, Dict[str, mx.array]] = {}  # layer_idx -> {keys, values}
        self.sequence_length = 0
        self.initialized = False
        
    def initialize_layer(self, layer_idx: int, key_shape: Tuple[int, ...], value_shape: Tuple[int, ...]):
        """Initialize cache for a specific layer."""
        if layer_idx not in self.cache:
            # Pre-allocate cache with max sequence length
            batch_size, num_heads, _, head_dim = key_shape
            
            self.cache[layer_idx] = {
                'keys': mx.zeros((batch_size, num_heads, self.max_sequence_length, head_dim), dtype=mx.float32),
                'values': mx.zeros((batch_size, num_heads, self.max_sequence_length, head_dim), dtype=mx.float32),
                'valid_length': 0
            }
            logger.debug(f"Initialized KV cache for layer {layer_idx} with shape {key_shape}")
    
    def get_kv(self, layer_idx: int, sequence_pos: int) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """
        Get cached key-value states for a layer up to sequence_pos.
        
        Returns:
            Tuple of (keys, values) or (None, None) if not cached
        """
        if layer_idx not in self.cache:
            return None, None
            
        cache_entry = self.cache[layer_idx]
        valid_length = cache_entry['valid_length']
        
        if valid_length < sequence_pos:
            return None, None
            
        # Return cached states up to sequence_pos
        keys = cache_entry['keys'][:, :, :sequence_pos, :]
        values = cache_entry['values'][:, :, :sequence_pos, :]
        
        return keys, values
    
    def update_kv(self, layer_idx: int, keys: mx.array, values: mx.array, sequence_pos: int):
        """
        Update cache with new key-value states.
        
        Args:
            layer_idx: Layer index
            keys: New key states [batch, heads, seq_len, head_dim]
            values: New value states [batch, heads, seq_len, head_dim]
            sequence_pos: Starting position in sequence
        """
        if layer_idx not in self.cache:
            self.initialize_layer(layer_idx, keys.shape, values.shape)
        
        cache_entry = self.cache[layer_idx]
        seq_len = keys.shape[2]
        
        # Update cache
        cache_entry['keys'][:, :, sequence_pos:sequence_pos + seq_len, :] = keys
        cache_entry['values'][:, :, sequence_pos:sequence_pos + seq_len, :] = values
        cache_entry['valid_length'] = max(cache_entry['valid_length'], sequence_pos + seq_len)
        
        # Update global sequence length
        self.sequence_length = max(self.sequence_length, sequence_pos + seq_len)
        
        logger.debug(f"Updated KV cache for layer {layer_idx} at pos {sequence_pos}, now valid up to {cache_entry['valid_length']}")
    
    def get_cache_info(self, layer_idx: int) -> Dict:
        """Get cache statistics for a layer."""
        if layer_idx not in self.cache:
            return {'valid_length': 0, 'max_length': self.max_sequence_length}
        
        return {
            'valid_length': self.cache[layer_idx]['valid_length'],
            'max_length': self.max_sequence_length,
            'memory_mb': self._estimate_memory_usage(layer_idx)
        }
    
    def _estimate_memory_usage(self, layer_idx: int) -> float:
        """Estimate memory usage for a layer's cache in MB."""
        if layer_idx not in self.cache:
            return 0.0
        
        keys = self.cache[layer_idx]['keys']
        values = self.cache[layer_idx]['values']
        
        # Estimate bytes (float32 = 4 bytes per element)
        total_elements = keys.size + values.size
        total_bytes = total_elements * 4
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def clear(self):
        """Clear all cached states."""
        self.cache.clear()
        self.sequence_length = 0
        logger.info("Cleared KV cache")
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all layers in MB."""
        total_mb = 0.0
        for layer_idx in self.cache:
            total_mb += self._estimate_memory_usage(layer_idx)
        return total_mb


class DistributedKVCache:
    """
    Distributed KV-Cache manager for multi-device inference.
    
    Manages KV-cache across multiple devices, ensuring consistency
    and optimal memory usage.
    """
    
    def __init__(self, device_id: str, coordinator_id: str):
        self.device_id = device_id
        self.coordinator_id = coordinator_id
        self.local_cache = KVCache()
        self.device_caches: Dict[str, KVCache] = {}
        
    def get_device_cache(self, device_id: str) -> KVCache:
        """Get or create cache for a specific device."""
        if device_id not in self.device_caches:
            self.device_caches[device_id] = KVCache()
        return self.device_caches[device_id]
    
    def coordinate_cache_update(self, layer_assignments: Dict[str, List[int]], 
                              sequence_pos: int) -> Dict[str, List[int]]:
        """
        Coordinate cache updates across devices.
        
        Returns which layers each device needs to compute based on cache state.
        """
        compute_plan = {}
        
        for device_id, layers in layer_assignments.items():
            device_cache = self.get_device_cache(device_id)
            layers_to_compute = []
            
            for layer_idx in layers:
                cache_info = device_cache.get_cache_info(layer_idx)
                if cache_info['valid_length'] <= sequence_pos:
                    layers_to_compute.append(layer_idx)
            
            compute_plan[device_id] = layers_to_compute
        
        return compute_plan
    
    def sync_cache_states(self, device_states: Dict[str, Dict[int, Tuple[mx.array, mx.array]]], 
                         sequence_pos: int):
        """
        Synchronize cache states across devices.
        
        Args:
            device_states: {device_id: {layer_idx: (keys, values)}}
            sequence_pos: Current sequence position
        """
        for device_id, layer_states in device_states.items():
            device_cache = self.get_device_cache(device_id)
            
            for layer_idx, (keys, values) in layer_states.items():
                device_cache.update_kv(layer_idx, keys, values, sequence_pos)
    
    def get_memory_report(self) -> Dict:
        """Get memory usage report across all devices."""
        report = {
            'local_device': self.device_id,
            'total_memory_mb': self.local_cache.get_total_memory_usage(),
            'devices': {}
        }
        
        for device_id, cache in self.device_caches.items():
            report['devices'][device_id] = {
                'memory_mb': cache.get_total_memory_usage(),
                'sequence_length': cache.sequence_length
            }
        
        return report


def create_attention_cache_key(batch_size: int, num_heads: int, sequence_length: int, 
                             head_dim: int) -> str:
    """Create a unique key for attention cache configuration."""
    return f"attn_{batch_size}_{num_heads}_{sequence_length}_{head_dim}"


def estimate_cache_memory_requirements(model_config: Dict, batch_size: int = 1, 
                                     sequence_length: int = 2048) -> Dict:
    """
    Estimate memory requirements for KV-caching.
    
    Args:
        model_config: Model configuration with layer info
        batch_size: Batch size
        sequence_length: Maximum sequence length
    
    Returns:
        Dictionary with memory estimates per layer and total
    """
    num_layers = model_config.get('num_hidden_layers', 28)
    num_heads = model_config.get('num_attention_heads', 16)
    head_dim = model_config.get('hidden_size', 2048) // num_heads
    
    # Each layer stores keys and values: 2 * [batch, heads, seq_len, head_dim] * 4 bytes (float32)
    memory_per_layer = 2 * batch_size * num_heads * sequence_length * head_dim * 4  # bytes
    memory_per_layer_mb = memory_per_layer / (1024 * 1024)
    
    total_memory_mb = memory_per_layer_mb * num_layers
    
    return {
        'memory_per_layer_mb': memory_per_layer_mb,
        'total_memory_mb': total_memory_mb,
        'num_layers': num_layers,
        'configuration': {
            'batch_size': batch_size,
            'num_heads': num_heads,
            'sequence_length': sequence_length,
            'head_dim': head_dim
        }
    }
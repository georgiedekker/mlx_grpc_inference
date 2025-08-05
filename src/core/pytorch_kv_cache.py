#!/usr/bin/env python3
"""
PyTorch KV-Cache implementation for distributed inference.

This module implements key-value caching for PyTorch models to avoid 
recomputing attention states, dramatically improving generation speed.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceCapability:
    """Device capability information"""
    device_id: str
    memory_gb: float
    gpu_cores: int
    bandwidth_gbps: float
    assigned_layers: int = 0
    
    def compute_score(self) -> float:
        """Calculate compute score: 60% GPU, 30% memory, 10% bandwidth"""
        return (self.gpu_cores * 0.6 + 
                self.memory_gb * 0.3 + 
                self.bandwidth_gbps/100 * 0.1)
    
    def available_memory_gb(self) -> float:
        """Calculate available memory after model loading"""
        # Estimate ~500MB per layer for model weights
        model_memory_gb = self.assigned_layers * 0.5
        # Reserve 2GB for system and PyTorch overhead
        available = self.memory_gb - model_memory_gb - 2.0
        return max(available, 1.0)  # Minimum 1GB for cache

class PyTorchKVCache:
    """
    Key-Value cache for PyTorch transformer layers.
    
    Stores attention key and value states to avoid recomputation during generation.
    """
    
    def __init__(self, device: torch.device, max_sequence_length: int = 2048, dtype: torch.dtype = torch.float16):
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}  # layer_idx -> {keys, values, seq_len}
        self.sequence_length = 0
        self.total_memory_mb = 0
        
    def initialize_layer(self, layer_idx: int, batch_size: int, num_heads: int, head_dim: int):
        """Initialize cache for a specific layer."""
        if layer_idx not in self.cache:
            # Pre-allocate cache with max sequence length
            keys = torch.zeros(
                (batch_size, num_heads, self.max_sequence_length, head_dim),
                device=self.device, dtype=self.dtype
            )
            values = torch.zeros(
                (batch_size, num_heads, self.max_sequence_length, head_dim),
                device=self.device, dtype=self.dtype
            )
            
            self.cache[layer_idx] = {
                'keys': keys,
                'values': values,
                'seq_len': 0,
                'batch_size': batch_size,
                'num_heads': num_heads,
                'head_dim': head_dim
            }
            
            # Update memory usage
            memory_mb = (keys.numel() + values.numel()) * keys.element_size() / (1024 * 1024)
            self.total_memory_mb += memory_mb
            
            logger.debug(f"Initialized KV cache for layer {layer_idx} with shape "
                        f"[{batch_size}, {num_heads}, {self.max_sequence_length}, {head_dim}], "
                        f"memory: {memory_mb:.1f}MB")
    
    def get_kv(self, layer_idx: int, sequence_pos: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get cached key-value states for a layer up to sequence_pos.
        
        Returns:
            Tuple of (keys, values) or (None, None) if not cached
        """
        if layer_idx not in self.cache:
            return None, None
            
        cache_entry = self.cache[layer_idx]
        cached_seq_len = cache_entry['seq_len']
        
        if cached_seq_len < sequence_pos:
            return None, None
            
        # Return cached states up to sequence_pos
        keys = cache_entry['keys'][:, :, :sequence_pos, :]
        values = cache_entry['values'][:, :, :sequence_pos, :]
        
        return keys, values
    
    def update_kv(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, sequence_pos: int):
        """
        Update cache with new key-value states.
        
        Args:
            layer_idx: Layer index
            keys: New key states [batch, heads, seq_len, head_dim]
            values: New value states [batch, heads, seq_len, head_dim]
            sequence_pos: Starting position in sequence
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        if layer_idx not in self.cache:
            self.initialize_layer(layer_idx, batch_size, num_heads, head_dim)
        
        cache_entry = self.cache[layer_idx]
        
        # Update cache - ensure tensors are on correct device
        keys_device = keys.to(self.device)
        values_device = values.to(self.device)
        
        end_pos = sequence_pos + seq_len
        if end_pos <= self.max_sequence_length:
            cache_entry['keys'][:, :, sequence_pos:end_pos, :] = keys_device
            cache_entry['values'][:, :, sequence_pos:end_pos, :] = values_device
            cache_entry['seq_len'] = max(cache_entry['seq_len'], end_pos)
            
            # Update global sequence length
            self.sequence_length = max(self.sequence_length, end_pos)
            
            logger.debug(f"Updated KV cache for layer {layer_idx} at pos {sequence_pos}, "
                        f"now valid up to {cache_entry['seq_len']}")
        else:
            logger.warning(f"Sequence too long for cache: {end_pos} > {self.max_sequence_length}")
    
    def get_cache_info(self, layer_idx: int) -> Dict[str, Any]:
        """Get cache statistics for a layer."""
        if layer_idx not in self.cache:
            return {'seq_len': 0, 'max_length': self.max_sequence_length, 'memory_mb': 0}
        
        cache_entry = self.cache[layer_idx]
        keys = cache_entry['keys']
        values = cache_entry['values']
        memory_mb = (keys.numel() + values.numel()) * keys.element_size() / (1024 * 1024)
        
        return {
            'seq_len': cache_entry['seq_len'],
            'max_length': self.max_sequence_length,
            'memory_mb': memory_mb,
            'batch_size': cache_entry['batch_size'],
            'num_heads': cache_entry['num_heads'],
            'head_dim': cache_entry['head_dim']
        }
    
    def clear(self):
        """Clear all cached states."""
        self.cache.clear()
        self.sequence_length = 0
        self.total_memory_mb = 0
        logger.info("Cleared KV cache")
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all layers in MB."""
        return self.total_memory_mb


class HeterogeneousCacheAllocator:
    """
    Allocates cache slots based on device capabilities.
    """
    
    def __init__(self, device_capabilities: List[DeviceCapability]):
        self.capabilities = device_capabilities
        self.cache_assignments = {}
        
    def calculate_cache_allocation(self, max_batch_size: int = 8, sequence_length: int = 2048) -> Dict[str, Dict[str, Any]]:
        """
        Calculate cache allocation for each device based on capabilities.
        
        Args:
            max_batch_size: Maximum batch size to support
            sequence_length: Maximum sequence length
            
        Returns:
            Dict mapping device_id to cache configuration
        """
        allocations = {}
        
        # Calculate memory requirements per cached sequence
        # Rough estimate: 2 tensors * batch * heads * seq_len * head_dim * bytes_per_element
        # For typical model: 16 heads, 128 head_dim, float16 (2 bytes)
        memory_per_sequence_mb = (2 * max_batch_size * 16 * sequence_length * 128 * 2) / (1024 * 1024)
        
        for device in self.capabilities:
            available_memory = device.available_memory_gb() * 1024  # Convert to MB
            
            # Reserve 20% for overhead and fragmentation
            usable_memory = available_memory * 0.8
            
            # Calculate how many sequences we can cache
            max_cached_sequences = max(1, int(usable_memory / memory_per_sequence_mb))
            
            # Proportional allocation based on compute score
            total_compute = sum(d.compute_score() for d in self.capabilities)
            proportion = device.compute_score() / total_compute
            
            allocations[device.device_id] = {
                'max_cached_sequences': max_cached_sequences,
                'max_sequence_length': sequence_length,
                'memory_budget_mb': usable_memory,
                'memory_per_sequence_mb': memory_per_sequence_mb,
                'compute_proportion': proportion,
                'device_info': {
                    'memory_gb': device.memory_gb,
                    'available_memory_gb': device.available_memory_gb(),
                    'gpu_cores': device.gpu_cores,
                    'assigned_layers': device.assigned_layers
                }
            }
            
            logger.info(f"Device {device.device_id}: {max_cached_sequences} sequences, "
                       f"{usable_memory:.1f}MB budget, {proportion:.1%} compute share")
        
        self.cache_assignments = allocations
        return allocations


class DistributedKVCacheManager:
    """
    Manages KV cache across distributed devices with heterogeneous capabilities.
    """
    
    def __init__(self, rank: int, world_size: int, device: torch.device, 
                 device_capabilities: Optional[List[DeviceCapability]] = None):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_cache = PyTorchKVCache(device)
        
        # Device capabilities and allocations
        self.device_capabilities = device_capabilities or []
        self.allocator = HeterogeneousCacheAllocator(self.device_capabilities) if device_capabilities else None
        self.cache_allocation = {}
        
        # Cache coordination
        self.cache_request_id = 0
        self.distributed_cache_enabled = world_size > 1
        
        # Memory monitoring
        self.memory_usage_history = []
        
    def initialize_cache_allocation(self, max_batch_size: int = 8, sequence_length: int = 2048):
        """Initialize cache allocation based on device capabilities."""
        if self.allocator:
            self.cache_allocation = self.allocator.calculate_cache_allocation(max_batch_size, sequence_length)
            
            # Configure local cache based on allocation
            if self.device_capabilities:
                my_device = self.device_capabilities[self.rank]
                allocation = self.cache_allocation.get(my_device.device_id, {})
                max_seq_len = allocation.get('max_sequence_length', sequence_length)
                
                # Update local cache configuration
                self.local_cache.max_sequence_length = max_seq_len
                
                logger.info(f"Rank {self.rank}: Cache initialized with max_seq_len={max_seq_len}")
    
    def should_use_cache(self, layer_idx: int, sequence_pos: int) -> bool:
        """Determine if cache should be used for this layer and position."""
        if not self.distributed_cache_enabled:
            return True
            
        # Check if we have valid cache for this layer
        cache_info = self.local_cache.get_cache_info(layer_idx)
        return cache_info['seq_len'] > sequence_pos
    
    def get_cached_kv(self, layer_idx: int, sequence_pos: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached key-value states for a layer."""
        return self.local_cache.get_kv(layer_idx, sequence_pos)
    
    def update_cache(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, sequence_pos: int):
        """Update local cache and optionally broadcast to other ranks."""
        # Update local cache
        self.local_cache.update_kv(layer_idx, keys, values, sequence_pos)
        
        # In a full implementation, you might broadcast cache updates to other ranks
        # For now, we keep cache local to each rank for simplicity
        if self.distributed_cache_enabled and self.rank == 0:
            # Coordinator could manage global cache state here
            logger.debug(f"Cache updated for layer {layer_idx} at position {sequence_pos}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        report = {
            'rank': self.rank,
            'device': str(self.device),
            'total_cache_memory_mb': self.local_cache.get_total_memory_usage(),
            'sequence_length': self.local_cache.sequence_length,
            'cached_layers': len(self.local_cache.cache),
            'allocation': self.cache_allocation.get(f'rank_{self.rank}', {}),
            'timestamp': time.time()
        }
        
        # Add per-layer details
        layer_details = {}
        for layer_idx in self.local_cache.cache:
            layer_details[layer_idx] = self.local_cache.get_cache_info(layer_idx)
        report['layer_details'] = layer_details
        
        # Track memory usage over time
        self.memory_usage_history.append({
            'timestamp': time.time(),
            'memory_mb': report['total_cache_memory_mb']
        })
        
        # Keep only last 100 measurements
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]
        
        return report
    
    def evict_oldest_cache(self, layers_to_keep: int = 5):
        """Simple LRU-style cache eviction."""
        if len(self.local_cache.cache) <= layers_to_keep:
            return
        
        # Simple eviction: remove layers with lowest indices (oldest)
        sorted_layers = sorted(self.local_cache.cache.keys())
        layers_to_remove = sorted_layers[:-layers_to_keep]
        
        for layer_idx in layers_to_remove:
            del self.local_cache.cache[layer_idx]
            logger.debug(f"Evicted cache for layer {layer_idx}")
        
        # Recalculate memory usage
        self.local_cache.total_memory_mb = sum(
            self.local_cache.get_cache_info(layer_idx)['memory_mb']
            for layer_idx in self.local_cache.cache
        )
    
    def clear_all_cache(self):
        """Clear all cache across the distributed system."""
        self.local_cache.clear()
        
        if self.distributed_cache_enabled:
            # In a full implementation, coordinate clearing across all ranks
            logger.info(f"Rank {self.rank}: All cache cleared")


def estimate_kv_cache_memory(model_config: Dict[str, Any], batch_size: int = 1, 
                           sequence_length: int = 2048, num_layers: int = 28) -> Dict[str, float]:
    """
    Estimate memory requirements for KV cache.
    
    Args:
        model_config: Model configuration
        batch_size: Batch size
        sequence_length: Maximum sequence length
        num_layers: Number of transformer layers
    
    Returns:
        Memory estimates in MB
    """
    # Default values for common models
    num_heads = model_config.get('num_attention_heads', 16)
    hidden_size = model_config.get('hidden_size', 2048)
    head_dim = hidden_size // num_heads
    
    # Each layer stores keys and values: 2 * [batch, heads, seq_len, head_dim] * 2 bytes (float16)
    memory_per_layer_mb = (2 * batch_size * num_heads * sequence_length * head_dim * 2) / (1024 * 1024)
    total_memory_mb = memory_per_layer_mb * num_layers
    
    return {
        'memory_per_layer_mb': memory_per_layer_mb,
        'total_memory_mb': total_memory_mb,
        'num_layers': num_layers,
        'configuration': {
            'batch_size': batch_size,
            'num_heads': num_heads,
            'sequence_length': sequence_length,
            'head_dim': head_dim,
            'hidden_size': hidden_size
        }
    }


def load_device_capabilities_from_config(config_path: str) -> List[DeviceCapability]:
    """Load device capabilities from cluster configuration."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        capabilities = []
        for device_config in config.get('devices', []):
            caps = device_config.get('capabilities', {})
            capability = DeviceCapability(
                device_id=device_config['device_id'],
                memory_gb=caps.get('memory_gb', 16),
                gpu_cores=caps.get('gpu_cores', 10),
                bandwidth_gbps=caps.get('bandwidth_gbps', 200.0)
            )
            capabilities.append(capability)
        
        return capabilities
    except Exception as e:
        logger.warning(f"Could not load device capabilities: {e}")
        return []
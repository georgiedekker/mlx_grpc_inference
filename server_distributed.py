#!/usr/bin/env python3
"""
True distributed MLX inference - splits model layers across devices
"""
import os
import sys
import asyncio
import logging
import time
import json
import grpc
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from concurrent import futures
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pickle
import hashlib
import zlib

# Import the proper tensor utilities
from src.utils.tensor_utils import (
    serialize_mlx_array, deserialize_mlx_array,
    serialize_kv_cache, deserialize_kv_cache
)

# Import protobuf definitions
from src.communication import inference_pb2, inference_pb2_grpc

# FastAPI for coordinator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] Rank %(rank)s - %(levelname)s - %(message)s'
)

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = os.environ.get('RANK', '?')
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

def compute_tensor_checksum(tensor: mx.array) -> str:
    """Compute SHA256 checksum of tensor in its native dtype"""
    mx.eval(tensor)
    tensor_bytes = bytes(memoryview(tensor))
    return hashlib.sha256(tensor_bytes).hexdigest()[:16]

def compute_mask_crc(mask: Optional[mx.array]) -> str:
    """Compute CRC32 of mask for consistency checking"""
    if mask is None:
        return "NO_MASK"
    import zlib
    mx.eval(mask)
    mask_bytes = bytes(memoryview(mask))
    return hex(zlib.crc32(mask_bytes))

def log_layernorm_stats(hidden_states: mx.array, layer_idx: int, position: str):
    """Log LayerNorm statistics for debugging with color coding and threshold checks"""
    mean = mx.mean(hidden_states).item()
    var = mx.var(hidden_states).item()
    max_abs = mx.max(mx.abs(hidden_states)).item()
    
    # Color codes for terminal output
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    
    # Thresholds
    MAX_ABS_THRESHOLD = 10.0
    MAX_ABS_WARNING = 5.0
    VAR_THRESHOLD = 100.0
    VAR_WARNING = 10.0
    
    # Determine color based on thresholds
    color = GREEN
    status = "OK"
    if max_abs > MAX_ABS_THRESHOLD:
        color = RED
        status = "CRITICAL"
    elif max_abs > MAX_ABS_WARNING:
        color = YELLOW
        status = "WARNING"
    elif var > VAR_THRESHOLD:
        color = RED
        status = "HIGH_VAR"
    elif var > VAR_WARNING:
        color = YELLOW
        status = "WARN_VAR"
    
    # Log with color coding
    message = (f"{color}{BOLD}LayerNorm stats {position} layer {layer_idx}: "
               f"mean={mean:.6f}, var={var:.6f}, max_abs={max_abs:.6f} [{status}]{RESET}")
    logger.info(message)
    
    # Log divergence immediately if detected
    if max_abs > MAX_ABS_THRESHOLD:
        logger.error(f"{RED}DIVERGENCE DETECTED at layer {layer_idx} {position}! max_abs={max_abs:.6f}{RESET}")
        # Log first few values for debugging
        flat = hidden_states.flatten()
        first_10 = flat[:10].tolist()
        logger.error(f"First 10 values: {first_10}")
        # Find indices of extreme values
        abs_flat = mx.abs(flat)
        max_indices = mx.argsort(abs_flat)[::-1][:5]  # Top 5 extreme values
        for idx in max_indices:
            logger.error(f"  Index {idx}: value={flat[idx].item():.6f}")

class WorkerServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC service implementation for worker nodes"""
    
    def __init__(self, rank: int, model_name: str):
        self.rank = rank
        self.ready = False
        self.device_id = f"mini{rank+1}"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.assigned_layers = []
        self.persistent_kv_cache = None  # Will be initialized after model load
        logger.info(f"Initialized WorkerServicer for rank {self.rank}")
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the MLX model"""
        try:
            logger.info(f"Worker {self.rank} loading model {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            self.ready = True
            
            # Determine which layers this worker handles
            total_layers = len(self.model.model.layers)
            layers_per_worker = total_layers // 2  # Split evenly for 2 devices
            
            if self.rank == 1:  # Worker 1 gets second half
                self.assigned_layers = list(range(layers_per_worker, total_layers))
            
            # Initialize persistent KV cache for ALL layers (not just assigned ones)
            from mlx_lm.models.cache import KVCache
            self.persistent_kv_cache = [KVCache() for _ in range(total_layers)]
            logger.info(f"Worker {self.rank} initialized persistent KV cache with {total_layers} objects")
            
            logger.info(f"Worker {self.rank} loaded successfully, handling layers: {self.assigned_layers}")
            
        except Exception as e:
            logger.error(f"Failed to load model on worker {self.rank}: {e}")
            self.ready = False
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        return inference_pb2.HealthStatus(
            healthy=self.ready,
            device_id=self.device_id,
            timestamp=int(time.time()),
            details={"status": "ready" if self.ready else "not ready", "rank": str(self.rank)}
        )
    
    def GetDeviceInfo(self, request, context):
        """Get device information"""
        return inference_pb2.DeviceInfo(
            device_id=self.device_id,
            hostname=f"mini{self.rank+1}.local",
            rank=self.rank,
            role="worker",
            assigned_layers=self.assigned_layers,
            capabilities={"device_type": "Apple Silicon", "memory": "8GB"},
            gpu_utilization=0.0,
            memory_usage_gb=0.0
        )
    
    def ProcessLayers(self, request, context):
        """Process specific model layers with KV cache support"""
        if not self.ready:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Model not loaded")
        
        logger.info(f"Worker {self.rank} processing layers {request.layer_indices}")
        logger.info(f"Worker received input_tensor size: {len(request.input_tensor)} bytes")
        
        try:
            # Deserialize input tensor
            metadata = {
                'shape': list(request.metadata.shape),
                'dtype': request.metadata.dtype,
                'compressed': getattr(request.metadata, 'compressed', False),
                'original_dtype': getattr(request.metadata, 'original_dtype', request.metadata.dtype),
                'requires_conversion': getattr(request.metadata, 'requires_conversion', False)
            }
            logger.info(f"WORKER RECV metadata: dtype={metadata['dtype']}, original_dtype={metadata['original_dtype']}, shape={metadata['shape']}, data_size={len(request.input_tensor)}")
            
            # CRITICAL: Log SHA-256 of bytes received from wire immediately
            import hashlib
            wire_recv_sha256 = hashlib.sha256(request.input_tensor).hexdigest()[:16]
            logger.info(f"WORKER<-COORDINATOR WIRE: SHA-256 of received bytes: {wire_recv_sha256}, size={len(request.input_tensor)}")
            
            # Log received data checksum (should match metadata)
            metadata_sha256 = metadata.get('checksum', 'NO_CHECKSUM')[:16]
            logger.info(f"WORKER<-COORDINATOR: wire sha256={wire_recv_sha256}, metadata checksum={metadata_sha256}")
            
            hidden_states = deserialize_mlx_array(request.input_tensor, metadata)
            
            # VALIDATION: Checksum of received bytes to verify transmission integrity
            # Compute checksum on the received bytes BEFORE deserialization
            received_bytes_sha256 = hashlib.sha256(request.input_tensor).hexdigest()[:16]
            
            # Now check the deserialized tensor
            mx.eval(hidden_states)
            
            # CRITICAL: Max-abs check on first 16 values after deserialization
            flat_tensor = hidden_states.flatten()
            first_16_count = min(16, flat_tensor.shape[0])
            if first_16_count > 0:
                first_16_mx = flat_tensor[:first_16_count].astype(mx.float32)
                mx.eval(first_16_mx)
                first_16_values = [float(first_16_mx[i]) for i in range(first_16_count)]
                max_abs = max(abs(v) for v in first_16_values)
                logger.info(f"WORKER RECV MAX-ABS (first 16): {max_abs:.6f}, values={first_16_values[:4]}...")
            
            hidden_norm = mx.linalg.norm(hidden_states).item()
            first_4 = hidden_states.flatten()[:4].tolist()
            received_checksum = hidden_states.sum().item()
            received_shape = hidden_states.shape  
            received_dtype = hidden_states.dtype
            logger.info(f"WORKER RECV: bytes_sha256={received_bytes_sha256}, norm={hidden_norm:.3f}, first_4={first_4}, checksum={received_checksum:.6f}, shape={received_shape}, dtype={received_dtype}")
            
            # Deserialize KV cache if present
            kv_cache = None
            if hasattr(request, 'kv_cache_data') and request.kv_cache_data:
                cache_metadata = {
                    'type': request.kv_cache_metadata.type,
                    'length': request.kv_cache_metadata.length,
                    'entries': []
                }
                for entry in request.kv_cache_metadata.entries:
                    entry_dict = {
                        'type': entry.type,
                        'offset': entry.offset,
                        'size': entry.size
                    }
                    if entry.HasField('keys_meta'):
                        entry_dict['keys_meta'] = {
                            'shape': list(entry.keys_meta.shape),
                            'dtype': entry.keys_meta.dtype,
                            'compressed': entry.keys_meta.compressed,
                            'original_dtype': entry.keys_meta.original_dtype,
                            'requires_conversion': entry.keys_meta.requires_conversion
                        }
                    if entry.HasField('values_meta'):
                        entry_dict['values_meta'] = {
                            'shape': list(entry.values_meta.shape),
                            'dtype': entry.values_meta.dtype,
                            'compressed': entry.values_meta.compressed,
                            'original_dtype': entry.values_meta.original_dtype,
                            'requires_conversion': entry.values_meta.requires_conversion
                        }
                    if entry.HasField('meta'):
                        entry_dict['meta'] = {
                            'shape': list(entry.meta.shape),
                            'dtype': entry.meta.dtype,
                            'compressed': entry.meta.compressed,
                            'original_dtype': entry.meta.original_dtype,
                            'requires_conversion': entry.meta.requires_conversion
                        }
                    cache_metadata['entries'].append(entry_dict)
                
                # Deserialize the compact cache from coordinator
                # Coordinator sends only worker layers in compact format
                compact_cache = deserialize_kv_cache(request.kv_cache_data, cache_metadata)
                logger.info(f"Received compact KV cache with {len(compact_cache)} entries for layers {self.assigned_layers}")
                
                # Update only the coordinator's layers in our persistent cache
                # The coordinator handles layers 0-13, so update those
                coordinator_layers = list(range(14))  # Layers 0-13
                for idx, layer_idx in enumerate(coordinator_layers):
                    if idx < len(compact_cache) and compact_cache[idx] is not None:
                        # Update this layer in our persistent cache
                        if self.persistent_kv_cache[layer_idx] is not None:
                            self.persistent_kv_cache[layer_idx].keys = compact_cache[idx].keys
                            self.persistent_kv_cache[layer_idx].values = compact_cache[idx].values
                            self.persistent_kv_cache[layer_idx].offset = compact_cache[idx].offset
                            self.persistent_kv_cache[layer_idx].step = compact_cache[idx].step
                        else:
                            # Create new cache entry if it doesn't exist
                            self.persistent_kv_cache[layer_idx] = compact_cache[idx]
            
            # Process assigned layers with KV cache
            # CRITICAL: Use the worker's persistent cache and update it with received data
            if kv_cache is not None:
                # Update our persistent cache with the received cache data
                # This preserves object identity on the worker side
                cache_to_use = self.persistent_kv_cache
            else:
                # Use our persistent cache directly
                cache_to_use = self.persistent_kv_cache
            
            # Debug: Check cache lengths for ALL layers - must match coordinator exactly
            cache_info = []
            for i, cache_entry in enumerate(cache_to_use):
                if cache_entry is not None:
                    if hasattr(cache_entry, 'keys') and cache_entry.keys is not None:
                        seq_len = cache_entry.keys.shape[1] if len(cache_entry.keys.shape) > 1 else 0
                        cache_info.append(f"({i},{seq_len})")
                    elif isinstance(cache_entry, (tuple, list)) and len(cache_entry) == 2 and cache_entry[0] is not None:
                        seq_len = cache_entry[0].shape[1] if len(cache_entry[0].shape) > 1 else 0
                        cache_info.append(f"({i},{seq_len})")
                    else:
                        cache_info.append(f"({i},0)")
                else:
                    cache_info.append(f"({i},N)")
            
            # Log worker cache state - must match coordinator
            logger.info(f"RANK{self.rank} cache before: {' '.join(cache_info[:10])}")
            if len(cache_info) > 10:
                logger.info(f"RANK{self.rank} cache before (cont): {' '.join(cache_info[10:20])}")
            if len(cache_info) > 20:
                logger.info(f"RANK{self.rank} cache before (end): {' '.join(cache_info[20:])}")
            
            # Log cache state BEFORE processing
            cache_info_before = []
            for i in self.assigned_layers:
                if i < len(cache_to_use) and cache_to_use[i] is not None:
                    cache_info_before.append(f"L{i}:off={cache_to_use[i].offset}")
                else:
                    cache_info_before.append(f"L{i}:None")
            logger.info(f"WORKER{self.rank} cache offsets BEFORE: {' '.join(cache_info_before)}")
            
            # Log mask CRC for consistency check
            mask_crc = compute_mask_crc(None)  # Worker always uses mask=None
            logger.info(f"WORKER processing with mask CRC: {mask_crc}")
            
            for layer_idx in request.layer_indices:
                if layer_idx in self.assigned_layers:
                    # Log LayerNorm stats for critical layers before processing
                    if layer_idx in [14, 27]:
                        log_layernorm_stats(hidden_states, layer_idx, "BEFORE")
                    
                    layer = self.model.model.layers[layer_idx]
                    # Pass the specific cache entry for this layer
                    # Call layer with mask=None to match single-device behavior
                    hidden_states = layer(hidden_states, mask=None, cache=cache_to_use[layer_idx])
                    
                    # Log LayerNorm stats for critical layers after processing
                    if layer_idx in [14, 27]:
                        log_layernorm_stats(hidden_states, layer_idx, "AFTER")
                    
                    # DETAILED DEBUGGING: Print cache state after each layer with sequence length and checksum
                    cache_entry = cache_to_use[layer_idx]
                    cache_debug = f"layer {layer_idx}: cache_is_none={cache_entry is None}"
                    
                    if cache_entry is not None:
                        if hasattr(cache_entry, 'keys') and cache_entry.keys is not None:
                            seq_len = cache_entry.keys.shape[1] if len(cache_entry.keys.shape) > 1 else 0
                            keys_sum = float(cache_entry.keys.sum().item()) if cache_entry.keys.size > 0 else 0.0
                            cache_debug += f", seq_len={seq_len}, keys_shape={cache_entry.keys.shape}, keys_sum={keys_sum:.3f}"
                        elif hasattr(cache_entry, 'keys'):
                            cache_debug += f", keys=None"  
                        else:
                            cache_debug += f", no_keys_attr"
                        
                        # Log cache offset after processing
                        if hasattr(cache_entry, 'offset'):
                            cache_debug += f", offset={cache_entry.offset}"
                        if hasattr(cache_entry, 'step'):
                            cache_debug += f", step={cache_entry.step}"
                            
                    logger.info(f"WORKER{self.rank} {cache_debug}")
            
            # Log cache state AFTER processing
            cache_info_after = []
            for i in self.assigned_layers:
                if i < len(cache_to_use) and cache_to_use[i] is not None:
                    cache_info_after.append(f"L{i}:off={cache_to_use[i].offset}")
                else:
                    cache_info_after.append(f"L{i}:None")
            logger.info(f"WORKER{self.rank} cache offsets AFTER: {' '.join(cache_info_after)}")
            
            # After processing, cache_to_use contains the updated cache entries
            
            # Serialize output tensor
            # Enable compression for better network efficiency
            compress = os.environ.get('COMPRESS_TENSORS', 'true').lower() == 'true'
            output_bytes, output_metadata = serialize_mlx_array(hidden_states, compress=compress)
            
            # VALIDATION: Checksum of bytes being sent back to coordinator
            import hashlib
            output_bytes_sha256 = hashlib.sha256(output_bytes).hexdigest()[:16]
            hidden_norm = mx.linalg.norm(hidden_states).item()
            first_4 = hidden_states.flatten()[:4].tolist()
            output_checksum = hidden_states.sum().item()
            output_shape = hidden_states.shape
            output_dtype = hidden_states.dtype
            logger.info(f"WORKER SEND: bytes_sha256={output_bytes_sha256}, norm={hidden_norm:.3f}, first_4={first_4}, checksum={output_checksum:.6f}, shape={output_shape}, dtype={output_dtype}, transmitted_dtype={output_metadata['dtype']}")
            
            # CRITICAL: Max-abs check before sending back
            flat_tensor = hidden_states.flatten()
            first_16_count = min(16, flat_tensor.shape[0])
            if first_16_count > 0:
                first_16_mx = flat_tensor[:first_16_count].astype(mx.float32)
                mx.eval(first_16_mx)
                first_16_values = [float(first_16_mx[i]) for i in range(first_16_count)]
                max_abs = max(abs(v) for v in first_16_values)
                logger.info(f"WORKER SEND MAX-ABS (first 16): {max_abs:.6f}, values={first_16_values[:4]}...")
            
            # Also log the SHA-256 from the metadata (should match)
            metadata_sha256 = output_metadata.get('checksum', 'NO_CHECKSUM')[:16]
            logger.info(f"WORKER->COORDINATOR: metadata checksum={metadata_sha256}, data size={len(output_bytes)} bytes")
            
            # Serialize updated KV cache - only send back the layers this worker processed
            cache_bytes = b''
            cache_metadata_proto = None
            if cache_to_use:
                # Create a compact representation: only worker's layers
                # This drastically reduces payload size
                worker_only_cache = []
                worker_layer_indices = []
                for i in self.assigned_layers:
                    worker_only_cache.append(cache_to_use[i])
                    worker_layer_indices.append(i)
                
                # Log the size reduction
                logger.info(f"Worker {self.rank} sending {len(worker_only_cache)} cache entries (indices: {worker_layer_indices}) instead of full {len(cache_to_use)}")
                
                cache_bytes, cache_meta = serialize_kv_cache(worker_only_cache)
                logger.info(f"Worker {self.rank} cache payload size: {len(cache_bytes) / 1024 / 1024:.2f}MB")
                
                # Add layer index mapping to metadata
                cache_meta['layer_indices'] = worker_layer_indices
                
                # Convert cache metadata to protobuf
                cache_entries = []
                for i, entry_meta in enumerate(cache_meta['entries']):
                    entry_proto = inference_pb2.KVCacheEntry(
                        type=entry_meta['type'],
                        offset=entry_meta['offset'],
                        size=entry_meta['size']
                    )
                    
                    # Add cache offset/step if available
                    if 'cache_offset' in entry_meta:
                        entry_proto.cache_offset = entry_meta['cache_offset']
                    if 'cache_step' in entry_meta:
                        entry_proto.cache_step = entry_meta['cache_step']
                    
                    if 'keys_meta' in entry_meta:
                        keys_meta = entry_meta['keys_meta']
                        entry_proto.keys_meta.CopyFrom(inference_pb2.TensorMetadata(
                            shape=keys_meta['shape'],
                            dtype=keys_meta['dtype'],
                            compressed=keys_meta.get('compressed', False),
                            original_dtype=keys_meta.get('original_dtype', keys_meta['dtype']),
                            requires_conversion=keys_meta.get('requires_conversion', False)
                        ))
                    if 'values_meta' in entry_meta:
                        values_meta = entry_meta['values_meta']
                        entry_proto.values_meta.CopyFrom(inference_pb2.TensorMetadata(
                            shape=values_meta['shape'],
                            dtype=values_meta['dtype'],
                            compressed=values_meta.get('compressed', False),
                            original_dtype=values_meta.get('original_dtype', values_meta['dtype']),
                            requires_conversion=values_meta.get('requires_conversion', False)
                        ))
                    if 'meta' in entry_meta:
                        meta = entry_meta['meta']
                        entry_proto.meta.CopyFrom(inference_pb2.TensorMetadata(
                            shape=meta['shape'],
                            dtype=meta['dtype'],
                            compressed=meta.get('compressed', False),
                            original_dtype=meta.get('original_dtype', meta['dtype']),
                            requires_conversion=meta.get('requires_conversion', False)
                        ))
                    cache_entries.append(entry_proto)
                
                cache_metadata_proto = inference_pb2.KVCacheMetadata(
                    type=cache_meta['type'],
                    length=cache_meta['length'],
                    entries=cache_entries
                )
            
            # Log total response size
            total_response_size = len(output_bytes) + len(cache_bytes)
            logger.info(f"Worker {self.rank} total response size: {total_response_size / 1024 / 1024:.2f}MB (tensor: {len(output_bytes) / 1024 / 1024:.2f}MB, cache: {len(cache_bytes) / 1024 / 1024:.2f}MB)")
            
            response = inference_pb2.LayerResponse(
                request_id=request.request_id,
                output_tensor=output_bytes,
                metadata=inference_pb2.TensorMetadata(
                    shape=output_metadata['shape'],
                    dtype=output_metadata['dtype'],
                    compressed=output_metadata.get('compressed', False),
                    original_dtype=output_metadata.get('original_dtype', output_metadata['dtype']),
                    requires_conversion=output_metadata.get('requires_conversion', False)
                ),
                processing_time_ms=10.0,
                device_id=self.device_id,
                kv_cache_data=cache_bytes
            )
            
            if cache_metadata_proto:
                response.kv_cache_metadata.CopyFrom(cache_metadata_proto)
            
            # CRITICAL: Log SHA-256 of exact bytes being sent back over the wire
            wire_send_sha256 = hashlib.sha256(response.output_tensor).hexdigest()[:16]
            logger.info(f"WORKER->COORDINATOR WIRE: SHA-256 of transmitted bytes: {wire_send_sha256}, size={len(response.output_tensor)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing layers: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

class DistributedServer:
    """
    True distributed MLX inference server
    """
    
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_coordinator = (self.rank == 0)
        
        # Set deterministic seed for debugging (can be disabled in production)
        if os.environ.get('DETERMINISTIC', 'false').lower() == 'true':
            mx.random.seed(0)
            logger.info("Running in DETERMINISTIC mode with seed=0")
        
        # Configuration
        self.model_name = os.environ.get('MODEL_NAME', 'mlx-community/Qwen2.5-1.5B-Instruct-4bit')
        self.master_addr = os.environ.get('MASTER_ADDR', '192.168.5.1')
        self.grpc_base_port = 50051
        
        # Components
        self.model = None
        self.tokenizer = None
        self.worker_stubs = {}
        self.grpc_server = None
        self.assigned_layers = []
        
        # CRITICAL: Persistent KV cache - same list instance across all generations
        self.persistent_kv_cache = None
        
        # FastAPI app for coordinator
        if self.is_coordinator:
            self.app = self._create_fastapi_app()
        
        logger.info(f"Initializing server on rank {self.rank}")
    
    def _load_model(self):
        """Load the MLX model on coordinator"""
        try:
            logger.info(f"Loading model {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            
            # Determine layer assignment
            total_layers = len(self.model.model.layers)
            layers_per_worker = total_layers // self.world_size
            
            # Coordinator gets first half of layers
            self.assigned_layers = list(range(layers_per_worker))
            
            logger.info(f"Model loaded, coordinator handling layers: {self.assigned_layers}")
            logger.info(f"Total layers: {total_layers}, Workers will handle remaining layers")
            
            # Initialize persistent KV cache - ONE list for the entire model lifecycle
            # CRITICAL: Create explicit cache objects - Qwen layers need real cache objects!
            
            # Import the proper KVCache from mlx-lm 
            from mlx_lm.models.cache import KVCache
            
            self.persistent_kv_cache = [KVCache() for _ in range(total_layers)]
            logger.info(f"Initialized persistent KV cache with {total_layers} MLX KVCache objects")
            
            # Log cache list object identity for debugging
            cache_id = id(self.persistent_kv_cache)
            logger.info(f"Cache list object ID: {cache_id}")
            
            # Log individual cache object IDs
            cache_obj_ids = [id(cache_obj) for cache_obj in self.persistent_kv_cache]
            logger.info(f"Individual cache object IDs: {cache_obj_ids[:5]}... (showing first 5)")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def distributed_forward(self, input_ids: mx.array, mask: mx.array = None, cache = None) -> Tuple[mx.array, Any]:
        """Forward pass distributed across devices with proper KV cache handling"""
        logger.info(f"Starting distributed forward pass on {self.world_size} devices")
        
        # CRITICAL: Validate mask is None to ensure consistency
        original_mask_crc = compute_mask_crc(mask)
        if mask is not None:
            logger.warning(f"WARNING: mask is not None (CRC: {original_mask_crc}), but distributed inference requires mask=None for consistency")
            logger.warning("Forcing mask=None to prevent divergence")
            mask = None
        
        # Log the actual mask being used (should always be None)
        actual_mask_crc = compute_mask_crc(mask)
        logger.info(f"COORDINATOR: Using mask with CRC: {actual_mask_crc} (original was {original_mask_crc})")
        
        # Get embeddings (coordinator only)
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # Debug: Log embedding details
        logger.info(f"Input tokens: {input_ids.tolist()}")
        logger.info(f"Embeddings shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        logger.info(f"Embeddings norm: {mx.linalg.norm(hidden_states).item():.3f}")
        logger.info(f"Embeddings first_4: {hidden_states[0, 0, :4].tolist()}")
        logger.info(f"Embeddings range: [{hidden_states.min().item():.3f}, {hidden_states.max().item():.3f}]")
        
        # Use persistent cache - NEVER recreate the list!
        if cache is None:
            cache = self.persistent_kv_cache
        
        # Log cache list identity to ensure it's preserved across calls
        cache_id = id(cache)
        logger.info(f"Using cache list ID: {cache_id} (persistent ID: {id(self.persistent_kv_cache)})")
        
        # CRITICAL CHECK 1: Log cache object IDs and offsets for ALL layers
        cache_obj_info = []
        for i, c in enumerate(cache):
            if c is not None:
                cache_obj_info.append(f"L{i}:id={id(c)},off={c.offset}")
            else:
                cache_obj_info.append(f"L{i}:None")
        logger.info(f"Cache object IDs/offsets BEFORE: {' '.join(cache_obj_info[:14])}")
        logger.info(f"Cache object IDs/offsets BEFORE (worker): {' '.join(cache_obj_info[14:])}")
        
        # CRITICAL CHECK 2: Track sequence length growth layer-by-layer
        cache_info = []
        cache_shapes = []
        for i, cache_entry in enumerate(cache):
            if cache_entry is not None:
                if hasattr(cache_entry, 'keys') and cache_entry.keys is not None:
                    seq_len = cache_entry.keys.shape[1] if len(cache_entry.keys.shape) > 1 else 0
                    full_shape = cache_entry.keys.shape
                    cache_info.append(f"({i},{seq_len})")
                    cache_shapes.append(f"L{i}:{full_shape}")
                elif isinstance(cache_entry, (tuple, list)) and len(cache_entry) == 2 and cache_entry[0] is not None:
                    seq_len = cache_entry[0].shape[1] if len(cache_entry[0].shape) > 1 else 0
                    full_shape = cache_entry[0].shape
                    cache_info.append(f"({i},{seq_len})")
                    cache_shapes.append(f"L{i}:{full_shape}")
                else:
                    cache_info.append(f"({i},0)")
                    cache_shapes.append(f"L{i}:empty")
            else:
                cache_info.append(f"({i},N)")
                cache_shapes.append(f"L{i}:None")
        
        # Log first 10 layers for readability
        logger.info(f"RANK0 cache before: {' '.join(cache_info[:10])}")
        if len(cache_info) > 10:
            logger.info(f"RANK0 cache before (cont): {' '.join(cache_info[10:20])}")
        if len(cache_info) > 20:
            logger.info(f"RANK0 cache before (end): {' '.join(cache_info[20:])}")
        
        # Log full shapes for critical layers (0, 13, 14, 27)
        logger.info(f"Cache shapes - Layer 0: {cache_shapes[0]}, Layer 13: {cache_shapes[13]}, Layer 14: {cache_shapes[14]}, Layer 27: {cache_shapes[27]}")
        
        # Log mask CRC for consistency check
        mask_crc = compute_mask_crc(mask)
        logger.info(f"COORDINATOR processing with mask CRC: {mask_crc}")
        
        # Process layers on coordinator with proper cache handling
        for layer_idx in self.assigned_layers:
            layer = self.model.model.layers[layer_idx]
            # Pass the specific cache entry for this layer
            cache_entry = cache[layer_idx]
            logger.info(f"DEBUG: layer_idx={layer_idx}, cache_entry type={type(cache_entry)}, has_offset={hasattr(cache_entry, 'offset') if cache_entry else 'None'}")
            # Debug: Check norm before and after layer
            norm_before = mx.linalg.norm(hidden_states).item()
            
            # Log LayerNorm stats for critical layers before processing
            if layer_idx in [0, 13]:
                log_layernorm_stats(hidden_states, layer_idx, "BEFORE")
            
            # Call layer with mask=None to match worker behavior and avoid mask mismatches
            result = layer(hidden_states, mask=None, cache=cache_entry)
            
            # Check if result is a tuple (some layers might return (hidden_states, cache))
            if isinstance(result, tuple):
                hidden_states = result[0]
                logger.info(f"Layer {layer_idx} returned tuple, extracting hidden_states")
            else:
                hidden_states = result
            
            norm_after = mx.linalg.norm(hidden_states).item()
            
            # Log LayerNorm stats for critical layers after processing
            if layer_idx in [0, 13]:
                log_layernorm_stats(hidden_states, layer_idx, "AFTER")
            
            logger.info(f"Layer {layer_idx} norm: before={norm_before:.3f}, after={norm_after:.3f}, ratio={norm_after/norm_before:.2f}x")
            
            # DETAILED DEBUGGING: Print cache state after each layer with sequence length
            cache_entry = cache[layer_idx]
            cache_debug = f"layer {layer_idx}: cache_is_none={cache_entry is None}"
            
            # Check if offset was updated
            if cache_entry is not None:
                logger.info(f"After processing layer {layer_idx}: offset={cache_entry.offset}, step={cache_entry.step}")
            
            if cache_entry is not None:
                if hasattr(cache_entry, 'keys') and cache_entry.keys is not None:
                    seq_len = cache_entry.keys.shape[1] if len(cache_entry.keys.shape) > 1 else 0
                    keys_sum = float(cache_entry.keys.sum().item()) if cache_entry.keys.size > 0 else 0.0
                    cache_debug += f", seq_len={seq_len}, keys_shape={cache_entry.keys.shape}, keys_sum={keys_sum:.3f}"
                elif hasattr(cache_entry, 'keys'):
                    cache_debug += f", keys=None"
                else:
                    cache_debug += f", no_keys_attr"
                    
            logger.info(f"COORDINATOR {cache_debug}")
        
        # Log cache object IDs and offsets AFTER coordinator processing
        cache_obj_info_after = []
        for i, c in enumerate(cache[:14]):  # Only coordinator layers
            if c is not None:
                cache_obj_info_after.append(f"L{i}:id={id(c)},off={c.offset}")
            else:
                cache_obj_info_after.append(f"L{i}:None")
        logger.info(f"Cache IDs/offsets AFTER coordinator: {' '.join(cache_obj_info_after)}")
        
        # Send to workers for remaining layers with KV cache
        if self.world_size > 1:
            for rank, stub in self.worker_stubs.items():
                # Determine which layers this worker should process
                total_layers = len(self.model.model.layers)
                layers_per_worker = total_layers // self.world_size
                worker_layers = list(range(layers_per_worker * rank, 
                                         layers_per_worker * (rank + 1) if rank < self.world_size - 1 else total_layers))
                logger.info(f"Layer assignment check: total={total_layers}, per_worker={layers_per_worker}, worker{rank}_layers={worker_layers}")
                
                # Serialize hidden states
                # Enable compression for better network efficiency
                compress = os.environ.get('COMPRESS_TENSORS', 'true').lower() == 'true'
                hidden_bytes, hidden_metadata = serialize_mlx_array(hidden_states, compress=compress)
                logger.info(f"Serialized metadata: dtype={hidden_metadata['dtype']}, original_dtype={hidden_metadata.get('original_dtype')}, shape={hidden_metadata['shape']}, data_size={len(hidden_bytes)}")
                
                # VALIDATION: Checksum and range validation before sending
                hidden_checksum = hidden_states.sum().item()
                hidden_min = hidden_states.min().item()
                hidden_max = hidden_states.max().item()
                hidden_has_nan = mx.isnan(hidden_states).any().item()
                hidden_has_inf = mx.isinf(hidden_states).any().item()
                logger.info(f"RANK0 SEND: checksum={hidden_checksum:.6f}, range=[{hidden_min:.3f}, {hidden_max:.3f}], nan={hidden_has_nan}, inf={hidden_has_inf}")
                
                # Debug: Log hidden state info before sending to worker - compute checksum on TRANSMITTED data
                import hashlib
                # Compute checksum on the actual bytes being sent
                hidden_sha256 = hashlib.sha256(hidden_bytes).hexdigest()[:16]
                hidden_norm = mx.linalg.norm(hidden_states).item()
                first_4 = hidden_states.flatten()[:4].tolist()
                logger.info(f"COORDINATOR->WORKER: hidden_states sha256={hidden_sha256}, norm={hidden_norm:.3f}, first_4={first_4}, shape={hidden_states.shape}, dtype={hidden_states.dtype}, transmitted_dtype={hidden_metadata['dtype']}")
                
                # CRITICAL: Max-abs check before sending
                flat_tensor = hidden_states.flatten()
                first_16_count = min(16, flat_tensor.shape[0])
                if first_16_count > 0:
                    first_16_mx = flat_tensor[:first_16_count].astype(mx.float32)
                    mx.eval(first_16_mx)
                    first_16_values = [float(first_16_mx[i]) for i in range(first_16_count)]
                    max_abs = max(abs(v) for v in first_16_values)
                    logger.info(f"COORDINATOR SEND MAX-ABS (first 16): {max_abs:.6f}, values={first_16_values[:4]}...")
                
                # Also log the SHA-256 from the metadata (should match)
                metadata_sha256 = hidden_metadata.get('checksum', 'NO_CHECKSUM')[:16]
                logger.info(f"COORDINATOR->WORKER: metadata checksum={metadata_sha256}, data size={len(hidden_bytes)} bytes")
                
                # Serialize ONLY the KV cache entries this worker needs
                cache_bytes = b''
                cache_metadata_proto = None
                
                if cache:
                    # Send COORDINATOR's cache entries (0-13) to worker
                    # The worker needs these to continue processing
                    coordinator_cache = []
                    coordinator_layers = list(range(14))  # Layers 0-13
                    for i in coordinator_layers:
                        coordinator_cache.append(cache[i])
                    
                    logger.info(f"Sending {len(coordinator_cache)} coordinator cache entries to worker (indices: {coordinator_layers})")
                    cache_bytes, cache_meta = serialize_kv_cache(coordinator_cache)
                    logger.info(f"Coordinator cache payload size to worker: {len(cache_bytes) / 1024 / 1024:.2f}MB")
                    
                    # Convert cache metadata to protobuf
                    cache_entries = []
                    for entry_meta in cache_meta['entries']:
                        entry_proto = inference_pb2.KVCacheEntry(
                            type=entry_meta['type'],
                            offset=entry_meta['offset'],
                            size=entry_meta['size']
                        )
                        
                        # Add cache offset/step if available
                        if 'cache_offset' in entry_meta:
                            entry_proto.cache_offset = entry_meta['cache_offset']
                        if 'cache_step' in entry_meta:
                            entry_proto.cache_step = entry_meta['cache_step']
                        
                        if 'keys_meta' in entry_meta:
                            keys_meta = entry_meta['keys_meta']
                            entry_proto.keys_meta.CopyFrom(inference_pb2.TensorMetadata(
                                shape=keys_meta['shape'],
                                dtype=keys_meta['dtype'],
                                compressed=keys_meta.get('compressed', False),
                                original_dtype=keys_meta.get('original_dtype', keys_meta['dtype']),
                                requires_conversion=keys_meta.get('requires_conversion', False)
                            ))
                        if 'values_meta' in entry_meta:
                            values_meta = entry_meta['values_meta']
                            entry_proto.values_meta.CopyFrom(inference_pb2.TensorMetadata(
                                shape=values_meta['shape'],
                                dtype=values_meta['dtype'],
                                compressed=values_meta.get('compressed', False),
                                original_dtype=values_meta.get('original_dtype', values_meta['dtype']),
                                requires_conversion=values_meta.get('requires_conversion', False)
                            ))
                        if 'meta' in entry_meta:
                            meta = entry_meta['meta']
                            entry_proto.meta.CopyFrom(inference_pb2.TensorMetadata(
                                shape=meta['shape'],
                                dtype=meta['dtype'],
                                compressed=meta.get('compressed', False),
                                original_dtype=meta.get('original_dtype', meta['dtype']),
                                requires_conversion=meta.get('requires_conversion', False)
                            ))
                        cache_entries.append(entry_proto)
                    
                    cache_metadata_proto = inference_pb2.KVCacheMetadata(
                        type=cache_meta['type'],
                        length=cache_meta['length'],
                        entries=cache_entries
                    )
                
                # Debug: Log the exact size being sent
                logger.info(f"Creating LayerRequest: hidden_bytes size={len(hidden_bytes)}, metadata dtype={hidden_metadata['dtype']}")
                
                # Create request with KV cache
                request = inference_pb2.LayerRequest(
                    request_id=f"forward-{time.time()}",
                    input_tensor=hidden_bytes,
                    layer_indices=worker_layers,
                    metadata=inference_pb2.TensorMetadata(
                        shape=hidden_metadata['shape'],
                        dtype=hidden_metadata['dtype'],
                        compressed=hidden_metadata.get('compressed', False),
                        original_dtype=hidden_metadata.get('original_dtype', hidden_metadata['dtype']),
                        requires_conversion=hidden_metadata.get('requires_conversion', False)
                    ),
                    kv_cache_data=cache_bytes
                )
                
                if cache_metadata_proto:
                    request.kv_cache_metadata.CopyFrom(cache_metadata_proto)
                
                # Debug: Check the actual request size before sending
                logger.info(f"LayerRequest.input_tensor size before sending: {len(request.input_tensor)} bytes")
                
                # CRITICAL: Log SHA-256 of the exact bytes being sent over the wire
                wire_sha256 = hashlib.sha256(request.input_tensor).hexdigest()[:16]
                logger.info(f"COORDINATOR->WORKER WIRE: SHA-256 of transmitted bytes: {wire_sha256}, size={len(request.input_tensor)}")
                
                # Process on worker
                response = await stub.ProcessLayers(request)
                
                # CRITICAL: Log SHA-256 of received bytes immediately after receiving from wire
                wire_recv_sha256 = hashlib.sha256(response.output_tensor).hexdigest()[:16]
                logger.info(f"COORDINATOR<-WORKER WIRE: SHA-256 of received bytes: {wire_recv_sha256}, size={len(response.output_tensor)}")
                
                # Deserialize response hidden states
                response_metadata = {
                    'shape': list(response.metadata.shape),
                    'dtype': response.metadata.dtype,
                    'compressed': getattr(response.metadata, 'compressed', False),
                    'original_dtype': getattr(response.metadata, 'original_dtype', response.metadata.dtype),
                    'requires_conversion': getattr(response.metadata, 'requires_conversion', False)
                }
                # Log received data checksum before deserialization
                import hashlib
                received_sha256 = hashlib.sha256(response.output_tensor).hexdigest()[:16]
                metadata_sha256 = response_metadata.get('checksum', 'NO_CHECKSUM')[:16]
                logger.info(f"COORDINATOR<-WORKER: received data sha256={received_sha256}, metadata checksum={metadata_sha256}, data size={len(response.output_tensor)} bytes")
                
                hidden_states = deserialize_mlx_array(response.output_tensor, response_metadata)
                
                # VALIDATION: Checksum of received bytes from worker
                received_bytes_sha256 = hashlib.sha256(response.output_tensor).hexdigest()[:16]
                
                # Deserialize tensor and validate
                mx.eval(hidden_states)
                
                # CRITICAL: Max-abs check on first 16 values after deserialization
                flat_tensor = hidden_states.flatten()
                first_16_count = min(16, flat_tensor.shape[0])
                if first_16_count > 0:
                    first_16_mx = flat_tensor[:first_16_count].astype(mx.float32)
                    mx.eval(first_16_mx)
                    first_16_values = [float(first_16_mx[i]) for i in range(first_16_count)]
                    max_abs = max(abs(v) for v in first_16_values)
                    logger.info(f"COORDINATOR RECV MAX-ABS (first 16): {max_abs:.6f}, values={first_16_values[:4]}...")
                
                hidden_norm = mx.linalg.norm(hidden_states).item()
                first_4 = hidden_states.flatten()[:4].tolist()
                received_checksum = hidden_states.sum().item()
                received_min = hidden_states.min().item()
                received_max = hidden_states.max().item()
                received_has_nan = mx.isnan(hidden_states).any().item()
                received_has_inf = mx.isinf(hidden_states).any().item()
                logger.info(f"WORKER->COORDINATOR: bytes_sha256={received_bytes_sha256}, norm={hidden_norm:.3f}, first_4={first_4}, checksum={received_checksum:.6f}, range=[{received_min:.3f}, {received_max:.3f}], nan={received_has_nan}, inf={received_has_inf}")
                
                # Deserialize updated KV cache from worker
                if hasattr(response, 'kv_cache_data') and response.kv_cache_data:
                    worker_cache_metadata = {
                        'type': response.kv_cache_metadata.type,
                        'length': response.kv_cache_metadata.length,
                        'entries': []
                    }
                    for entry in response.kv_cache_metadata.entries:
                        entry_dict = {
                            'type': entry.type,
                            'offset': entry.offset,
                            'size': entry.size
                        }
                        
                        # Extract cache offset/step if present
                        if hasattr(entry, 'cache_offset'):
                            entry_dict['cache_offset'] = entry.cache_offset
                        if hasattr(entry, 'cache_step'):
                            entry_dict['cache_step'] = entry.cache_step
                        if entry.HasField('keys_meta'):
                            entry_dict['keys_meta'] = {
                                'shape': list(entry.keys_meta.shape),
                                'dtype': entry.keys_meta.dtype,
                                'compressed': entry.keys_meta.compressed,
                                'original_dtype': entry.keys_meta.original_dtype,
                                'requires_conversion': entry.keys_meta.requires_conversion
                            }
                        if entry.HasField('values_meta'):
                            entry_dict['values_meta'] = {
                                'shape': list(entry.values_meta.shape),
                                'dtype': entry.values_meta.dtype,
                                'compressed': entry.values_meta.compressed,
                                'original_dtype': entry.values_meta.original_dtype,
                                'requires_conversion': entry.values_meta.requires_conversion
                            }
                        if entry.HasField('meta'):
                            entry_dict['meta'] = {
                                'shape': list(entry.meta.shape),
                                'dtype': entry.meta.dtype,
                                'compressed': entry.meta.compressed,
                                'original_dtype': entry.meta.original_dtype,
                                'requires_conversion': entry.meta.requires_conversion
                            }
                        worker_cache_metadata['entries'].append(entry_dict)
                    
                    # Worker sent a compact cache (only its layers)
                    # We need to map them back to the correct indices
                    worker_compact_cache = deserialize_kv_cache(response.kv_cache_data, worker_cache_metadata)
                    
                    # Update only the worker's layers in our persistent cache
                    worker_layer_idx = 0
                    for layer_idx in worker_layers:
                        if worker_layer_idx < len(worker_compact_cache):
                            # Update this specific layer in-place
                            if cache[layer_idx] is not None and worker_compact_cache[worker_layer_idx] is not None:
                                # Copy the updated cache data
                                cache[layer_idx].keys = worker_compact_cache[worker_layer_idx].keys
                                cache[layer_idx].values = worker_compact_cache[worker_layer_idx].values
                                cache[layer_idx].offset = worker_compact_cache[worker_layer_idx].offset
                                cache[layer_idx].step = worker_compact_cache[worker_layer_idx].step
                            worker_layer_idx += 1
                    
                    logger.info(f"Updated {worker_layer_idx} cache entries from worker {rank}")
                    
                    # The cache has been updated in-place, no need for manual copying
                    # Object identity is preserved!
                    
                    # CRITICAL CHECK 3: Verify worker cache updates are propagated back
                    cache_info_after = []
                    cache_shapes_after = []
                    for i in range(len(cache)):
                        cache_entry = cache[i]
                        if cache_entry is not None:
                            if hasattr(cache_entry, 'keys') and cache_entry.keys is not None:
                                seq_len = cache_entry.keys.shape[1] if len(cache_entry.keys.shape) > 1 else 0
                                full_shape = cache_entry.keys.shape
                                cache_info_after.append(f"({i},{seq_len})")
                                cache_shapes_after.append(f"L{i}:{full_shape}")
                            elif isinstance(cache_entry, (tuple, list)) and len(cache_entry) == 2 and cache_entry[0] is not None:
                                seq_len = cache_entry[0].shape[1] if len(cache_entry[0].shape) > 1 else 0
                                full_shape = cache_entry[0].shape
                                cache_info_after.append(f"({i},{seq_len})")
                                cache_shapes_after.append(f"L{i}:{full_shape}")
                            else:
                                cache_info_after.append(f"({i},0)")
                                cache_shapes_after.append(f"L{i}:empty")
                        else:
                            cache_info_after.append(f"({i},N)")
                            cache_shapes_after.append(f"L{i}:None")
                    
                    # Log all layers to see the full picture
                    logger.info(f"RANK0 cache AFTER worker{rank} - All layers: {' '.join(cache_info_after[:14])}")
                    logger.info(f"RANK0 cache AFTER worker{rank} - Worker layers: {' '.join(cache_info_after[14:])}")
                    
                    # Check if sequence lengths are consistent
                    logger.info(f"After worker{rank} shapes - L0: {cache_shapes_after[0]}, L13: {cache_shapes_after[13]}, L14: {cache_shapes_after[14]}, L27: {cache_shapes_after[27]}")
                
                logger.info(f"Received processed layers from worker {rank}")
        
        # Final norm and output projection (on coordinator)
        norm_before = mx.linalg.norm(hidden_states).item()
        
        # Log stats before final layer norm
        log_layernorm_stats(hidden_states, 28, "BEFORE_FINAL_NORM")
        
        hidden_states = self.model.model.norm(hidden_states)
        norm_after = mx.linalg.norm(hidden_states).item()
        
        # Log stats after final layer norm
        log_layernorm_stats(hidden_states, 28, "AFTER_FINAL_NORM")
        
        logger.info(f"Final layer norm: before={norm_before:.3f}, after={norm_after:.3f}")
        
        # Qwen3 uses tied word embeddings, so use embed_tokens.as_linear for output projection
        if hasattr(self.model, 'args') and getattr(self.model.args, 'tie_word_embeddings', True):
            logits = self.model.model.embed_tokens.as_linear(hidden_states)
        else:
            # Fallback to lm_head if not tied embeddings
            logits = self.model.lm_head(hidden_states)
        
        # Log logits info
        logits_norm = mx.linalg.norm(logits).item()
        logits_first_4 = logits.flatten()[:4].tolist()
        logger.info(f"Logits: norm={logits_norm:.3f}, first_4={logits_first_4}, shape={logits.shape}")
        
        # Final check: Log cache object IDs and offsets after ENTIRE forward pass
        final_cache_info = []
        for i, c in enumerate(cache):
            if c is not None:
                final_cache_info.append(f"L{i}:id={id(c)},off={c.offset}")
            else:
                final_cache_info.append(f"L{i}:None")
        logger.info(f"FINAL cache IDs/offsets: {' '.join(final_cache_info[:14])}")
        logger.info(f"FINAL cache IDs/offsets (worker): {' '.join(final_cache_info[14:])}")
        
        return logits, cache
    
    async def _generate_text_distributed(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using distributed model"""
        logger.info(f"DISTRIBUTED GENERATION STARTING - Using {self.world_size} devices")
        
        try:
            # Implement proper distributed generation with KV cache
            logger.info("Using TRUE DISTRIBUTED generation with proper KV cache handling")
            
            # Tokenize
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            tokens = self.tokenizer.encode(prompt)
            generated = tokens[:]
            
            # Use persistent cache - CRITICAL for maintaining context across tokens
            # Do NOT create a new cache list here!
            
            # Generate tokens one by one with distributed processing
            for token_idx in range(max_tokens):
                if token_idx == 0:
                    # First token: process full prompt to populate all KV caches
                    input_array = mx.array([generated], dtype=mx.int32)
                    logger.info(f"=== TOKEN 0 (PROMPT): processing full prompt with {len(generated)} tokens ===")
                else:
                    # Subsequent tokens: process only the new token (incremental generation)
                    input_array = mx.array([[generated[-1]]], dtype=mx.int32)
                    logger.info(f"=== TOKEN {token_idx}: processing new token {generated[-1]} (total seq len should be {len(generated)}) ===")
                
                # Log mask CRC for this generation step
                step_mask_crc = compute_mask_crc(None)  # We always use mask=None
                logger.info(f"Token {token_idx} mask CRC: {step_mask_crc}")
                
                # Distributed forward pass with persistent cache
                logits, updated_cache = await self.distributed_forward(input_array, mask=None, cache=self.persistent_kv_cache)
                
                # CRITICAL: Assert cache offsets are consistent across all layers
                expected_offset = len(generated)
                total_layers = len(self.persistent_kv_cache)
                
                # Quick per-token assertion as requested
                try:
                    for layer_idx in range(total_layers):
                        cache_entry = self.persistent_kv_cache[layer_idx]
                        if cache_entry is not None and hasattr(cache_entry, 'offset'):
                            assert cache_entry.offset == expected_offset, f"Layer {layer_idx} offset mismatch"
                except AssertionError as e:
                    # Dump all offsets for debugging
                    logger.error(f"CACHE DRIFT ASSERTION FAILED at token {token_idx}: {e}")
                    all_offsets = []
                    for i in range(total_layers):
                        c = self.persistent_kv_cache[i]
                        if c is not None and hasattr(c, 'offset'):
                            all_offsets.append(f"L{i}:{c.offset}")
                        else:
                            all_offsets.append(f"L{i}:None")
                    logger.error(f"All cache offsets (expected {expected_offset}): {' '.join(all_offsets)}")
                    raise ValueError(f"Cache drift detected at token {token_idx}, layer {layer_idx}: offset={cache_entry.offset}, expected={expected_offset}")
                
                # Also keep detailed logging for monitoring
                cache_offset_groups = {
                    'coordinator': [],
                    'worker': [],
                    'mismatched': []
                }
                
                for layer_idx, cache_entry in enumerate(self.persistent_kv_cache):
                    if cache_entry is not None and hasattr(cache_entry, 'offset'):
                        actual_offset = cache_entry.offset
                        if layer_idx < 14:  # Coordinator layers
                            cache_offset_groups['coordinator'].append(f"L{layer_idx}:{actual_offset}")
                        else:  # Worker layers
                            cache_offset_groups['worker'].append(f"L{layer_idx}:{actual_offset}")
                        
                        if actual_offset != expected_offset:
                            cache_offset_groups['mismatched'].append(f"L{layer_idx}:{actual_offset}!={expected_offset}")
                
                # Log per-token cache offset summary
                logger.info(f"Token {token_idx} cache offsets - Coordinator: {cache_offset_groups['coordinator'][:3]}..., Worker: {cache_offset_groups['worker'][:3]}...")
                
                # Sample next token
                logits_last = logits[0, -1, :]
                
                # Apply temperature scaling BEFORE selecting top-k
                temperature = 0.7  # Can be made configurable
                if temperature > 0:
                    logits_scaled = logits_last / temperature
                else:
                    logits_scaled = logits_last
                
                # Get top 5 indices using argsort on SCALED logits
                sorted_indices = mx.argsort(logits_scaled)[::-1]  # Sort descending
                top_5_indices = sorted_indices[:5]
                top_5_values = logits_scaled[top_5_indices]
                
                # Compute softmax probabilities on SCALED logits
                softmax_probs = mx.softmax(logits_scaled)
                top_5_probs = softmax_probs[top_5_indices]
                
                # Create detailed top-5 info with logits and probabilities
                top_5_tokens = []
                for i, idx in enumerate(top_5_indices):
                    token_id = idx.item()
                    logit = logits_last[idx].item()  # Original logit for display
                    scaled_logit = logits_scaled[idx].item()
                    prob = top_5_probs[i].item()
                    decoded = self.tokenizer.decode([token_id])
                    top_5_tokens.append({
                        'rank': i+1,
                        'id': token_id,
                        'logit': logit,
                        'scaled_logit': scaled_logit,
                        'prob': prob,
                        'token': decoded
                    })
                
                # Log top-5 BEFORE sampling (for debugging)
                logger.info(f"Top-5 tokens BEFORE sampling (token {token_idx}, temp={temperature}):")
                for t in top_5_tokens:
                    logger.info(f"  #{t['rank']}: id={t['id']}, logit={t['logit']:.3f}, scaled={t['scaled_logit']:.3f}, prob={t['prob']:.4f}, token='{t['token']}'")
                
                # For now, still use argmax for deterministic behavior during debugging
                # In production, you would sample from the distribution
                next_token = mx.argmax(logits_scaled).item()
                generated.append(next_token)
                
                # Log token generation with top candidates
                logger.info(f"Token {token_idx}: sampled token_id={next_token}, decoded='{self.tokenizer.decode([next_token])}'")
                
                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    logger.info(f"EOS token encountered at position {token_idx}")
                    break
            
            # Decode
            result = self.tokenizer.decode(generated)
            
            # Remove prompt from result
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # For chat template, decode only the generated tokens (excluding original prompt)
                generated_tokens = generated[len(tokens):]
                if generated_tokens:
                    result = self.tokenizer.decode(generated_tokens)
                else:
                    result = ""
            else:
                # For regular tokenizer, remove the original prompt
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
            
            logger.info(f"GENERATION COMPLETE - Generated {len(generated) - len(tokens)} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return f"Error in generation: {str(e)}"
    
    async def initialize_grpc(self):
        """Initialize gRPC communication"""
        if self.rank > 0:
            # Start gRPC server for workers with increased message size and compression
            self.grpc_server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ('grpc.max_send_message_length', 32 * 1024 * 1024),  # 32MB - enough for cache
                    ('grpc.max_receive_message_length', 32 * 1024 * 1024),  # 32MB - enough for cache
                ],
                compression=grpc.Compression.Gzip  # Enable gzip compression
            )
            servicer = WorkerServicer(self.rank, self.model_name)
            inference_pb2_grpc.add_InferenceServiceServicer_to_server(
                servicer, self.grpc_server
            )
            
            port = self.grpc_base_port + self.rank
            self.grpc_server.add_insecure_port(f'[::]:{port}')
            await self.grpc_server.start()
            logger.info(f"Worker {self.rank} gRPC server started on port {port}")
        
        if self.is_coordinator:
            # Connect to all workers
            logger.info("Coordinator connecting to workers...")
            await asyncio.sleep(3)  # Give workers time to start
            
            for rank in range(1, self.world_size):
                # Use MASTER_ADDR if set, otherwise use localhost for local testing
                master_addr = os.environ.get('MASTER_ADDR', 'localhost')
                if rank == 1 and master_addr == 'localhost':
                    # For local testing, always use localhost
                    worker_addr = f"localhost:{self.grpc_base_port + rank}"
                elif rank == 1:
                    # In distributed mode, use the configured address
                    worker_addr = f"192.168.5.2:{self.grpc_base_port + rank}"
                else:
                    worker_addr = f"localhost:{self.grpc_base_port + rank}"
                
                logger.info(f"Connecting to worker {rank} at {worker_addr}")
                channel = grpc.aio.insecure_channel(
                    worker_addr,
                    options=[
                        ('grpc.max_send_message_length', 32 * 1024 * 1024),  # 32MB - enough for cache
                        ('grpc.max_receive_message_length', 32 * 1024 * 1024),  # 32MB - enough for cache
                        ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
                        ('grpc.default_compression_level', 'low'),  # Low level for speed
                    ]
                )
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                self.worker_stubs[rank] = stub
                
                # Test connection
                try:
                    response = await stub.HealthCheck(inference_pb2.Empty())
                    logger.info(f"Worker {rank} responded: healthy={response.healthy}, device={response.device_id}")
                except Exception as e:
                    logger.error(f"Failed to connect to worker {rank}: {e}")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize gRPC
            await self.initialize_grpc()
            
            # Load model on coordinator
            if self.is_coordinator:
                self._load_model()
            
            logger.info(f"Server initialized successfully on rank {self.rank}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app for coordinator"""
        app = FastAPI(title="Distributed MLX Inference API")
        
        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: int = 50
            temperature: float = 0.7
        
        @app.get("/health")
        async def health():
            worker_status = {}
            if self.world_size > 1:
                for rank, stub in self.worker_stubs.items():
                    try:
                        response = await stub.GetDeviceInfo(inference_pb2.Empty())
                        worker_status[f"worker_{rank}"] = {
                            "healthy": True,
                            "device": response.device_id,
                            "assigned_layers": list(response.assigned_layers)
                        }
                    except:
                        worker_status[f"worker_{rank}"] = {"healthy": False}
            
            return {
                "status": "healthy" if self.model else "model_not_loaded",
                "coordinator": {
                    "device": "mini1",
                    "assigned_layers": self.assigned_layers
                },
                "workers": worker_status,
                "total_devices": self.world_size,
                "model": self.model_name
            }
        
        @app.post("/generate")
        async def generate(request: GenerateRequest):
            try:
                # Use distributed generation
                result = await self._generate_text_distributed(
                    request.prompt,
                    request.max_tokens
                )
                return {"generated_text": result, "mode": "distributed"}
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        class ChatMessage(BaseModel):
            role: str
            content: str
        
        class ChatCompletionRequest(BaseModel):
            model: str = self.model_name
            messages: List[ChatMessage] = [ChatMessage(role="user", content="Hi mom!")]
            temperature: float = 0.2
            top_p: float = 0.9
            max_tokens: int = 100
            stream: bool = False
            
            class Config:
                schema_extra = {
                    "example": {
                        "model": "mlx-community/Qwen3-1.7B-8bit",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Hi mom!"
                            }
                        ],
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "max_tokens": 100,
                        "stream": False
                    }
                }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible endpoint using distributed inference"""
            try:
                messages = request.messages
                prompt = messages[-1].content if messages else ""
                max_tokens = request.max_tokens
                
                # Use distributed generation
                result = await self._generate_text_distributed(prompt, max_tokens)
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(result.split()),
                        "total_tokens": len(prompt.split()) + len(result.split())
                    },
                    "system_fingerprint": f"distributed-{self.world_size}-devices"
                }
            except Exception as e:
                logger.error(f"Chat completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def run_coordinator(self):
        """Run as coordinator with API server"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can run API server")
        
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run_worker(self):
        """Run as worker node"""
        logger.info(f"Worker {self.rank} ready and serving")
        
        # Keep worker alive
        try:
            await self.grpc_server.wait_for_termination()
        except KeyboardInterrupt:
            await self.grpc_server.stop(0)
    
    async def run(self):
        """Main run method"""
        # Initialize server
        if not await self.initialize():
            logger.error("Failed to initialize server")
            return
        
        # Run based on role
        if self.is_coordinator:
            await self.run_coordinator()
        else:
            await self.run_worker()
    
    async def cleanup(self):
        """Clean up resources"""
        if self.grpc_server:
            await self.grpc_server.stop(0)
        logger.info("Server shutdown complete")


async def main():
    """Main entry point"""
    server = DistributedServer()
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
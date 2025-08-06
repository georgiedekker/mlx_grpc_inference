#!/usr/bin/env python3
"""
Tensor Parallelism implementation for MLX distributed inference.
Splits model layers across devices for parallel computation.
"""
import mlx.core as mx
import mlx.nn as nn
import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallel execution."""
    device_id: int
    world_size: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    head_dim: int
    
    @property
    def heads_per_device(self) -> int:
        """Number of attention heads per device."""
        return self.num_attention_heads // self.world_size
    
    @property
    def local_head_dim(self) -> int:
        """Hidden dimension for local attention heads."""
        return self.heads_per_device * self.head_dim
    
    @property
    def local_intermediate_size(self) -> int:
        """Intermediate size per device for MLP."""
        return self.intermediate_size // self.world_size


class AllReduceManager:
    """Manages AllReduce operations across devices."""
    
    def __init__(self, device_id: int, world_size: int, worker_stubs: List[Any]):
        self.device_id = device_id
        self.world_size = world_size
        self.worker_stubs = worker_stubs
        self._pending_reduces = {}
        
    async def all_reduce_sum(self, tensor: mx.array, session_id: str) -> mx.array:
        """
        Perform AllReduce SUM operation across all devices.
        Each device contributes its local tensor, result is sum of all.
        """
        if self.world_size == 1:
            return tensor
            
        # For now, implement simple gather-broadcast pattern
        # TODO: Implement ring AllReduce for better efficiency
        
        from src.communication import inference_pb2
        from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
        
        # Serialize tensor for communication
        data, metadata = serialize_mlx_array(tensor, compress=True)
        
        if self.device_id == 0:
            # Coordinator: gather from all workers and sum
            partial_tensors = [tensor]  # Start with local tensor
            
            # Gather from other devices
            for worker_id in range(1, self.world_size):
                stub = self.worker_stubs[worker_id]
                request = inference_pb2.AllReduceRequest()
                request.tensor.data = data
                request.tensor.shape.extend(metadata['shape'])
                request.tensor.dtype = metadata['dtype']
                request.operation = "GATHER"
                request.session_id = session_id
                request.device_id = worker_id
                request.world_size = self.world_size
                
                response = await stub.AllReduce(request)
                partial_tensor = deserialize_mlx_array(response.result_tensor.data, {
                    'shape': list(response.result_tensor.shape),
                    'dtype': response.result_tensor.dtype,
                    'compressed': True,
                    'compression_info': {'algorithm': 'lz4'}
                })
                partial_tensors.append(partial_tensor)
            
            # Sum all partials
            result = mx.sum(mx.stack(partial_tensors), axis=0)
            
            # Broadcast result back to all workers
            result_data, result_metadata = serialize_mlx_array(result, compress=True)
            for worker_id in range(1, self.world_size):
                stub = self.worker_stubs[worker_id]
                request = inference_pb2.AllReduceRequest()
                request.tensor.data = result_data
                request.tensor.shape.extend(result_metadata['shape'])
                request.tensor.dtype = result_metadata['dtype']
                request.operation = "BROADCAST"
                request.session_id = session_id
                request.device_id = 0
                request.world_size = self.world_size
                
                await stub.AllReduce(request)
            
            return result
            
        else:
            # Worker: send to coordinator and wait for broadcast
            stub = self.worker_stubs[0]
            request = inference_pb2.AllReduceRequest()
            request.tensor.data = data
            request.tensor.shape.extend(metadata['shape'])
            request.tensor.dtype = metadata['dtype']
            request.operation = "GATHER"
            request.session_id = session_id
            request.device_id = self.device_id
            request.world_size = self.world_size
            
            response = await stub.AllReduce(request)
            return deserialize_mlx_array(response.result_tensor.data, {
                'shape': list(response.result_tensor.shape),
                'dtype': response.result_tensor.dtype,
                'compressed': True,
                'compression_info': {'algorithm': 'lz4'}
            })


class TensorParallelAttention(nn.Module):
    """
    Tensor-parallel multi-head attention.
    Each device processes a subset of attention heads.
    """
    
    def __init__(self, config: TensorParallelConfig, all_reduce_manager: AllReduceManager):
        super().__init__()
        self.config = config
        self.all_reduce = all_reduce_manager
        
        # Each device gets a subset of attention heads
        self.local_heads = config.heads_per_device
        self.head_dim = config.head_dim
        self.local_hidden = self.local_heads * self.head_dim
        
        # Local projections (column parallel)
        self.q_proj = nn.Linear(config.hidden_size, self.local_hidden, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.local_hidden, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.local_hidden, bias=False)
        
        # Output projection (row parallel)
        self.o_proj = nn.Linear(self.local_hidden, config.hidden_size, bias=False)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    async def __call__(self, x: mx.array, mask: Optional[mx.array] = None, 
                       cache: Optional[Any] = None, session_id: str = "") -> mx.array:
        """
        Forward pass with tensor parallelism.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Attention mask
            cache: KV cache for incremental generation
            session_id: Session ID for AllReduce operations
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Compute local Q, K, V projections
        queries = self.q_proj(x)  # [batch, seq, local_hidden]
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # 2. Reshape for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.local_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.local_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.local_heads, self.head_dim)
        
        # 3. Transpose to [batch, heads, seq, head_dim]
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        
        # 4. Update KV cache if provided
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        
        # 5. Compute attention scores
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # 6. Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # 7. Apply softmax
        scores = mx.softmax(scores, axis=-1)
        
        # 8. Apply attention to values
        output = scores @ values  # [batch, local_heads, seq, head_dim]
        
        # 9. Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.local_hidden)
        
        # 10. Apply output projection
        output = self.o_proj(output)
        
        # 11. AllReduce across devices to sum partial outputs
        output = await self.all_reduce.all_reduce_sum(output, session_id)
        
        return output


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP layer.
    Each device processes a subset of the intermediate dimension.
    """
    
    def __init__(self, config: TensorParallelConfig, all_reduce_manager: AllReduceManager):
        super().__init__()
        self.config = config
        self.all_reduce = all_reduce_manager
        
        # Each device gets a slice of intermediate dimension
        local_intermediate = config.local_intermediate_size
        
        # Gate and up projections (column parallel)
        self.gate_proj = nn.Linear(config.hidden_size, local_intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, local_intermediate, bias=False)
        
        # Down projection (row parallel)
        self.down_proj = nn.Linear(local_intermediate, config.hidden_size, bias=False)
        
    async def __call__(self, x: mx.array, session_id: str = "") -> mx.array:
        """
        Forward pass with tensor parallelism.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            session_id: Session ID for AllReduce operations
        """
        # 1. Compute local gate and up projections
        gate = self.gate_proj(x)
        # SiLU/Swish activation: x * sigmoid(x)
        gate = gate * mx.sigmoid(gate)
        up = self.up_proj(x)
        
        # 2. Element-wise multiplication
        intermediate = gate * up
        
        # 3. Down projection
        output = self.down_proj(intermediate)
        
        # 4. AllReduce to sum outputs from all devices
        output = await self.all_reduce.all_reduce_sum(output, session_id)
        
        return output


class TensorParallelTransformerBlock(nn.Module):
    """
    Complete transformer block with tensor parallelism.
    """
    
    def __init__(self, config: TensorParallelConfig, all_reduce_manager: AllReduceManager):
        super().__init__()
        self.config = config
        
        # Layer normalization (replicated on all devices)
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        
        # Tensor parallel attention and MLP
        self.self_attn = TensorParallelAttention(config, all_reduce_manager)
        self.mlp = TensorParallelMLP(config, all_reduce_manager)
        
    async def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                       cache: Optional[Any] = None, session_id: str = "") -> mx.array:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Attention mask
            cache: KV cache
            session_id: Session ID for AllReduce
        """
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = await self.self_attn(x, mask, cache, session_id)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = await self.mlp(x, session_id)
        x = residual + x
        
        return x


def shard_model_weights(model: Any, world_size: int) -> Dict[int, Dict[str, mx.array]]:
    """
    Shard model weights for tensor parallelism.
    
    Args:
        model: Complete model with all weights
        world_size: Number of devices
        
    Returns:
        Dictionary mapping device_id to weight shards
    """
    shards = {i: {} for i in range(world_size)}
    
    for layer_idx, layer in enumerate(model.model.layers):
        # Shard attention weights
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # Q, K, V projections - split by columns (output dimension)
            for name in ['q_proj', 'k_proj', 'v_proj']:
                if hasattr(attn, name):
                    weight = getattr(attn, name).weight
                    splits = mx.split(weight, world_size, axis=0)  # Split output dim
                    for device_id, split in enumerate(splits):
                        shards[device_id][f"layer.{layer_idx}.self_attn.{name}.weight"] = split
            
            # Output projection - split by rows (input dimension)
            if hasattr(attn, 'o_proj'):
                weight = attn.o_proj.weight
                splits = mx.split(weight, world_size, axis=1)  # Split input dim
                for device_id, split in enumerate(splits):
                    shards[device_id][f"layer.{layer_idx}.self_attn.o_proj.weight"] = split
        
        # Shard MLP weights
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # Gate and up projections - split by columns
            for name in ['gate_proj', 'up_proj']:
                if hasattr(mlp, name):
                    weight = getattr(mlp, name).weight
                    splits = mx.split(weight, world_size, axis=0)  # Split output dim
                    for device_id, split in enumerate(splits):
                        shards[device_id][f"layer.{layer_idx}.mlp.{name}.weight"] = split
            
            # Down projection - split by rows
            if hasattr(mlp, 'down_proj'):
                weight = mlp.down_proj.weight
                splits = mx.split(weight, world_size, axis=1)  # Split input dim
                for device_id, split in enumerate(splits):
                    shards[device_id][f"layer.{layer_idx}.mlp.down_proj.weight"] = split
        
        # Layer norms are replicated (not sharded)
        for name in ['input_layernorm', 'post_attention_layernorm']:
            if hasattr(layer, name):
                norm = getattr(layer, name)
                if hasattr(norm, 'weight'):
                    weight = norm.weight
                    for device_id in range(world_size):
                        shards[device_id][f"layer.{layer_idx}.{name}.weight"] = weight
    
    logger.info(f"Sharded model into {world_size} parts")
    for device_id, shard in shards.items():
        logger.info(f"Device {device_id}: {len(shard)} weight tensors")
    
    return shards
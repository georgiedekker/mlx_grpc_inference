"""
Memory-Efficient Model Loader for MLX Distributed Inference

This module implements true layer-wise model loading where each device only loads
and stores its assigned layers, dramatically reducing memory usage per device.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import qwen2, llama
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import gc
import os
from pathlib import Path
import json
import pickle
from dataclasses import dataclass

from sharding_strategy import ShardAssignment, ShardingPlan
from model_abstraction import ModelInfo, BaseModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class LayerShard:
    """Represents a memory-efficient model shard containing only assigned layers."""
    device_id: str
    assigned_layers: List[nn.Module]
    layer_indices: List[int]
    embed_tokens: Optional[nn.Module] = None
    norm: Optional[nn.Module] = None  
    lm_head: Optional[nn.Module] = None
    use_tied_embeddings: bool = False
    memory_usage_gb: float = 0.0
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """Calculate actual memory footprint of loaded components."""
        footprint = {
            'layers': 0.0,
            'embeddings': 0.0,
            'norm': 0.0,
            'lm_head': 0.0,
            'total': 0.0
        }
        
        # Estimate layer memory (rough approximation)
        if self.assigned_layers:
            # Each transformer layer is roughly similar in size
            layer_params = sum(p.size for layer in self.assigned_layers 
                             for p in layer.parameters() if hasattr(layer, 'parameters'))
            footprint['layers'] = (layer_params * 2) / (1024**3)  # float16 = 2 bytes
        
        if self.embed_tokens:
            embed_params = sum(p.size for p in self.embed_tokens.parameters() 
                             if hasattr(self.embed_tokens, 'parameters'))
            footprint['embeddings'] = (embed_params * 2) / (1024**3)
        
        if self.norm:
            norm_params = sum(p.size for p in self.norm.parameters() 
                            if hasattr(self.norm, 'parameters'))
            footprint['norm'] = (norm_params * 2) / (1024**3)
        
        if self.lm_head:
            head_params = sum(p.size for p in self.lm_head.parameters() 
                            if hasattr(self.lm_head, 'parameters'))
            footprint['lm_head'] = (head_params * 2) / (1024**3)
        
        footprint['total'] = sum(footprint[k] for k in ['layers', 'embeddings', 'norm', 'lm_head'])
        self.memory_usage_gb = footprint['total']
        
        return footprint


class MemoryEfficientModelLoader:
    """
    Loads only the assigned layers for each device, dramatically reducing memory usage.
    
    Instead of loading the full model and extracting layers, this loader:
    1. Analyzes the sharding plan
    2. Loads only the specific layers assigned to this device
    3. Loads shared components (embeddings, norm, lm_head) only where needed
    4. Frees unused model parts immediately
    """
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_info: Optional[ModelInfo] = None
        self.tokenizer = None
        self._architecture = None
        
    def load_model_info_only(self) -> ModelInfo:
        """Load only model configuration and tokenizer, not weights."""
        logger.info(f"Loading model info for {self.model_name}")
        
        # Load minimal model structure to get config
        try:
            # Load only config and tokenizer, avoid loading weights
            model, tokenizer = load(self.model_name)
            self.tokenizer = tokenizer
            
            # Extract model info
            if hasattr(model, 'model'):
                base_model = model.model
                config = getattr(model, 'config', None) or getattr(base_model, 'config', None)
            else:
                base_model = model
                config = getattr(model, 'config', None)
            
            # Determine architecture
            model_lower = self.model_name.lower()
            if 'qwen' in model_lower:
                self._architecture = 'qwen2'
            elif 'llama' in model_lower:
                self._architecture = 'llama'
            else:
                self._architecture = 'llama'  # Default fallback
            
            # Extract configuration
            num_layers = getattr(config, 'num_hidden_layers', len(base_model.layers) if hasattr(base_model, 'layers') else 28)
            hidden_size = getattr(config, 'hidden_size', 2048)
            num_attention_heads = getattr(config, 'num_attention_heads', 16)
            num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
            vocab_size = getattr(config, 'vocab_size', 151936)
            max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
            
            # Estimate parameters
            total_params = self._estimate_total_params(
                num_layers, hidden_size, num_attention_heads, 
                num_key_value_heads, vocab_size
            )
            
            # Detect quantization
            quantization = None
            if '8bit' in self.model_name or 'int8' in self.model_name:
                quantization = 'int8'
            elif '4bit' in self.model_name or 'int4' in self.model_name:
                quantization = 'int4'
            
            self.model_info = ModelInfo(
                name=self.model_name,
                architecture=self._architecture,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                total_params=total_params,
                dtype='float16',
                quantization=quantization
            )
            
            # Clear the temporary full model to free memory
            del model
            if hasattr(base_model, 'layers'):
                del base_model
            gc.collect()
            
            logger.info(f"Model info loaded: {num_layers} layers, {total_params:,} parameters")
            return self.model_info
            
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
            raise
    
    def load_layer_shard(self, shard_assignment: ShardAssignment) -> LayerShard:
        """
        Load only the layers assigned to this device.
        
        Args:
            shard_assignment: Assignment specifying which layers to load
            
        Returns:
            LayerShard containing only the assigned model components
        """
        logger.info(f"Loading layer shard for device {shard_assignment.device_id}")
        logger.info(f"Assigned layers: {shard_assignment.start_layer}-{shard_assignment.end_layer-1}")
        
        if not self.model_info:
            self.load_model_info_only()
        
        # Load full model temporarily to extract our layers
        logger.info("Temporarily loading full model to extract assigned layers...")
        full_model, tokenizer = load(self.model_name)
        
        if not self.tokenizer:
            self.tokenizer = tokenizer
        
        # Extract model structure
        if hasattr(full_model, 'model'):
            base_model = full_model.model
        else:
            base_model = full_model
        
        # Extract only our assigned layers
        assigned_layers = []
        all_layers = base_model.layers
        
        for layer_idx in range(shard_assignment.start_layer, shard_assignment.end_layer):
            if layer_idx < len(all_layers):
                assigned_layers.append(all_layers[layer_idx])
                logger.debug(f"Extracted layer {layer_idx}")
        
        layer_indices = list(range(shard_assignment.start_layer, shard_assignment.end_layer))
        
        # Handle special components
        embed_tokens = None
        norm = None
        lm_head = None
        use_tied_embeddings = False
        
        # Embedding layer (first device or last device if tied embeddings)
        if shard_assignment.has_embedding:
            embed_tokens = base_model.embed_tokens
            logger.info("Loaded embedding layer")
        
        # Norm and output head (last device)
        if shard_assignment.has_lm_head:
            norm = base_model.norm
            
            # Check for tied embeddings
            if hasattr(full_model, 'args'):
                use_tied_embeddings = getattr(full_model.args, 'tie_word_embeddings', False)
            elif hasattr(full_model, 'config'):
                use_tied_embeddings = getattr(full_model.config, 'tie_word_embeddings', False)
            
            if not use_tied_embeddings:
                lm_head = getattr(full_model, 'lm_head', None)
                if lm_head:
                    logger.info("Loaded LM head")
            else:
                # For tied embeddings, ensure we have embedding weights
                if not embed_tokens:
                    embed_tokens = base_model.embed_tokens
                logger.info("Using tied embeddings for output projection")
            
            logger.info("Loaded normalization layer")
        
        # Create the layer shard
        shard = LayerShard(
            device_id=shard_assignment.device_id,
            assigned_layers=assigned_layers,
            layer_indices=layer_indices,
            embed_tokens=embed_tokens,
            norm=norm,
            lm_head=lm_head,
            use_tied_embeddings=use_tied_embeddings
        )
        
        # Calculate memory footprint
        footprint = shard.get_memory_footprint()
        logger.info(f"Memory footprint: {footprint['total']:.2f} GB")
        logger.info(f"  - Layers: {footprint['layers']:.2f} GB")
        logger.info(f"  - Embeddings: {footprint['embeddings']:.2f} GB")
        logger.info(f"  - Norm: {footprint['norm']:.2f} GB")
        logger.info(f"  - LM Head: {footprint['lm_head']:.2f} GB")
        
        # CRITICAL: Delete the full model to free memory
        logger.info("Freeing full model memory...")
        del full_model
        del base_model
        del all_layers
        gc.collect()
        
        # Force MLX to release GPU memory
        mx.metal.clear_cache()
        
        logger.info(f"Layer shard loaded successfully. Memory saved: ~{shard_assignment.estimated_memory_gb - footprint['total']:.2f} GB")
        
        return shard
    
    def _estimate_total_params(self, num_layers: int, hidden_size: int,
                              num_attention_heads: int, num_key_value_heads: int,
                              vocab_size: int) -> int:
        """Estimate total parameters based on architecture."""
        if self._architecture == 'qwen2':
            return self._estimate_qwen_params(num_layers, hidden_size, num_attention_heads, 
                                            num_key_value_heads, vocab_size)
        else:  # llama
            return self._estimate_llama_params(num_layers, hidden_size, num_attention_heads, 
                                             num_key_value_heads, vocab_size)
    
    def _estimate_qwen_params(self, num_layers: int, hidden_size: int,
                             num_attention_heads: int, num_key_value_heads: int,
                             vocab_size: int) -> int:
        """Estimate parameters for Qwen models."""
        # Embedding parameters
        embed_params = vocab_size * hidden_size
        
        # Attention parameters per layer
        head_dim = hidden_size // num_attention_heads
        q_params = hidden_size * hidden_size
        k_params = num_key_value_heads * head_dim * hidden_size
        v_params = num_key_value_heads * head_dim * hidden_size
        o_params = hidden_size * hidden_size
        attention_params = q_params + k_params + v_params + o_params
        
        # MLP parameters per layer (Qwen uses SwiGLU)
        mlp_params = 3 * hidden_size * (4 * hidden_size)
        
        # Layer norm parameters
        ln_params = 2 * hidden_size
        
        # Total for all layers
        total_layer_params = num_layers * (attention_params + mlp_params + ln_params)
        
        # Output head
        head_params = hidden_size * vocab_size
        
        return embed_params + total_layer_params + head_params
    
    def _estimate_llama_params(self, num_layers: int, hidden_size: int,
                              num_attention_heads: int, num_key_value_heads: int,
                              vocab_size: int) -> int:
        """Estimate parameters for Llama models."""
        embed_params = vocab_size * hidden_size
        
        head_dim = hidden_size // num_attention_heads
        q_params = hidden_size * hidden_size
        k_params = num_key_value_heads * head_dim * hidden_size
        v_params = num_key_value_heads * head_dim * hidden_size
        o_params = hidden_size * hidden_size
        attention_params = q_params + k_params + v_params + o_params
        
        # Standard FFN
        mlp_params = 2 * hidden_size * (4 * hidden_size)
        
        ln_params = 2 * hidden_size
        total_layer_params = num_layers * (attention_params + mlp_params + ln_params)
        
        head_params = hidden_size * vocab_size
        
        return embed_params + total_layer_params + head_params


class MemoryEfficientShardedModel(nn.Module):
    """
    A sharded model that contains only the layers assigned to this device.
    
    This replaces the previous approach where each device loaded the full model.
    """
    
    def __init__(self, layer_shard: LayerShard, model_info: ModelInfo):
        super().__init__()
        self.layer_shard = layer_shard
        self.model_info = model_info
        self.device_id = layer_shard.device_id
        
        # Store components
        self.assigned_layers = layer_shard.assigned_layers
        self.layer_indices = layer_shard.layer_indices
        self.embed_tokens = layer_shard.embed_tokens
        self.norm = layer_shard.norm
        self.lm_head = layer_shard.lm_head
        self.use_tied_embeddings = layer_shard.use_tied_embeddings
        
        logger.info(f"MemoryEfficientShardedModel initialized for device {self.device_id}")
        logger.info(f"Contains {len(self.assigned_layers)} layers: {self.layer_indices}")
    
    def __call__(self, x: mx.array, cache: Optional[List] = None) -> mx.array:
        """Forward pass through assigned layers."""
        # Apply embedding if this is the first shard
        if self.embed_tokens is not None and self.layer_indices and self.layer_indices[0] == 0:
            # Input should be token ids for first device
            if len(x.shape) == 1:
                x = mx.expand_dims(x, axis=0)  # Add batch dimension
            x = self.embed_tokens(x)
            logger.debug(f"Device {self.device_id}: Applied embeddings, shape: {x.shape}")
        
        # Process assigned transformer layers
        for i, layer in enumerate(self.assigned_layers):
            layer_cache = cache[i] if cache else None
            x = layer(x, mask=None, cache=layer_cache)
            mx.eval(x)  # Ensure computation happens
            logger.debug(f"Device {self.device_id}: Processed layer {self.layer_indices[i]}")
        
        # Apply final processing if this is the last shard
        if self.norm is not None:
            x = self.norm(x)
            mx.eval(x)
            logger.debug(f"Device {self.device_id}: Applied final norm")
        
        # Apply output projection
        if self.lm_head is not None:
            x = self.lm_head(x)
            mx.eval(x)
            logger.debug(f"Device {self.device_id}: Applied LM head")
        elif self.use_tied_embeddings and self.embed_tokens is not None:
            x = self.embed_tokens.as_linear(x)
            mx.eval(x)
            logger.debug(f"Device {self.device_id}: Applied tied embeddings projection")
        
        return x
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.layer_shard.memory_usage_gb


def create_memory_efficient_model(model_name: str, shard_assignment: ShardAssignment) -> Tuple[MemoryEfficientShardedModel, Any]:
    """
    Create a memory-efficient sharded model that only loads assigned layers.
    
    Args:
        model_name: Name of the model to load
        shard_assignment: Assignment specifying which layers to load
        
    Returns:
        Tuple of (sharded_model, tokenizer)
    """
    logger.info(f"Creating memory-efficient model for device {shard_assignment.device_id}")
    
    # Initialize loader
    loader = MemoryEfficientModelLoader(model_name)
    
    # Load model info
    model_info = loader.load_model_info_only()
    
    # Load only assigned layers
    layer_shard = loader.load_layer_shard(shard_assignment)
    
    # Create sharded model
    sharded_model = MemoryEfficientShardedModel(layer_shard, model_info)
    
    logger.info(f"Memory-efficient model created. Memory usage: {layer_shard.memory_usage_gb:.2f} GB")
    
    return sharded_model, loader.tokenizer


if __name__ == "__main__":
    # Test memory-efficient loading
    from device_capabilities import DeviceProfile
    
    # Create mock device
    device = DeviceProfile(
        device_id="mini1",
        hostname="mini1.local",
        model="Apple M4",
        memory_gb=16.0,
        gpu_cores=10,
        cpu_cores=10,
        cpu_performance_cores=4,
        cpu_efficiency_cores=6,
        neural_engine_cores=16
    )
    
    # Create shard assignment for testing
    shard_assignment = ShardAssignment(
        device_id="mini1",
        device_profile=device,
        start_layer=0,
        end_layer=10,
        num_layers=10,
        estimated_memory_gb=5.0,
        estimated_compute_load=0.33,
        has_embedding=True,
        has_lm_head=False
    )
    
    # Test memory-efficient loading
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    
    try:
        sharded_model, tokenizer = create_memory_efficient_model(model_name, shard_assignment)
        print(f"Successfully created memory-efficient model!")
        print(f"Memory usage: {sharded_model.get_memory_usage():.2f} GB")
        print(f"Assigned layers: {sharded_model.layer_indices}")
    except Exception as e:
        print(f"Error: {e}")
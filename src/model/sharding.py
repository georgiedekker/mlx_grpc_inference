"""
Model sharding utilities for distributing layers across devices.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from ..core.config import ClusterConfig, DeviceConfig

logger = logging.getLogger(__name__)


@dataclass
class LayerShard:
    """Information about a model shard."""
    device_id: str
    layer_indices: List[int]
    start_layer: int
    end_layer: int  # exclusive
    

class ModelShardingStrategy:
    """Strategy for sharding a model across devices."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.shards = self._create_shards()
    
    def _create_shards(self) -> Dict[str, LayerShard]:
        """Create layer shards based on configuration."""
        shards = {}
        
        for device_id, layer_indices in self.config.model.layer_distribution.items():
            if layer_indices:
                shard = LayerShard(
                    device_id=device_id,
                    layer_indices=layer_indices,
                    start_layer=min(layer_indices),
                    end_layer=max(layer_indices) + 1
                )
                shards[device_id] = shard
                logger.info(f"Device {device_id} assigned layers {layer_indices}")
        
        return shards
    
    def get_device_shard(self, device_id: str) -> Optional[LayerShard]:
        """Get shard information for a specific device."""
        return self.shards.get(device_id)
    
    def get_layer_device(self, layer_index: int) -> Optional[str]:
        """Get the device ID that should process a specific layer."""
        for device_id, shard in self.shards.items():
            if layer_index in shard.layer_indices:
                return device_id
        return None
    
    def validate_coverage(self) -> bool:
        """Validate that all layers are assigned."""
        all_layers = set()
        for shard in self.shards.values():
            all_layers.update(shard.layer_indices)
        
        expected_layers = set(range(self.config.model.total_layers))
        missing = expected_layers - all_layers
        extra = all_layers - expected_layers
        
        if missing:
            logger.error(f"Missing layer assignments: {sorted(missing)}")
        if extra:
            logger.error(f"Extra layer assignments: {sorted(extra)}")
        
        return len(missing) == 0 and len(extra) == 0


class ModelSharder:
    """Handles the actual sharding of model weights."""
    
    def __init__(self, model: nn.Module, sharding_strategy: ModelShardingStrategy):
        self.model = model
        self.strategy = sharding_strategy
        self.model_config = self._analyze_model()
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model structure."""
        config = {
            'total_layers': 0,
            'layer_types': {},
            'total_params': 0
        }
        
        # Count transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            config['total_layers'] = len(layers)
            
            # Analyze layer types
            for i, layer in enumerate(layers):
                layer_type = type(layer).__name__
                if layer_type not in config['layer_types']:
                    config['layer_types'][layer_type] = []
                config['layer_types'][layer_type].append(i)
        
        logger.info(f"Model analysis: {config['total_layers']} layers found")
        return config
    
    def extract_device_weights(self, device_id: str) -> Dict[str, mx.array]:
        """Extract weights for a specific device's layers."""
        shard = self.strategy.get_device_shard(device_id)
        if not shard:
            logger.warning(f"No shard found for device {device_id}")
            return {}
        
        weights = {}
        
        # Extract embeddings (all devices need these for now)
        if hasattr(self.model.model, 'embed_tokens'):
            weights['embed_tokens'] = self.model.model.embed_tokens.weight
        
        # Extract assigned layers
        if hasattr(self.model.model, 'layers'):
            for layer_idx in shard.layer_indices:
                if layer_idx < len(self.model.model.layers):
                    layer = self.model.model.layers[layer_idx]
                    layer_weights = self._extract_layer_weights(layer, layer_idx)
                    weights.update(layer_weights)
        
        # Extract output layers (only last device needs these)
        if device_id == self._get_last_device():
            if hasattr(self.model.model, 'norm'):
                weights['norm.weight'] = self.model.model.norm.weight
            if hasattr(self.model, 'lm_head'):
                weights['lm_head.weight'] = self.model.lm_head.weight
        
        logger.info(f"Extracted {len(weights)} weight tensors for device {device_id}")
        return weights
    
    def _extract_layer_weights(self, layer: nn.Module, layer_idx: int) -> Dict[str, mx.array]:
        """Extract weights from a single layer."""
        weights = {}
        prefix = f'layers.{layer_idx}'
        
        # Self-attention weights
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if hasattr(attn, 'q_proj'):
                weights[f'{prefix}.self_attn.q_proj.weight'] = attn.q_proj.weight
            if hasattr(attn, 'k_proj'):
                weights[f'{prefix}.self_attn.k_proj.weight'] = attn.k_proj.weight
            if hasattr(attn, 'v_proj'):
                weights[f'{prefix}.self_attn.v_proj.weight'] = attn.v_proj.weight
            if hasattr(attn, 'o_proj'):
                weights[f'{prefix}.self_attn.o_proj.weight'] = attn.o_proj.weight
        
        # MLP weights
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'gate_proj'):
                weights[f'{prefix}.mlp.gate_proj.weight'] = mlp.gate_proj.weight
            if hasattr(mlp, 'up_proj'):
                weights[f'{prefix}.mlp.up_proj.weight'] = mlp.up_proj.weight
            if hasattr(mlp, 'down_proj'):
                weights[f'{prefix}.mlp.down_proj.weight'] = mlp.down_proj.weight
        
        # Layer norms
        if hasattr(layer, 'input_layernorm'):
            weights[f'{prefix}.input_layernorm.weight'] = layer.input_layernorm.weight
        if hasattr(layer, 'post_attention_layernorm'):
            weights[f'{prefix}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight
        
        return weights
    
    def _get_last_device(self) -> str:
        """Get the device ID that processes the last layers."""
        max_layer = -1
        last_device = None
        
        for device_id, shard in self.strategy.shards.items():
            if shard.layer_indices:
                device_max = max(shard.layer_indices)
                if device_max > max_layer:
                    max_layer = device_max
                    last_device = device_id
        
        return last_device
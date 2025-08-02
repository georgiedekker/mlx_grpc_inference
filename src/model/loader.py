"""
Model loading utilities for distributed inference.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from ..core.config import ClusterConfig
from .sharding import ModelShardingStrategy, ModelSharder

logger = logging.getLogger(__name__)


class DistributedModelLoader:
    """Handles loading and sharding of models for distributed inference."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.strategy = ModelShardingStrategy(config)
        
    def load_model_shard(self, device_id: str) -> Tuple[Optional[nn.Module], Optional[Any]]:
        """
        Load only the model shard for a specific device.
        
        Args:
            device_id: ID of the device to load shard for
            
        Returns:
            Tuple of (model shard, tokenizer)
        """
        try:
            # Validate sharding strategy
            if not self.strategy.validate_coverage():
                raise ValueError("Invalid layer distribution in configuration")
            
            # Get shard info
            shard = self.strategy.get_device_shard(device_id)
            if not shard:
                logger.warning(f"No shard assigned to device {device_id}")
                return None, None
            
            logger.info(f"Loading model shard for {device_id}: layers {shard.layer_indices}")
            
            # Load full model first (we'll optimize this later)
            model, tokenizer = load(self.config.model.name)
            
            # For now, return the full model
            # In a production system, we would extract only the needed layers
            logger.info(f"Model shard loaded for {device_id}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model shard for {device_id}: {e}")
            raise
    
    def load_full_model(self) -> Tuple[nn.Module, Any]:
        """
        Load the complete model (for coordinator).
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading full model: {self.config.model.name}")
        
        try:
            model, tokenizer = load(self.config.model.name)
            
            # Validate model matches configuration
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                actual_layers = len(model.model.layers)
                if actual_layers != self.config.model.total_layers:
                    logger.warning(
                        f"Model layer count mismatch: "
                        f"expected {self.config.model.total_layers}, got {actual_layers}"
                    )
            
            logger.info("Full model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'name': self.config.model.name,
            'total_layers': 0,
            'total_parameters': 0,
            'layer_types': []
        }
        
        # Count layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            info['total_layers'] = len(model.model.layers)
            
            # Get unique layer types
            layer_types = set()
            for layer in model.model.layers:
                layer_types.add(type(layer).__name__)
            info['layer_types'] = sorted(list(layer_types))
        
        # Count parameters
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.size
        info['total_parameters'] = total_params
        info['total_parameters_millions'] = round(total_params / 1e6, 2)
        
        return info


class ModelCache:
    """Cache for loaded model shards."""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[nn.Module, Any]] = {}
        
    def get(self, device_id: str) -> Optional[Tuple[nn.Module, Any]]:
        """Get cached model shard."""
        return self.cache.get(device_id)
    
    def put(self, device_id: str, model: nn.Module, tokenizer: Any):
        """Cache a model shard."""
        self.cache[device_id] = (model, tokenizer)
        
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
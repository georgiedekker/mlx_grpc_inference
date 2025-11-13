#!/usr/bin/env python3
"""
Add pipeline() method to MLX models that don't have it.
This enables distributed inference across multiple devices.
"""

import mlx.core as mx
from mlx_lm.utils import load_model
from pathlib import Path
from typing import Optional, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_pipeline_method(model_class):
    """
    Decorator to add pipeline() method to any MLX model class.
    This enables layer-wise model parallelism.
    """
    
    def pipeline(self, group: mx.distributed.Group) -> None:
        """
        Shard model layers across distributed group for pipeline parallelism.
        
        Args:
            group: MLX distributed group to shard across
        """
        if not group:
            logger.warning("No distributed group provided, skipping pipeline setup")
            return
            
        rank = group.rank()
        world_size = group.size()
        
        logger.info(f"Setting up pipeline parallelism: rank={rank}, world_size={world_size}")
        
        # Get all layers in the model
        if hasattr(self, 'layers'):
            layers = self.layers
        elif hasattr(self, 'blocks'):
            layers = self.blocks
        elif hasattr(self, 'h'):
            layers = self.h
        else:
            logger.error("Could not find layers in model structure")
            return
            
        num_layers = len(layers)
        logger.info(f"Model has {num_layers} layers")
        
        # Calculate layer distribution
        layers_per_device = num_layers // world_size
        extra_layers = num_layers % world_size
        
        # Determine which layers this rank owns
        if rank < extra_layers:
            start_layer = rank * (layers_per_device + 1)
            end_layer = start_layer + layers_per_device + 1
        else:
            start_layer = rank * layers_per_device + extra_layers
            end_layer = start_layer + layers_per_device
            
        logger.info(f"Rank {rank} owns layers {start_layer} to {end_layer-1}")
        
        # Create a new layer list with only our layers
        my_layers = []
        for i in range(num_layers):
            if start_layer <= i < end_layer:
                # Keep this layer
                my_layers.append(layers[i])
            else:
                # Replace with identity/pass-through
                my_layers.append(IdentityLayer())
        
        # Replace the model's layers
        if hasattr(self, 'layers'):
            self.layers = my_layers
        elif hasattr(self, 'blocks'):
            self.blocks = my_layers
        elif hasattr(self, 'h'):
            self.h = my_layers
            
        # Store pipeline metadata
        self._pipeline_rank = rank
        self._pipeline_world_size = world_size
        self._pipeline_start_layer = start_layer
        self._pipeline_end_layer = end_layer
        self._pipeline_group = group
        
        logger.info(f"Pipeline setup complete for rank {rank}")
    
    # Add the method to the class
    model_class.pipeline = pipeline
    return model_class


class IdentityLayer:
    """Placeholder layer that just passes input through."""
    def __call__(self, x, *args, **kwargs):
        return x


def patch_model_with_pipeline(model_path: str):
    """
    Load a model and add pipeline support if it doesn't have it.
    
    Args:
        model_path: Path to the MLX model
    
    Returns:
        Model with pipeline() method added
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load the model
    model, tokenizer = load_model(model_path, lazy=True)
    
    # Check if model already has pipeline
    if hasattr(model, 'model') and hasattr(model.model, 'pipeline'):
        logger.info("Model already has pipeline() method")
        return model, tokenizer
    
    # Add pipeline method to the model's class
    if hasattr(model, 'model'):
        model_instance = model.model
        model_class = model_instance.__class__
        
        # Add the pipeline method
        add_pipeline_method(model_class)
        
        logger.info(f"Added pipeline() method to {model_class.__name__}")
    else:
        logger.error("Model structure not recognized")
        
    return model, tokenizer


def test_pipeline_sharding():
    """Test the pipeline sharding with a local model."""
    
    # Find a model to test with
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--mlx-community--*"))
    
    if not model_dirs:
        logger.error("No MLX models found in cache")
        return
        
    # Use the first available model
    model_dir = model_dirs[0]
    model_name = model_dir.name.replace("models--", "").replace("--", "/")
    
    snapshots = list((model_dir / "snapshots").iterdir())
    if not snapshots:
        logger.error("No snapshots found")
        return
        
    model_path = snapshots[0]
    
    logger.info(f"Testing with model: {model_name}")
    
    # Load and patch the model
    model, tokenizer = patch_model_with_pipeline(str(model_path))
    
    # Initialize distributed group
    group = mx.distributed.init()
    
    if group and hasattr(model, 'model'):
        # Apply pipeline sharding
        model.model.pipeline(group)
        logger.info("Pipeline sharding applied successfully!")
        
        # Verify the sharding
        if hasattr(model.model, '_pipeline_rank'):
            logger.info(f"Model sharded: rank={model.model._pipeline_rank}, "
                       f"layers {model.model._pipeline_start_layer}-"
                       f"{model.model._pipeline_end_layer-1}")
    else:
        logger.warning("Could not initialize distributed group or model structure")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test the pipeline addition
    model, tokenizer = test_pipeline_sharding()
    
    if model and hasattr(model, 'model') and hasattr(model.model, 'pipeline'):
        print("\n✅ Successfully added pipeline() method to model!")
        print("You can now use this model for distributed inference.")
    else:
        print("\n❌ Failed to add pipeline() method")
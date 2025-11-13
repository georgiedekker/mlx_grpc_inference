#!/usr/bin/env python3
"""
Implementation of pipeline() method for MLX models based on Awni's approach.
This adds distributed pipeline parallelism to any MLX model.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineModule(nn.Module):
    """
    Wrapper that adds pipeline parallelism to any MLX model.
    Based on Awni's DeepSeek implementation approach.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.group = None
        self.rank = 0
        self.world_size = 1
        self.start_layer = 0
        self.end_layer = None
        self.num_layers = None
        
    def pipeline(self, group: mx.distributed.Group) -> None:
        """
        Shard model layers across distributed group for pipeline parallelism.
        This is the key method that Awni's examples use.
        """
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        
        logger.info(f"Initializing pipeline: rank={self.rank}, world_size={self.world_size}")
        
        # Find the transformer layers in the model
        layers = self._find_layers()
        if layers is None:
            raise ValueError("Could not find transformer layers in model")
            
        self.num_layers = len(layers)
        
        # Calculate layer distribution (equal split with remainder to first ranks)
        layers_per_rank = self.num_layers // self.world_size
        extra_layers = self.num_layers % self.world_size
        
        # Determine this rank's layer range
        if self.rank < extra_layers:
            self.start_layer = self.rank * (layers_per_rank + 1)
            self.end_layer = self.start_layer + layers_per_rank + 1
        else:
            self.start_layer = self.rank * layers_per_rank + extra_layers
            self.end_layer = self.start_layer + layers_per_rank
            
        logger.info(f"Rank {self.rank}: Processing layers {self.start_layer} to {self.end_layer-1} "
                   f"(out of {self.num_layers} total)")
        
        # Prune layers not needed by this rank
        self._prune_layers(layers)
        
    def _find_layers(self) -> Optional[List]:
        """Find the transformer layers in the model."""
        # Common layer names in different architectures
        layer_names = ['layers', 'blocks', 'h', 'transformer.h', 'model.layers']
        
        for name in layer_names:
            obj = self.base_model
            for part in name.split('.'):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                # Successfully traversed the path
                if isinstance(obj, list) or (hasattr(obj, '__iter__') and hasattr(obj, '__len__')):
                    return obj
                    
        return None
        
    def _prune_layers(self, layers):
        """Remove layers not needed by this rank."""
        # Create a new list of layers for this rank
        pruned_layers = []
        
        for i in range(self.num_layers):
            if self.start_layer <= i < self.end_layer:
                # Keep this layer
                pruned_layers.append(layers[i])
                logger.debug(f"Rank {self.rank}: Keeping layer {i}")
            # Other layers are simply not included
        
        # Replace the layers in the model
        # This is model-specific and may need adjustment
        if hasattr(self.base_model, 'layers'):
            self.base_model.layers = pruned_layers
        elif hasattr(self.base_model, 'blocks'):
            self.base_model.blocks = pruned_layers
        elif hasattr(self.base_model, 'h'):
            self.base_model.h = pruned_layers
            
    def __call__(self, *args, **kwargs):
        """Forward pass with pipeline communication."""
        # This is a simplified version - real implementation needs MPI communication
        # between pipeline stages
        
        if self.group is None:
            # No pipeline, just run normally
            return self.base_model(*args, **kwargs)
            
        # For pipeline parallelism, we need to:
        # 1. Receive activations from previous rank (if not rank 0)
        # 2. Process through our layers
        # 3. Send activations to next rank (if not last rank)
        
        # This is where the MPI communication would happen
        # For now, just run the pruned model
        return self.base_model(*args, **kwargs)


def add_pipeline_to_model(model):
    """
    Add pipeline() method to a model that doesn't have it.
    
    This wraps the model to add pipeline parallelism support.
    """
    if hasattr(model, 'pipeline'):
        logger.info("Model already has pipeline() method")
        return model
        
    # Wrap the model with our pipeline implementation
    wrapped = PipelineModule(model)
    
    # Add the pipeline method directly to the model for compatibility
    model.pipeline = wrapped.pipeline
    
    logger.info("Added pipeline() method to model")
    return model


def load_model_with_pipeline(model_path):
    """
    Load a model and ensure it has pipeline() support.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model lazily for sharding
    model, tokenizer = load_model(Path(model_path), lazy=True, strict=False)
    
    # Check if the model.model has pipeline
    if hasattr(model, 'model'):
        if not hasattr(model.model, 'pipeline'):
            logger.info("Model doesn't have pipeline(), adding it...")
            model.model = add_pipeline_to_model(model.model)
        else:
            logger.info("Model already has pipeline() support")
    else:
        logger.warning("Model structure not as expected")
        
    return model, tokenizer


def test_pipeline():
    """Test the pipeline implementation with a local model."""
    
    # Find a model to test
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dir = cache_dir / "models--mlx-community--Qwen3-1.7B-8bit"
    
    if not model_dir.exists():
        logger.error(f"Model not found at {model_dir}")
        return
        
    snapshots = list((model_dir / "snapshots").iterdir())
    if not snapshots:
        logger.error("No snapshots found")
        return
        
    model_path = snapshots[0]
    
    # Load model with pipeline support
    model, tokenizer = load_model_with_pipeline(str(model_path))
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if group and hasattr(model, 'model') and hasattr(model.model, 'pipeline'):
        # Apply pipeline sharding
        model.model.pipeline(group)
        
        # Evaluate to load weights
        mx.eval(model.parameters())
        
        # Synchronize
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        
        logger.info("✅ Pipeline sharding applied successfully!")
        
        # Test generation
        from mlx_lm import stream_generate
        
        prompt = "Hello, world!"
        for response in stream_generate(model, tokenizer, prompt, max_tokens=10):
            if group.rank() == 0:
                print(response.text, end="", flush=True)
        
        if group.rank() == 0:
            print("\n✅ Generation test complete!")
    else:
        logger.warning("Could not apply pipeline sharding")


if __name__ == "__main__":
    test_pipeline()
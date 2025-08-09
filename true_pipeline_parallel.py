#!/usr/bin/env python3
"""
True Pipeline Parallelism for MLX Models
Implements proper activation passing between pipeline stages
"""
import mlx.core as mx
import mlx.nn as nn
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def add_true_pipeline_parallelism(model_instance):
    """
    Add true pipeline parallelism with MPI communication.
    Each rank processes different layers and passes activations.
    """
    
    def pipeline(self, group=None):
        """Setup pipeline parallelism - shard layers across ranks."""
        if group is None:
            group = mx.distributed.init()
        
        if not group or group.size() == 1:
            return
        
        rank = group.rank()
        world_size = group.size()
        
        # Store pipeline metadata
        self._pipeline_rank = rank
        self._pipeline_world_size = world_size
        self._pipeline_group = group
        
        # Determine which layers this rank owns
        if hasattr(self, 'layers'):
            num_layers = len(self.layers)
            layers_per_rank = num_layers // world_size
            remainder = num_layers % world_size
            
            # Distribute layers
            if rank < remainder:
                start_layer = rank * (layers_per_rank + 1)
                end_layer = start_layer + layers_per_rank + 1
            else:
                start_layer = rank * layers_per_rank + remainder
                end_layer = start_layer + layers_per_rank
            
            self._start_layer = start_layer
            self._end_layer = end_layer
            self._my_layers = list(range(start_layer, end_layer))
            
            logger.info(f"Rank {rank}: Will process layers {start_layer}-{end_layer-1} ({len(self._my_layers)} layers)")
            
            # Mark which layers are ours but keep the list intact
            # We'll handle this in the forward pass
            
            # Only keep embedding/norm layers on appropriate ranks
            if rank > 0 and hasattr(self, 'embed_tokens'):
                self.embed_tokens = None  # Only rank 0 needs embeddings
            if rank < world_size - 1 and hasattr(self, 'norm'):
                self.norm = None  # Only last rank needs final norm
    
    # Add the pipeline method
    model_instance.pipeline = lambda group: pipeline(model_instance, group)
    
    # Store original forward
    if not hasattr(model_instance, '_original_forward'):
        model_instance._original_forward = model_instance.__call__
    
    def pipeline_forward(inputs, mask=None, cache=None):
        """
        Pipeline parallel forward pass with activation passing.
        """
        if not hasattr(model_instance, '_pipeline_group'):
            # Fallback to original forward
            return model_instance._original_forward(inputs, mask, cache)
        
        rank = model_instance._pipeline_rank
        world_size = model_instance._pipeline_world_size
        group = model_instance._pipeline_group
        
        # Handle embeddings (only rank 0)
        if rank == 0:
            if hasattr(model_instance, 'embed_tokens') and model_instance.embed_tokens:
                h = model_instance.embed_tokens(inputs)
            else:
                h = inputs
        else:
            # Receive activations from previous rank
            h = receive_activations(rank - 1, group)
        
        # Process our layers
        if hasattr(model_instance, '_my_layers'):
            for layer_idx in model_instance._my_layers:
                # Apply layer with mask and cache if available
                if mask is not None or cache is not None:
                    h = model_instance.layers[layer_idx](h, mask=mask, cache=cache)
                else:
                    h = model_instance.layers[layer_idx](h)
        
        # Pass to next rank or apply final norm
        if rank < world_size - 1:
            # Send to next rank
            send_activations(h, rank + 1, group)
            # Return dummy output for non-final ranks
            return h
        else:
            # Final rank - apply norm if present
            if hasattr(model_instance, 'norm') and model_instance.norm:
                h = model_instance.norm(h)
            return h
    
    # Replace forward method
    model_instance.__call__ = pipeline_forward
    
    logger.info(f"Added TRUE pipeline parallelism to {type(model_instance).__name__}")
    return model_instance


def send_activations(tensor, target_rank, group):
    """Send activations to target rank."""
    # Flatten to 1D for sending
    shape = tensor.shape
    flat = tensor.reshape(-1)
    
    # Send shape first
    shape_array = mx.array(shape, dtype=mx.int32)
    mx.distributed.send(shape_array, dst=target_rank, group=group)
    
    # Send data
    mx.distributed.send(flat, dst=target_rank, group=group)
    mx.eval(flat)  # Ensure send completes
    
    logger.debug(f"Sent activations shape {shape} to rank {target_rank}")


def receive_activations(source_rank, group):
    """Receive activations from source rank."""
    # Receive shape first
    shape_array = mx.zeros((3,), dtype=mx.int32)  # Max 3D shape
    mx.distributed.recv(shape_array, src=source_rank, group=group)
    mx.eval(shape_array)
    
    # Extract actual shape (filter out zeros)
    shape = tuple(int(s) for s in shape_array if s > 0)
    
    # Receive data
    num_elements = 1
    for s in shape:
        num_elements *= s
    
    flat = mx.zeros((num_elements,), dtype=mx.float32)
    mx.distributed.recv(flat, src=source_rank, group=group)
    mx.eval(flat)
    
    # Reshape
    tensor = flat.reshape(shape)
    logger.debug(f"Received activations shape {shape} from rank {source_rank}")
    
    return tensor


class PipelineParallelModel(nn.Module):
    """
    Wrapper for models to add pipeline parallelism.
    """
    def __init__(self, base_model, group=None):
        super().__init__()
        self.base_model = base_model
        
        # Add pipeline parallelism
        add_true_pipeline_parallelism(base_model.model if hasattr(base_model, 'model') else base_model)
        
        # Initialize pipeline
        if group:
            if hasattr(base_model, 'model'):
                base_model.model.pipeline(group)
            else:
                base_model.pipeline(group)
    
    def __call__(self, inputs, mask=None, cache=None):
        """Forward pass through pipeline parallel model."""
        if hasattr(self.base_model, 'model'):
            return self.base_model(inputs, mask=mask, cache=cache)
        else:
            return self.base_model(inputs, mask=mask, cache=cache)
    
    def generate(self, *args, **kwargs):
        """Generate method for compatibility."""
        if hasattr(self.base_model, 'generate'):
            return self.base_model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("Model doesn't have generate method")
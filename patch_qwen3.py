#!/usr/bin/env python3
"""
Patch to add pipeline parallelism support to Qwen3 models
This modifies the loaded model in-place to add the pipeline method
"""

import types
import mlx.core as mx


def add_pipeline_to_qwen3(model):
    """
    Add pipeline parallelism support to a loaded Qwen3 model
    Following the exact pattern from DeepSeek-V3
    """
    
    # Add pipeline attributes
    model.model.pipeline_rank = 0
    model.model.pipeline_size = 1
    
    def pipeline(self, group):
        """
        Pipeline implementation from DeepSeek-V3
        Split layers in reverse so rank=0 gets the last layers
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        
        if self.pipeline_rank < extra:
            layers_per_rank += 1
            
        # Calculate indices (reverse order like DeepSeek)
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        
        # Slice layers and replace unused with None
        self.layers = self.layers[:self.end_idx]
        self.layers[:self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx
        
        print(f"Pipeline rank {self.pipeline_rank}: layers {self.start_idx}-{self.end_idx-1}")
    
    # Store original forward
    original_forward = model.model.__class__.__call__
    
    def pipelined_forward(self, inputs, mask=None, cache=None):
        """
        Forward pass with pipeline communication (from DeepSeek-V3)
        """
        pipeline_rank = getattr(self, 'pipeline_rank', 0)
        pipeline_size = getattr(self, 'pipeline_size', 1)
        
        if pipeline_size == 1:
            # Single GPU - use original forward
            return original_forward(self, inputs, mask, cache)
        
        # Embeddings
        h = self.embed_tokens(inputs)
        
        if mask is None:
            from mlx_lm.models.qwen3 import create_attention_mask
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * self.num_layers
        
        # Receive from previous rank
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))
        
        # Process our layers
        for i in range(self.num_layers):
            layer = self.layers[self.start_idx + i]
            if layer is not None:
                h = layer(h, mask, cache[i])
        
        # Send to next rank
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
        
        # Broadcast while keeping in graph
        h = mx.distributed.all_gather(h)[:h.shape[0]]
        
        return self.norm(h)
    
    # Bind methods
    model.model.pipeline = types.MethodType(pipeline, model.model)
    # Replace the __call__ method on the class to ensure it's used
    model.model.__class__.__call__ = pipelined_forward
    
    return model


if __name__ == "__main__":
    from mlx_lm import load
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with mpirun -n 2")
        exit(1)
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}: Loading Qwen3...")
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Add pipeline support
    model = add_pipeline_to_qwen3(model)
    
    # Apply pipeline parallelism
    model.model.pipeline(group)
    
    # Evaluate to load weights
    mx.eval(model.parameters())
    
    # Synchronize
    sync = mx.distributed.all_sum(mx.array([1.0]))
    mx.eval(sync)
    
    print(f"Rank {rank}: Ready with pipeline support!")
    
    if rank == 0:
        print("\n✅ Successfully added pipeline support to Qwen3!")
        print(f"   Rank 0: layers {model.model.start_idx}-{model.model.end_idx-1}")
        print(f"   Rank 1: layers 0-{model.model.start_idx-1}")
        print("\nNow you can use this model with stream_generate!")
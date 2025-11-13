#!/usr/bin/env python3
"""
Add pipeline parallelism support to Qwen3 model
Based on DeepSeek implementation in MLX
"""

import mlx.core as mx
from typing import Optional, Any

def add_pipeline_to_qwen3(model):
    """Add pipeline method to Qwen3Model"""
    
    # Add pipeline attributes
    model.model.pipeline_rank = 0
    model.model.pipeline_size = 1
    
    def pipeline(self, group):
        """
        Pipeline parallelism implementation for Qwen3.
        Split layers in reverse so rank=0 gets the last layers and
        rank=pipeline_size-1 gets the first layers.
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        
        if self.pipeline_rank < extra:
            layers_per_rank += 1

        # Calculate which layers this rank should handle
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.num_layers = layers_per_rank
        
        # Keep track of which layers to process, don't modify the layers list
        # DeepSeek approach: use start_idx and num_layers instead of None layers
        self.num_layers = layers_per_rank
        
        print(f"Rank {self.pipeline_rank}: handling layers {self.start_idx} to {self.end_idx-1}")
    
    # Override the __call__ method to handle pipelined layers
    original_call = model.model.__call__
    
    def pipelined_call(self, inputs: mx.array, mask=None, cache=None):
        # DEBUG: Log when this method is called
        import os
        rank = getattr(self, 'pipeline_rank', -1)
        print(f"🔥 RANK {rank} (PID {os.getpid()}): pipelined_call() executing with input shape {inputs.shape}")
        
        # Import at top to avoid conflicts
        import mlx.core as mx
        
        h = self.embed_tokens(inputs)
        
        if mask is None and inputs.shape[1] > 1:
            # Import create_attention_mask from the same module
            from mlx_lm.models.qwen3 import create_attention_mask
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)
        
        # Pipeline processing: only process assigned layers
        if hasattr(self, 'pipeline_size') and self.pipeline_size > 1:
            # Handle distributed pipeline using CPU for communication
            try:
                import mlx.core as mx
                mx_dist = mx.distributed
            except (ImportError, AttributeError):
                # Fallback if distributed not available
                print(f"⚠️ RANK {rank}: mlx.distributed not available, falling back to single GPU")
                for layer, c in zip(self.layers, cache):
                    h = layer(h, mask, c)
                return self.norm(h)
            
            # Process layers in pipeline order
            # Rank 1 (mini2) processes first layers (0-13), then sends to rank 0
            # Rank 0 (mini1) receives from rank 1, processes last layers (14-27)
            
            if self.pipeline_rank == 1:  # mini2 - first stage
                print(f"🚀 RANK 1: Processing layers {self.start_idx}-{self.start_idx+self.num_layers-1}")
                # Process first layers
                for i in range(self.num_layers):
                    layer_idx = self.start_idx + i
                    h = self.layers[layer_idx](h, mask, cache[layer_idx])
                print(f"📤 RANK 1: Sending result shape {h.shape} to rank 0")
                # Send result to rank 0
                try:
                    h = mx_dist.send(h, 0)
                    mx.eval(h)  # Ensure completion
                except Exception as e:
                    print(f"❌ RANK 1: Send failed: {e}")
                    return h  # Fallback to local processing
                
            elif self.pipeline_rank == 0:  # mini1 - second stage  
                print(f"📥 RANK 0: Receiving from rank 1...")
                # Receive intermediate result from rank 1
                try:
                    h = mx_dist.recv_like(h, 1)
                    mx.eval(h)  # Ensure completion
                except Exception as e:
                    print(f"❌ RANK 0: Receive failed: {e}")
                    # Continue with current h for local processing
                print(f"🚀 RANK 0: Processing layers {self.start_idx}-{self.start_idx+self.num_layers-1}")
                # Process final layers
                for i in range(self.num_layers):
                    layer_idx = self.start_idx + i
                    h = self.layers[layer_idx](h, mask, cache[layer_idx])
                print(f"✅ RANK 0: Final result shape {h.shape}")
                # No need to send - this is the final result
        else:
            # Original processing for single GPU
            for layer, c in zip(self.layers, cache):
                h = layer(h, mask, c)
        
        return self.norm(h)
    
    # Bind both methods to the model - more aggressive approach
    import types
    
    # Store original for fallback
    model.model._original_call = model.model.__call__
    
    # Replace the method directly on the class to ensure all calls go through it
    model.model.__class__.__call__ = pipelined_call
    model.model.pipeline = types.MethodType(pipeline, model.model)
    
    print(f"✅ PATCHED: {model.model.__class__.__name__}.__call__ method replaced")
    
    return model

if __name__ == "__main__":
    from mlx_lm import load
    model, tokenizer = load('mlx-community/Qwen3-1.7B-8bit')
    
    print("Before:", hasattr(model.model, 'pipeline'))
    model = add_pipeline_to_qwen3(model)
    print("After:", hasattr(model.model, 'pipeline'))
    print("Layers:", len(model.model.layers))
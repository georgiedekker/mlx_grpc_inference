#!/usr/bin/env python3
"""
Fixed pipeline implementation for Qwen3 that ensures both ranks participate
"""

import mlx.core as mx
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

def add_working_pipeline(model):
    """Add working pipeline method to Qwen3Model that coordinates both ranks"""
    
    # Add pipeline attributes
    model.model.pipeline_rank = 0
    model.model.pipeline_size = 1
    
    def pipeline(self, group):
        """Pipeline parallelism implementation for Qwen3"""
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        
        layers_per_rank = len(self.layers) // self.pipeline_size
        
        # Reverse order: rank 0 gets last layers, rank N-1 gets first
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        
        logger.info(f"Rank {self.pipeline_rank}: Assigned layers {self.start_idx} to {self.end_idx-1}")
    
    # Override __call__ to use coordinated pipeline execution
    def pipelined_call(self, inputs: mx.array, mask=None, cache=None):
        import os
        rank = getattr(self, 'pipeline_rank', 0)
        size = getattr(self, 'pipeline_size', 1)
        
        # Embeddings
        h = self.embed_tokens(inputs)
        
        if mask is None and inputs.shape[1] > 1:
            from mlx_lm.models.qwen3 import create_attention_mask
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        # Pipeline processing
        if size > 1:
            # Both ranks must participate
            if rank == self.pipeline_size - 1:  # First pipeline stage (rank 1 for 2 GPUs)
                logger.info(f"Rank {rank}: Processing first layers {self.start_idx}-{self.end_idx-1}")
                
                # Process first layers
                for i in range(self.start_idx, self.end_idx):
                    h = self.layers[i](h, mask, cache[i])
                
                # Send to next stage
                logger.info(f"Rank {rank}: Sending intermediate result to rank {rank-1}")
                h = mx.distributed.send(h, dst=rank-1)
                mx.eval(h)
                
                # Return dummy for this rank
                return mx.zeros_like(h)
                
            elif rank == 0:  # Last pipeline stage
                logger.info(f"Rank {rank}: Waiting for intermediate from rank {rank+1}")
                
                # Receive from previous stage
                h = mx.distributed.recv_like(h, src=rank+1)
                mx.eval(h)
                
                logger.info(f"Rank {rank}: Processing final layers {self.start_idx}-{self.end_idx-1}")
                
                # Process final layers
                for i in range(self.start_idx, self.end_idx):
                    h = self.layers[i](h, mask, cache[i])
                
                # Apply final norm
                return self.norm(h)
                
            else:  # Middle stages (if > 2 GPUs)
                # Receive from previous
                h = mx.distributed.recv_like(h, src=rank+1)
                mx.eval(h)
                
                # Process layers
                for i in range(self.start_idx, self.end_idx):
                    h = self.layers[i](h, mask, cache[i])
                
                # Send to next
                h = mx.distributed.send(h, dst=rank-1)
                mx.eval(h)
                
                return mx.zeros_like(h)
        else:
            # Single GPU - process all layers
            for layer, c in zip(self.layers, cache):
                h = layer(h, mask, c)
            return self.norm(h)
    
    # Bind methods
    import types
    model.model.__class__.__call__ = pipelined_call
    model.model.pipeline = types.MethodType(pipeline, model.model)
    
    logger.info(f"✅ Added working pipeline to {model.model.__class__.__name__}")
    
    return model
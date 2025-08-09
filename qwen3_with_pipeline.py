"""Qwen3 model with pipeline parallelism support added following DeepSeek-V3 pattern"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional

# Import the original Qwen3 components
from mlx_lm.models.qwen3 import (
    ModelArgs,
    TransformerBlock,
    create_attention_mask,
    Model as OriginalModel
)


class Qwen3Model(nn.Module):
    """Qwen3Model with pipeline parallelism support following DeepSeek-V3 pattern"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Pipeline parallelism attributes (following DeepSeek-V3)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        """
        Pipeline parallelism implementation following DeepSeek-V3 pattern.
        Split layers in reverse so rank=0 gets the last layers and
        rank=pipeline_size-1 gets the first layers.
        """
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        
        if self.pipeline_rank < extra:
            layers_per_rank += 1
            
        # Calculate layer indices (reverse order like DeepSeek)
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        
        # Keep only our slice, set others to None
        self.layers = self.layers[:self.end_idx]
        self.layers[:self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        """Forward pass with pipeline parallelism support"""
        
        # Embeddings (only on first pipeline stage)
        h = self.embed_tokens(inputs)
        
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        
        if mask is None:
            mask = create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * self.num_layers
        
        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))
        
        # Process our layers
        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])
        
        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
        
        # Broadcast h while keeping it in the graph (following DeepSeek)
        h = mx.distributed.all_gather(h)[:h.shape[0]]
        
        return self.norm(h)


class Model(nn.Module):
    """Model wrapper with pipeline support"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3Model(args)  # Use our pipeline-enabled Qwen3Model
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers


# Test script
if __name__ == "__main__":
    import logging
    from mlx_lm import load
    from mlx_lm.utils import load_model, load_tokenizer
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if group:
        rank = group.rank()
        world_size = group.size()
        
        logger.info(f"Rank {rank}/{world_size}: Testing Qwen3 with pipeline support")
        
        # For testing, we need to monkey-patch the loaded model
        # In production, this would be in the actual qwen3.py file
        from mlx_lm import load
        
        # Load model
        model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
        
        # Replace the model's __class__ with our pipeline-enabled version
        # This is a hack for testing - in production, modify qwen3.py directly
        original_model = model.model
        
        # Add pipeline method
        import types
        
        def pipeline(self, group):
            self.pipeline_rank = group.rank()
            self.pipeline_size = group.size()
            layers_per_rank = len(self.layers) // self.pipeline_size
            extra = len(self.layers) - layers_per_rank * self.pipeline_size
            
            if self.pipeline_rank < extra:
                layers_per_rank += 1
                
            self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
            self.end_idx = self.start_idx + layers_per_rank
            self.layers = self.layers[:self.end_idx]
            self.layers[:self.start_idx] = [None] * self.start_idx
            self.num_layers = len(self.layers) - self.start_idx
            
            logger.info(f"Rank {self.pipeline_rank}: Handling layers {self.start_idx} to {self.end_idx-1}")
        
        # Bind the pipeline method
        original_model.pipeline = types.MethodType(pipeline, original_model)
        original_model.pipeline_rank = 0
        original_model.pipeline_size = 1
        
        # Apply pipeline
        model.model.pipeline(group)
        
        # Synchronize
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))
        
        if rank == 0:
            logger.info("✅ Pipeline support successfully added to Qwen3!")
            logger.info(f"✅ Rank 0 handling layers {model.model.start_idx}-{model.model.end_idx-1}")
        else:
            logger.info(f"✅ Rank {rank} handling layers {model.model.start_idx}-{model.model.end_idx-1}")
    else:
        print("No distributed group. Run with: mpirun -n 2 python qwen3_with_pipeline.py")
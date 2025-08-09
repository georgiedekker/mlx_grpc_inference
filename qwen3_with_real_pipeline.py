#!/usr/bin/env python3
"""
Qwen3 model with pipeline parallelism support added exactly like DeepSeek-V3
This should be integrated into mlx_lm/models/qwen3.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.qwen3 import (
    ModelArgs,
    TransformerBlock,
    create_attention_mask,
)


class Qwen3Model(nn.Module):
    """Qwen3Model with pipeline parallelism support (following DeepSeek-V3)"""
    
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
        
        # Pipeline parallelism attributes (from DeepSeek-V3)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        """
        Pipeline parallelism implementation (exact copy from DeepSeek-V3).
        Split layers in reverse so rank=0 gets the last layers and
        rank=pipeline_size-1 gets the first
        """
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

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        """Forward pass with pipeline parallelism (adapted from DeepSeek-V3)"""
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
        
        # Broadcast h while keeping it in the graph
        h = mx.distributed.all_gather(h)[:h.shape[0]]
        
        return self.norm(h)


class Model(nn.Module):
    """Top-level model wrapper"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3Model(args)
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

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads


def patch_existing_qwen3_model(model):
    """
    Patch an already-loaded Qwen3 model to add pipeline support.
    This monkey-patches the model instance to add the pipeline method.
    """
    import types
    
    # Add pipeline attributes
    model.model.pipeline_rank = 0
    model.model.pipeline_size = 1
    model.model.num_layers = len(model.model.layers)
    model.model.start_idx = 0
    model.model.end_idx = len(model.model.layers)
    
    def pipeline(self, group):
        """Pipeline implementation from DeepSeek-V3"""
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
        print(f"Rank {self.pipeline_rank}: layers {self.start_idx}-{self.end_idx-1}")
    
    def pipelined_forward(self, inputs, mask=None, cache=None):
        """Forward pass with pipeline (from DeepSeek-V3)"""
        h = self.embed_tokens(inputs)
        
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        
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
            h = self.layers[self.start_idx + i](h, mask, cache[i])
        
        # Send to next rank
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
        
        # Broadcast while keeping in graph
        h = mx.distributed.all_gather(h)[:h.shape[0]]
        
        return self.norm(h)
    
    # Bind methods
    model.model.pipeline = types.MethodType(pipeline, model.model)
    
    # Replace __call__ on the class
    original_class = model.model.__class__
    
    # Create a new class that inherits from the original
    class PipelinedQwen3Model(original_class):
        def __call__(self, inputs, mask=None, cache=None):
            return pipelined_forward(self, inputs, mask, cache)
    
    # Change the instance's class
    model.model.__class__ = PipelinedQwen3Model
    
    return model


if __name__ == "__main__":
    import sys
    from mlx_lm import load, stream_generate
    
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with: mpirun -n 2 python qwen3_with_real_pipeline.py")
        sys.exit(1)
    
    rank = group.rank()
    world_size = group.size()
    
    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    rprint(f"Loading Qwen3 with {world_size} GPUs...")
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Patch the model to add pipeline support
    model = patch_existing_qwen3_model(model)
    
    # Apply pipeline
    model.model.pipeline(group)
    
    # Evaluate parameters
    mx.eval(model.parameters())
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    rprint("✅ Model ready with pipeline support!")
    
    # Test generation
    prompt = "What is 2+2? The answer is"
    
    rprint(f"\nPrompt: {prompt}")
    rprint("Response: ", end="", flush=True)
    
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=20
    ):
        rprint(response.text, end="", flush=True)
    
    rprint()
    
    if rank == 0:
        rprint(f"\n{'='*50}")
        rprint(f"Generation: {response.generation_tokens} tokens")
        rprint(f"Speed: {response.generation_tps:.1f} tokens/sec")
        rprint(f"Memory: {response.peak_memory:.2f} GB")
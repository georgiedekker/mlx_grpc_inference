#!/usr/bin/env python
"""
MLX Model Parallel implementation for distributed inference.

Based on MLX documentation for distributed model parallelism.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import generate_step
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import time


@dataclass
class ModelArgs:
    """Model configuration arguments."""
    model_name: str = "mlx-community/Qwen3-1.7B-8bit"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 5504
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    num_key_value_heads: int = 16
    rope_theta: float = 10000.0
    model_type: str = "qwen2"


class ShardedTransformer(nn.Module):
    """Sharded transformer for distributed inference."""
    
    def __init__(self, args: ModelArgs, rank: int, world_size: int):
        super().__init__()
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # Load full model first
        self.full_model, self.tokenizer = load(args.model_name)
        
        # Determine layer distribution
        layers_per_rank = args.num_hidden_layers // world_size
        extra_layers = args.num_hidden_layers % world_size
        
        if rank < extra_layers:
            self.start_layer = rank * (layers_per_rank + 1)
            self.end_layer = self.start_layer + layers_per_rank + 1
        else:
            self.start_layer = rank * layers_per_rank + extra_layers
            self.end_layer = self.start_layer + layers_per_rank
        
        print(f"Rank {rank}: Handling layers {self.start_layer} to {self.end_layer-1}")
        
        # Keep only necessary components
        if rank == 0:
            self.embed_tokens = self.full_model.model.embed_tokens
        else:
            self.embed_tokens = None
            
        if rank == world_size - 1:
            self.norm = self.full_model.model.norm
            # Handle lm_head - check if it exists or if embeddings are tied  
            self.lm_head = getattr(self.full_model, 'lm_head', None)
            self.use_tied_embeddings = (hasattr(self.full_model, 'args') and 
                                       getattr(self.full_model.args, 'tie_word_embeddings', False))
        else:
            self.norm = None
            self.lm_head = None
            self.use_tied_embeddings = False
        
        # Keep only our layers
        self.layers = self.full_model.model.layers[self.start_layer:self.end_layer]
    
    def __call__(self, inputs: mx.array, cache=None):
        """Forward pass through sharded layers."""
        h = inputs
        
        # Process our layers
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache else None)
        
        return h


def distributed_generate(
    model: ShardedTransformer,
    prompt: mx.array,
    max_tokens: int = 100,
    temp: float = 0.7,
    rank: int = 0,
    world_size: int = 1,
    comm=None
) -> List[int]:
    """Generate tokens using distributed model."""
    
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))
    
    # Only rank 0 handles tokenization
    if rank == 0:
        y = prompt
        tokens = []
    
    cache = []
    
    # Generate tokens
    for _ in range(max_tokens):
        # Forward pass through distributed model
        if rank == 0:
            # Embeddings
            x = model.embed_tokens(y)
        else:
            # Receive from previous rank
            x = comm.recv(source=rank-1)
        
        # Process through our layers
        x = model(x, cache)
        
        # Send to next rank or compute output
        if rank < world_size - 1:
            comm.send(x, dest=rank+1)
            # Receive final logits from last rank
            logits = comm.recv(source=world_size-1)
        else:
            # Final processing
            x = model.norm(x)
            if model.lm_head is not None:
                logits = model.lm_head(x)
            elif model.use_tied_embeddings and model.embed_tokens is not None:
                # Use embedding weights for output projection
                logits = model.embed_tokens.as_linear(x)
            else:
                raise RuntimeError("No output projection method available")
            # Send back to rank 0
            if rank != 0:
                comm.send(logits, dest=0)
        
        # Rank 0 does sampling
        if rank == 0:
            # Sample next token
            y = sample(logits[:, -1, :])
            tokens.append(y.item())
            
            # Check for EOS
            if y.item() == model.tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            y = mx.concatenate([prompt, mx.array(tokens)])
    
    return tokens if rank == 0 else []


def main():
    """Test distributed model parallel generation."""
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    print(f"Initializing distributed model on rank {rank}/{world_size}")
    
    # Create model arguments
    args = ModelArgs()
    
    # Create sharded model
    model = ShardedTransformer(args, rank, world_size)
    
    if rank == 0:
        # Tokenize prompt
        prompt_text = "Hello, how are you?"
        prompt_tokens = model.tokenizer.encode(prompt_text)
        prompt = mx.array(prompt_tokens)
        print(f"Prompt: {prompt_text}")
    else:
        prompt = None
    
    # Generate
    start_time = time.time()
    tokens = distributed_generate(
        model, 
        prompt, 
        max_tokens=50,
        temp=0.7,
        rank=rank,
        world_size=world_size,
        comm=comm
    )
    
    if rank == 0:
        elapsed = time.time() - start_time
        response = model.tokenizer.decode(tokens)
        print(f"\nGenerated: {response}")
        print(f"Time: {elapsed:.2f}s")
    
    # Synchronize before exit
    comm.barrier()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simplified pipeline parallelism test.
Just does a single forward pass to prove activation passing works.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, load_tokenizer
from pathlib import Path
from huggingface_hub import snapshot_download
import os

mx.set_default_device(mx.gpu)

def test_pipeline():
    # Initialize distributed
    group = mx.distributed.init()
    if not group:
        print("Failed to initialize distributed")
        return
    
    rank = group.rank()
    world_size = group.size()
    hostname = os.uname().nodename
    
    print(f"Rank {rank}/{world_size} on {hostname}")
    
    # Load a simple model
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    print(f"Rank {rank}: Loading model...")
    
    model_path = Path(snapshot_download(
        model_name,
        allow_patterns=["*.json", "*.py", "tokenizer.model", "*.safetensors", "*.tiktoken", "*.txt"]
    ))
    
    model, config = load_model(model_path)
    tokenizer = load_tokenizer(model_path, {"trust_remote_code": True})
    
    # Get the actual model
    actual_model = model.model
    
    # Split layers manually
    num_layers = len(actual_model.layers)
    layers_per_rank = num_layers // world_size
    
    if rank == 0:
        # Rank 0: embeddings + first half of layers
        start, end = 0, layers_per_rank
        print(f"Rank 0: Processing layers {start}-{end-1}")
        my_layers = actual_model.layers[start:end]
        
        # Create input
        input_ids = mx.array([[1, 2, 3, 4, 5]])  # 5 tokens
        
        # Embedding
        h = actual_model.embed_tokens(input_ids)
        print(f"Rank 0: Embedded input, shape {h.shape}")
        
        # Process layers
        for i, layer in enumerate(my_layers):
            h = layer(h)
            if i % 5 == 0:
                print(f"Rank 0: Processed layer {start + i}")
        
        print(f"Rank 0: Sending activations to rank 1, shape {h.shape}")
        
        # Send to rank 1
        mx.distributed.send(h, dst=1)
        mx.eval(h)
        
        # Receive final output from rank 1
        output = mx.distributed.recv_like(h, src=1)
        mx.eval(output)
        
        print(f"Rank 0: Received final output from rank 1, shape {output.shape}")
        
        # Check GPU memory
        memory = mx.get_active_memory() / (1024**3)
        print(f"✅ Rank 0 on {hostname}: GPU memory = {memory:.2f} GB")
        
    else:
        # Rank 1: second half of layers + norm
        start, end = layers_per_rank, num_layers
        print(f"Rank 1: Processing layers {start}-{end-1}")
        my_layers = actual_model.layers[start:end]
        
        # Receive from rank 0
        # Create dummy tensor with expected shape
        dummy = mx.zeros((1, 5, 2048))  # (batch, seq_len, hidden_size)
        h = mx.distributed.recv_like(dummy, src=0)
        mx.eval(h)
        
        print(f"Rank 1: Received activations from rank 0, shape {h.shape}")
        
        # Process layers
        for i, layer in enumerate(my_layers):
            h = layer(h)
            if i % 5 == 0:
                print(f"Rank 1: Processed layer {start + i}")
        
        # Apply final norm
        h = actual_model.norm(h)
        print(f"Rank 1: Applied final norm")
        
        # Send back to rank 0
        mx.distributed.send(h, dst=0)
        mx.eval(h)
        
        print(f"Rank 1: Sent final output to rank 0")
        
        # Check GPU memory
        memory = mx.get_active_memory() / (1024**3)
        print(f"✅ Rank 1 on {hostname}: GPU memory = {memory:.2f} GB")
    
    print(f"Rank {rank} on {hostname}: Pipeline test complete!")


if __name__ == "__main__":
    test_pipeline()
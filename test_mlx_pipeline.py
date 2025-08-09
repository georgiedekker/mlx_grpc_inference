#!/usr/bin/env python3
"""
Test if MLX models support pipeline parallelism
"""
import mlx.core as mx
from mlx_lm import load

# Initialize distributed
group = mx.distributed.init()

if group:
    rank = group.rank()
    world_size = group.size()
    print(f"Rank {rank}/{world_size}")
    
    # Try to load a model and check for pipeline method
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Check if model has pipeline method
    if hasattr(model, 'pipeline'):
        print(f"Rank {rank}: Model has pipeline() method")
        try:
            pipelined_model = model.pipeline(group)
            print(f"Rank {rank}: Pipeline setup successful!")
        except Exception as e:
            print(f"Rank {rank}: Pipeline setup failed: {e}")
    else:
        print(f"Rank {rank}: Model does NOT have pipeline() method")
        print(f"Rank {rank}: Available methods: {[m for m in dir(model) if not m.startswith('_')][:10]}")
else:
    print("No distributed group - running single GPU test")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    print(f"Model type: {type(model)}")
    print(f"Has pipeline: {hasattr(model, 'pipeline')}")
    print(f"Methods: {[m for m in dir(model) if not m.startswith('_')][:20]}")
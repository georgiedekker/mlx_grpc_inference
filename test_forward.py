#!/usr/bin/env python3
"""
Test just the forward pass with pipeline
"""

import mlx.core as mx
from mlx_lm import load
from patch_qwen3 import add_pipeline_to_qwen3


def main():
    # Initialize distributed
    group = mx.distributed.init()
    
    if not group:
        print("No distributed group. Run with: mpirun -n 2")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}: Starting test")
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Add pipeline support
    model = add_pipeline_to_qwen3(model)
    
    # Apply pipeline
    model.model.pipeline(group)
    
    # Evaluate parameters
    mx.eval(model.parameters())
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    print(f"Rank {rank}: Model ready, testing forward pass...")
    
    # Create test input
    test_input = mx.array([[1, 2, 3, 4, 5]])  # Simple token sequence
    
    # Test forward pass
    try:
        output = model.model(test_input)
        mx.eval(output)
        print(f"Rank {rank}: Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Rank {rank}: Forward pass failed: {e}")
    
    # Final sync
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    if rank == 0:
        print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
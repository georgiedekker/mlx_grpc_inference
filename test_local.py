#!/usr/bin/env python3
"""
Quick test script to verify distributed inference works locally
"""
import time
import mlx.core as mx
from server import shard_and_load, generate_pipeline_parallel

def test_local():
    """Test inference with 2 local processes."""
    print("Testing distributed inference locally...")
    
    # Load model
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    shard_and_load(model_name)
    
    # Test generation
    print("\nTesting generation...")
    start = time.time()
    
    result = generate_pipeline_parallel(
        prompt="What is 2+2? Answer:",
        max_tokens=20,
        temperature=0.7
    )
    
    elapsed = time.time() - start
    
    if result:
        print(f"\nGenerated text: {result.get('text', 'No text')}")
        print(f"Tokens per second: {result.get('tokens_per_second', 0):.2f}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"GPUs used: {result.get('gpus_used', 1)}")
    else:
        print("Worker rank - no result")

if __name__ == "__main__":
    test_local()
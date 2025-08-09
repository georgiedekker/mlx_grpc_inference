#!/usr/bin/env python3
"""
Test distributed inference speed to diagnose the slowness issue
"""
import time
import mlx.core as mx
import mlx.nn as nn

def test_allreduce_speed():
    """Test the speed of allreduce operations."""
    
    # Initialize distributed
    group = mx.distributed.init()
    if not group:
        print("No distributed group, exiting")
        return
    
    rank = group.rank()
    world_size = group.size()
    
    print(f"Rank {rank}/{world_size}")
    
    # Test different tensor sizes
    sizes = [(1, 512, 2048), (1, 1024, 2048), (1, 2048, 2048)]
    
    for shape in sizes:
        tensor = mx.random.normal(shape)
        
        # Warm up
        for _ in range(3):
            result = mx.distributed.all_sum(tensor)
            mx.eval(result)
        
        # Time allreduce
        iterations = 10
        start = time.time()
        
        for _ in range(iterations):
            result = mx.distributed.all_sum(tensor)
            mx.eval(result)
        
        elapsed = time.time() - start
        avg_time = (elapsed / iterations) * 1000  # ms
        
        if rank == 0:
            print(f"Shape {shape}: {avg_time:.2f}ms per allreduce")
            print(f"  That's {1000/avg_time:.1f} allreduces/second")
    
    # Test sequential allreduce (simulating token generation)
    if rank == 0:
        print("\nSimulating token generation (100 sequential allreduces):")
    
    shape = (1, 1, 2048)  # Single token shape
    start = time.time()
    
    for i in range(100):
        tensor = mx.random.normal(shape)
        result = mx.distributed.all_sum(tensor)
        mx.eval(result)
    
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"100 tokens: {elapsed:.2f}s total")
        print(f"Speed: {100/elapsed:.1f} tokens/second")
        print(f"Per token: {elapsed/100*1000:.1f}ms")

if __name__ == "__main__":
    test_allreduce_speed()
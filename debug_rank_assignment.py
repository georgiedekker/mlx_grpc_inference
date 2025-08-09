#!/usr/bin/env python3
"""
Debug MLX rank assignment issue
"""
import os
import socket
import mlx.core as mx

print("="*60)
print("MLX RANK ASSIGNMENT DEBUG")
print("="*60)
print(f"Hostname: {socket.gethostname()}")
print(f"PID: {os.getpid()}")

# Check environment variables
print("\nEnvironment Variables:")
env_vars = [
    'OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK',
    'MLX_RANK', 'MLX_WORLD_SIZE', 'MLX_HOSTFILE', 'MLX_RING_VERBOSE',
    'PMIX_RANK', 'PMI_RANK', 'PMI_SIZE'
]
for var in env_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"  {var}: {value}")

# Check if distributed is available
print(f"\nmx.distributed.is_available(): {mx.distributed.is_available()}")

# Try different backends
backends = ['any', 'ring', 'mpi']
for backend in backends:
    print(f"\n--- Testing backend: {backend} ---")
    try:
        group = mx.distributed.init(strict=False, backend=backend)
        if group:
            rank = group.rank()
            size = group.size()
            print(f"✅ Success: rank={rank}/{size}")
            
            # Test basic communication if size > 1
            if size > 1:
                test_array = mx.array([float(rank)])
                result = mx.distributed.all_sum(test_array, group=group)
                mx.eval(result)
                print(f"✅ Communication test: {result.item()}")
                
                # Expected sum for ranks 0,1,2,...,n-1 is n*(n-1)/2
                expected = sum(range(size))
                if abs(result.item() - expected) < 0.001:
                    print(f"🎉 Communication working correctly!")
                else:
                    print(f"❌ Expected {expected}, got {result.item()}")
        else:
            print(f"❌ Failed to initialize {backend} backend")
    except Exception as e:
        print(f"❌ Exception with {backend}: {e}")

print("="*60)
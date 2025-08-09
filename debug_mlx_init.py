#!/usr/bin/env python3
"""
Debug why MLX distributed isn't connecting the nodes
"""
import os
import socket

print(f"Host: {socket.gethostname()}")
print(f"OMPI_COMM_WORLD_SIZE: {os.environ.get('OMPI_COMM_WORLD_SIZE', 'NOT SET')}")
print(f"OMPI_COMM_WORLD_RANK: {os.environ.get('OMPI_COMM_WORLD_RANK', 'NOT SET')}")

# Don't import mpi4py first - let MLX initialize MPI
import mlx.core as mx

print("\nCalling mx.distributed.init()...")
group = mx.distributed.init()

if group:
    print(f"✅ MLX group created: rank={group.rank()}, size={group.size()}")
    
    # Try a simple all_sum to verify communication
    test_tensor = mx.array([float(group.rank())])
    print(f"Before all_sum: {test_tensor}")
    
    result = mx.distributed.all_sum(test_tensor, group=group)
    mx.eval(result)
    print(f"After all_sum: {result}")
    
    if group.size() == 2 and result.item() == 1.0:
        print("✅ DISTRIBUTED COMMUNICATION WORKING!")
    else:
        print(f"⚠️ Expected sum=1.0 with 2 ranks, got sum={result.item()} with {group.size()} ranks")
else:
    print("❌ No MLX group created")
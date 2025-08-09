#!/usr/bin/env python3
"""
Test MLX distributed with explicit MPI backend
"""
import os
import socket

hostname = socket.gethostname()
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))
size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', -1))

print(f"[{hostname}] MPI env: rank={rank}/{size}")

import mlx.core as mx

# Check if distributed is available
print(f"[{hostname}] mx.distributed.is_available() = {mx.distributed.is_available()}")

# Try to init with MPI backend explicitly
print(f"[{hostname}] Initializing with backend='mpi'...")
try:
    group = mx.distributed.init(strict=True, backend="mpi")
    print(f"✅ [{hostname}] MLX group: rank={group.rank()}/{group.size()}")
    
    # Test communication
    tensor = mx.array([float(group.rank())])
    result = mx.distributed.all_sum(tensor, group=group)
    mx.eval(result)
    print(f"✅ [{hostname}] all_sum result: {result.item()}")
    
    if group.size() == 2 and result.item() == 1.0:
        print(f"🎉 [{hostname}] DISTRIBUTED WORKING ACROSS NODES!")
    
except Exception as e:
    print(f"❌ [{hostname}] Failed: {e}")
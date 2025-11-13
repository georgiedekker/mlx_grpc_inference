#!/usr/bin/env python3
"""
Initialize MPI4py first, then MLX
"""

# Initialize MPI4py FIRST - this is critical
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import socket
hostname = socket.gethostname()

print(f"[{hostname}] MPI4py: rank={rank}/{size}")

# Now import and initialize MLX
import mlx.core as mx

print(f"[{hostname}] Calling mx.distributed.init()...")
group = mx.distributed.init()

if group:
    print(f"✅ [{hostname}] MLX group: rank={group.rank()}/{group.size()}")
    
    # Test communication
    tensor = mx.array([float(group.rank())])
    result = mx.distributed.all_sum(tensor, group=group)
    mx.eval(result)
    
    print(f"[{hostname}] all_sum result: {result.item()}")
    
    if group.size() == 2 and result.item() == 1.0:
        print(f"🎉 DISTRIBUTED WORKING! Rank {rank} on {hostname}")
        print(f"   Both Mac minis connected via MLX distributed!")
else:
    print(f"❌ [{hostname}] No group created")
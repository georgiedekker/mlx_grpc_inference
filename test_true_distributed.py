#!/usr/bin/env python3
"""
Simple test to verify both Mac minis are running
"""

import mlx.core as mx
import os
import socket

# Initialize distributed
group = mx.distributed.init()

if group:
    rank = group.rank()
    world_size = group.size()
    hostname = socket.gethostname()
    
    print(f"✅ Rank {rank}/{world_size} on {hostname}")
    
    # Create a tensor on each GPU
    tensor = mx.array([float(rank)], dtype=mx.float32)
    mx.eval(tensor)
    
    # All-gather to verify communication
    gathered = mx.distributed.all_gather(tensor, group=group)
    mx.eval(gathered)
    
    if rank == 0:
        print(f"Gathered from all ranks: {gathered}")
        print(f"✅ SUCCESS: {world_size} nodes active!")
        if world_size == 2:
            print("🚀 Both Mac minis are processing!")
        else:
            print(f"⚠️ Only {world_size} node(s) active, expected 2")
else:
    print("❌ No distributed group - running single node")
    print("Run with: mpirun -n 2 --hostfile mpi_hostfile")
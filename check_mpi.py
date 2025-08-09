#!/usr/bin/env python3
"""
Check MPI environment
"""
import os
import socket

hostname = socket.gethostname()
print(f"Host: {hostname}")
print(f"OMPI_COMM_WORLD_SIZE: {os.environ.get('OMPI_COMM_WORLD_SIZE', 'NOT SET')}")
print(f"OMPI_COMM_WORLD_RANK: {os.environ.get('OMPI_COMM_WORLD_RANK', 'NOT SET')}")
print(f"PMIX_RANK: {os.environ.get('PMIX_RANK', 'NOT SET')}")
print(f"PMI_SIZE: {os.environ.get('PMI_SIZE', 'NOT SET')}")
print(f"PMI_RANK: {os.environ.get('PMI_RANK', 'NOT SET')}")

# Try to init MLX distributed
try:
    import mlx.core as mx
    group = mx.distributed.init()
    if group:
        print(f"MLX: rank={group.rank()}, size={group.size()}")
    else:
        print("MLX: No group created")
except Exception as e:
    print(f"MLX Error: {e}")
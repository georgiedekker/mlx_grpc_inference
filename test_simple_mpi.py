#!/usr/bin/env python3
"""Simple MPI test"""
import mlx.core as mx
import os

hostname = os.uname().nodename
print(f"Starting on {hostname}")

group = mx.distributed.init()
if group:
    rank = group.rank()
    size = group.size()
    print(f"{hostname}: Rank {rank}/{size}")
    
    # Simple all_sum test
    data = mx.array([rank + 1.0])
    result = mx.distributed.all_sum(data, group=group)
    mx.eval(result)
    print(f"{hostname}: all_sum result = {result.item()} (expected {size * (size + 1) / 2})")
else:
    print(f"{hostname}: No distributed group")
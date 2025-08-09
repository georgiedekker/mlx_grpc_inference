#!/usr/bin/env python3
"""Test MLX distributed connectivity"""

import mlx.core as mx
import mlx.core.distributed as dist

# Initialize distributed
group = dist.init()
rank = group.rank()
size = group.size()

print(f"Rank {rank}/{size} initialized")

# Test communication
data = mx.array([float(rank)])
result = dist.all_sum(data, group=group)
mx.eval(result)

print(f"Rank {rank}: all_sum result = {result.item()}")

if size == 2:
    expected = 0.0 + 1.0  # sum of rank 0 and rank 1
    if result.item() == expected:
        print(f"✅ SUCCESS: Both machines connected! (rank {rank})")
    else:
        print(f"❌ FAIL: Expected {expected}, got {result.item()}")
else:
    print(f"⚠️  Running with {size} rank(s), expected 2 for distributed")
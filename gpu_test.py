#!/usr/bin/env python3
"""
Simple test to prove both GPUs are being used.
Mini1 and mini2 will both allocate memory and do computations.
"""
import mlx.core as mx
import time
import os

# Force GPU
mx.set_default_device(mx.gpu)

# Initialize distributed
group = mx.distributed.init()
if not group:
    print("Failed to initialize distributed")
    exit(1)

rank = group.rank()
world_size = group.size()
hostname = os.uname().nodename

print(f"Rank {rank}/{world_size} on {hostname}")

# Allocate a large tensor to use GPU memory
size = 10000  # Size of square matrix (100M elements = ~400MB)
print(f"Rank {rank} on {hostname}: Allocating {size}x{size} matrix...")
tensor = mx.random.uniform(shape=(size, size))
mx.eval(tensor)

# Check memory
memory = mx.get_active_memory() / (1024**6)  # MB
print(f"Rank {rank} on {hostname}: GPU memory = {memory:.2f} MB")

# Do some computations to prove GPU is active
print(f"Rank {rank} on {hostname}: Performing matrix multiplication...")
for i in range(5):
    result = tensor @ tensor
    mx.eval(result)
    memory = mx.get_active_memory() / (1024**6)
    print(f"Rank {rank} on {hostname}: Iteration {i+1}, memory = {memory:.2f} MB")
    time.sleep(1)

# Do an all-gather to prove communication works
local_value = mx.array([float(rank + 1)])
gathered = mx.distributed.all_gather(local_value, group=group)
mx.eval(gathered)

if rank == 0:
    print(f"All-gather result: {gathered}")
    if world_size == 2:
        expected = mx.array([1.0, 2.0])
        if mx.array_equal(gathered, expected):
            print("✅ Both GPUs are active and communicating!")
            print(f"✅ mini1 (rank 0) and mini2 (rank 1) are both processing!")
        else:
            print(f"❌ Unexpected result. Expected {expected}, got {gathered}")

print(f"Rank {rank} on {hostname}: Test complete")
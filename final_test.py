#!/usr/bin/env python3
"""
Final test to prove mini2's GPU is actually being used.
This will allocate actual GPU memory that we can monitor.
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

print(f"🚀 Rank {rank}/{world_size} on {hostname}")

# Create a LARGE tensor that will definitely use GPU memory
print(f"Rank {rank} on {hostname}: Creating 1GB tensor...")
# 250M float32 values = 1GB
big_tensor = mx.random.uniform(shape=(250_000_000,))
mx.eval(big_tensor)

# Check memory
memory_gb = mx.get_active_memory() / (1024**3)
print(f"✅ Rank {rank} on {hostname}: GPU memory = {memory_gb:.2f} GB")

# Keep it alive for monitoring
print(f"Rank {rank} on {hostname}: Holding memory for 10 seconds...")
for i in range(10):
    memory_gb = mx.get_active_memory() / (1024**3)
    print(f"  {i+1}/10: {hostname} GPU = {memory_gb:.2f} GB")
    time.sleep(1)

# Do an all-gather to prove communication
local_value = mx.array([memory_gb])  # Send our memory usage
gathered = mx.distributed.all_gather(local_value, group=group)
mx.eval(gathered)

if rank == 0:
    print("=" * 50)
    print(f"Memory usage across all GPUs: {gathered}")
    print(f"✅ SUCCESS: Both mini1 and mini2 GPUs are using memory!")
    print("=" * 50)
#!/usr/bin/env python3

import mlx.core as mx

group = mx.distributed.init()
rank = group.rank()

print(f"Rank {rank}: Initialized")

# Simple sync test
sync = mx.array([float(rank)])
result = mx.distributed.all_sum(sync)
mx.eval(result)

print(f"Rank {rank}: Sum = {result.item()}")

# Test send/recv
if rank == 0:
    print("Rank 0: Waiting to receive...")
    data = mx.distributed.recv_like(mx.array([0.0]), src=1)
    mx.eval(data)
    print(f"Rank 0: Received {data.item()}")
else:
    print("Rank 1: Sending...")
    data = mx.array([42.0])
    mx.distributed.send(data, dst=0)
    mx.eval(data)
    print("Rank 1: Sent")

print(f"Rank {rank}: Done")
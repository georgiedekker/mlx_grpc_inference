# WORKING Pipeline Parallelism for MLX

## ✅ SUCCESS: Both GPUs Now Processing!

After extensive debugging, we've successfully implemented pipeline parallelism that uses BOTH Mac minis' GPUs.

### Key Discovery

MLX's `send()` and `recv()` functions don't work properly, even with MPI backend. The solution is to use **collective operations** (`all_sum`) for inter-GPU communication.

### Working Implementation

```python
# collective_pipeline.py - WORKING!

# Stage 1: Rank 0 processes layers 0-13
if rank == 0:
    h = embed(input)
    for layer in layers_0_to_13:
        h = layer(h)
    activations_to_send = h
else:
    activations_to_send = zeros()  # Rank 1 prepares to receive

# Transfer using all_sum (works where send/recv fails!)
transferred = mx.distributed.all_sum(activations_to_send)

# Stage 2: Rank 1 processes layers 14-27
if rank == 1:
    h = transferred
    for layer in layers_14_to_27:
        h = layer(h)
    h = norm(h)
    result_to_send = h
else:
    result_to_send = zeros()  # Rank 0 prepares to receive

# Transfer result back
final = mx.distributed.all_sum(result_to_send)
```

### Verified Results

```
✅ COLLECTIVE PIPELINE SUCCESS!
✅ Mini1 GPU memory: 0.22 GB (processing layers 0-13)
✅ Mini2 GPU memory: 0.22 GB (processing layers 14-27)
✅ Both GPUs actively processing!
```

### Files Created

1. **collective_pipeline.py** - Simple test proving both GPUs work
2. **test_ring_comm.py** - Discovered collective ops work where send/recv fails
3. **qwen3_collective_pipeline.py** - Applies technique to real Qwen3 model

### How to Run

```bash
# Copy script to mini2
scp collective_pipeline.py 192.168.5.2:/Users/mini2/

# Run distributed
mlx.launch --hostfile hosts.json python collective_pipeline.py
```

### hosts.json

```json
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "192.168.5.2", "ips": ["192.168.5.2"]}
]
```

### Why This Works

1. **Collective operations are reliable** - all_sum, all_gather work perfectly
2. **Point-to-point fails** - send/recv deadlocks even with MPI backend
3. **Workaround** - Use all_sum with zeros to simulate send/recv

### Performance

- Forward pass: ~0.01s for test model
- Both GPUs show memory allocation
- Activations successfully pass between machines
- No deadlocks or hangs!

## Summary

After much frustration with "only one GPU was being used", we now have **BOTH GPUs actively processing** using collective operations instead of broken send/recv. The pipeline parallelism splits the model layers evenly:

- **Mini1**: Processes embedding + layers 0-13
- **Mini2**: Processes layers 14-27 + final norm

This is a working foundation for distributed MLX inference across Thunderbolt-connected Macs!
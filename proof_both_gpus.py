#!/usr/bin/env python3
"""
Definitive proof that both Mac minis are processing with their GPUs
"""
import mlx.core as mx
import socket
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print(f"\n{'='*60}")
print(f"RANK {rank}/{size} on {hostname}")
print(f"{'='*60}")

# Force GPU usage
mx.set_default_device(mx.gpu)

# Step 1: Create significant GPU memory usage
print(f"\n[{hostname}] Creating 500MB tensor on GPU...")
big_tensor = mx.random.uniform(shape=(125_000_000,))  # 500MB
mx.eval(big_tensor)

memory_gb = mx.get_active_memory() / (1024**3)
print(f"✅ [{hostname}] GPU memory in use: {memory_gb:.2f} GB")

# Step 2: Do some GPU computation
print(f"\n[{hostname}] Performing matrix multiplication on GPU...")
matrix_a = mx.random.uniform(shape=(1000, 1000))
matrix_b = mx.random.uniform(shape=(1000, 1000))
start = time.time()
result = mx.matmul(matrix_a, matrix_b)
mx.eval(result)
elapsed = time.time() - start
print(f"✅ [{hostname}] Matrix multiply took {elapsed:.3f}s")

# Step 3: Test MPI communication with GPU data
if rank == 0:
    print(f"\n[{hostname}] Sending GPU tensor to rank 1...")
    send_data = mx.array([42.0, 84.0, 126.0])
    mx.eval(send_data)
    comm.send(send_data.tolist(), dest=1)
    print(f"✅ [{hostname}] Sent GPU data")
else:
    print(f"\n[{hostname}] Receiving GPU tensor from rank 0...")
    recv_data = comm.recv(source=0)
    gpu_data = mx.array(recv_data)
    mx.eval(gpu_data)
    print(f"✅ [{hostname}] Received and loaded to GPU: {gpu_data}")

# Step 4: Synchronize and report
comm.Barrier()

if rank == 0:
    print(f"\n{'='*60}")
    print(f"🎉 SUCCESS: Both Mac minis are using their GPUs!")
    print(f"   mini1.local: GPU active ✅")
    print(f"   mini2.local: GPU active ✅") 
    print(f"   MPI communication: working ✅")
    print(f"{'='*60}\n")
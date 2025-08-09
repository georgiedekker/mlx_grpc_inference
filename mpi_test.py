import mlx.core as mx
from mlx_lm import load, generate
import socket

mx.set_default_device(mx.gpu)
hostname = socket.gethostname()

group = mx.distributed.init()
rank = group.rank() if group else 0
size = group.size() if group else 1

print(f"MPI Rank {rank}/{size} on {hostname}")

# Load model on all ranks
print(f"MPI Rank {rank}: Loading model on {hostname}...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Heavy GPU work to verify both GPUs are used
print(f"MPI Rank {rank}: GPU computation on {hostname}...")
for i in range(3):
    big_matrix = mx.random.normal(shape=(8000, 8000))
    result = big_matrix @ big_matrix.T
    mx.eval(result)
    print(f"MPI Rank {rank} on {hostname}: Iteration {i+1}/3")

# Rank 0 generates text
if rank == 0:
    print(f"MPI Rank 0: Generating on {hostname}...")
    response = generate(model, tokenizer, "def fibonacci(n):", max_tokens=50)
    print(f"Generated code: {response}")

print(f"MPI Rank {rank} on {hostname}: Complete")

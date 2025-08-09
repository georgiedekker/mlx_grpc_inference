import mlx.core as mx
from mlx_lm import load, generate
import time
import socket
import subprocess

# Force GPU usage
mx.set_default_device(mx.gpu)

# Get hostname and IP
hostname = socket.gethostname()
ip = subprocess.check_output("ifconfig bridge0 | grep 'inet ' | awk '{print $2}'", shell=True).decode().strip()

group = mx.distributed.init()
rank = group.rank() if group else 0
size = group.size() if group else 1

print(f"Rank {rank}/{size} on {hostname} ({ip})")

# BOTH ranks load model to use GPU memory
print(f"Rank {rank} on {hostname}: Loading Qwen3-1.7B-8bit to GPU...")
model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

# Do GPU work on BOTH ranks - HEAVY workload to ensure GPU usage
print(f"Rank {rank} on {hostname}: Running HEAVY GPU computation...")
for i in range(10):
    # Create BIG arrays for real GPU usage
    test_array = mx.random.normal(shape=(5000, 5000))
    result = test_array @ test_array.T
    mx.eval(result)  # Force GPU execution
    print(f"Rank {rank} on {hostname}: GPU iteration {i+1}/10")
    time.sleep(0.5)

# Generate text on rank 0
if rank == 0:
    print(f"Rank 0 on {hostname}: Generating text...")
    response = generate(model, tokenizer, "Write Python hello world:", max_tokens=30)
    print(f"Generated: {response}")
else:
    print(f"Rank {rank} on {hostname}: GPU work completed")

print(f"Rank {rank} on {hostname}: Done")

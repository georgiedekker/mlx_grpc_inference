import mlx.core as mx
from mlx_lm import load, generate
import socket

hostname = socket.gethostname()
mx.set_default_device(mx.gpu)

group = mx.distributed.init()
rank = group.rank() if group else 0

print(f"MLX.launch: Rank {rank} on {hostname}")

if rank == 0:
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    response = generate(model, tokenizer, "# Python code to print hello:", max_tokens=30)
    print(f"Output: {response}")
else:
    # Rank 1 does GPU work to show it's active
    print(f"Rank {rank}: GPU stress test on {hostname}")
    for i in range(5):
        test = mx.random.normal(shape=(5000, 5000))
        mx.eval(test @ test.T)
        print(f"Rank {rank}: Iteration {i+1}/5")

print(f"Rank {rank} on {hostname}: Done")

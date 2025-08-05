#!/bin/bash
# Test PyTorch binding on both machines

# Copy test to mini2
scp test_pytorch_binding.py mini2@mini2.local:/Users/mini2/Movies/mlx_grpc_inference/

# Start worker on mini2
echo "Starting worker on mini2..."
ssh mini2@mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && \
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=192.168.5.1 \
    /Users/mini2/.local/bin/uv run python test_pytorch_binding.py" &

# Start master on mini1
echo "Starting master on mini1..."
RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.168.5.1 \
    uv run python test_pytorch_binding.py

wait
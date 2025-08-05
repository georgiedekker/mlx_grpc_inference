#!/bin/bash
# Test TCP store on both machines

# Copy test script to mini2
scp test_tcp_store.py mini2@mini2.local:/Users/mini2/Movies/mlx_grpc_inference/

# Start rank 1 on mini2 in background
echo "Starting rank 1 on mini2..."
ssh mini2@mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && RANK=1 WORLD_SIZE=2 MASTER_ADDR=192.168.5.1 MASTER_PORT=29500 /Users/mini2/.local/bin/uv run python test_tcp_store.py" &

# Start rank 0 on mini1 immediately (it needs to bind first)
echo "Starting rank 0 on mini1..."
RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.168.5.1 MASTER_PORT=29500 uv run python test_tcp_store.py

# Wait for background job
wait
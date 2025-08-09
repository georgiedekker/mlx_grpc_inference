#!/bin/bash

echo "🚀 MoE Server with MLX Ring Backend"
echo "==================================="

# Sync files to mini2
echo "Syncing to mini2..."
scp -q server_moe.py shard.py base.py qwen_moe_mini.py hosts_tb.json mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Kill existing servers
pkill -f server_moe.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null

echo ""
echo "Starting distributed inference with MLX ring backend..."
echo "====================================================="

# Create absolute path to hosts file
HOSTS_FILE="$(pwd)/hosts_tb.json"

echo "Using hosts file: $HOSTS_FILE"
echo ""

# Launch rank 0 on mini1
echo "Starting rank 0 on mini1..."
MLX_RANK=0 MLX_WORLD_SIZE=2 MLX_HOSTFILE="$HOSTS_FILE" python3 server_moe.py &
RANK0_PID=$!

# Launch rank 1 on mini2
echo "Starting rank 1 on mini2..."
ssh mini2@192.168.5.2 "cd /Users/mini2/Movies/mlx_grpc_inference && MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_HOSTFILE=/Users/mini2/Movies/mlx_grpc_inference/hosts_tb.json python3 server_moe.py" &
RANK1_PID=$!

echo ""
echo "Both processes started!"
echo "Rank 0 PID: $RANK0_PID"
echo "Rank 1 PID: $RANK1_PID"
echo ""
echo "API should be available at: http://localhost:8100"
echo ""
echo "Press Ctrl+C to stop both processes"

# Wait for user interrupt
trap 'echo ""; echo "Stopping processes..."; kill $RANK0_PID 2>/dev/null; ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null; exit 0' INT

# Keep script running
wait
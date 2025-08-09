#!/bin/bash

echo "🔗 Testing MLX Distributed over Thunderbolt"
echo "==========================================="

# Set MLX to use the ring backend explicitly and point to Thunderbolt IPs
export MLX_DISTRIBUTED_BACKEND=ring
export MLX_MASTER_ADDR=192.168.5.1
export MLX_MASTER_PORT=9999

# Kill any existing processes
pkill -f test_mlx_dist.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f test_mlx_dist.py" 2>/dev/null

echo "Starting rank 0 on Thunderbolt IP 192.168.5.1..."
MLX_RANK=0 MLX_WORLD_SIZE=2 python3 test_mlx_dist.py &
RANK0_PID=$!

echo "Starting rank 1 on mini2 (Thunderbolt IP 192.168.5.2)..."
ssh mini2@192.168.5.2 "cd /Users/mini2/Movies/mlx_grpc_inference && MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_MASTER_ADDR=192.168.5.1 MLX_MASTER_PORT=9999 MLX_DISTRIBUTED_BACKEND=ring python3 test_mlx_dist.py" &
RANK1_PID=$!

# Wait for both processes
echo "Waiting for processes to complete..."
wait $RANK0_PID
wait $RANK1_PID

echo "Test completed!"
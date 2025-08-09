#!/bin/bash
# Start rank 1 (worker) on mini2

echo "Starting Rank 1 (Worker) on mini2..."
echo "This will handle layers 8-15"

export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_RANK=1
export OMPI_COMM_WORLD_LOCAL_RANK=0

# Use system python3 with MLX installed
cd /Users/mini2/Movies/mlx_grpc_inference
python3 server_moe.py
#!/bin/bash

echo "🚀 Simple MoE Distributed Launch"
echo "================================"

# Sync files to mini2
echo "Syncing to mini2..."
scp -q server_moe.py shard.py base.py qwen_moe_mini.py start_rank1.sh mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Kill any existing servers
pkill -f server_moe.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null

echo ""
echo "Starting distributed inference..."
echo "================================"
echo ""

# First ensure mini2 has all files
echo "Ensuring mini2 has all required files..."
ssh mini2@192.168.5.2 "cd /Users/mini2/Movies/mlx_grpc_inference && pwd" || {
    echo "Creating directory on mini2..."
    ssh mini2@192.168.5.2 "mkdir -p /Users/mini2/Movies/mlx_grpc_inference"
}

# Use mpirun with both hostfile and rankfile to ensure proper allocation and assignment
mpirun \
    --hostfile hostfile \
    --map-by rankfile:file=rankfile \
    -np 2 \
    bash -c 'if [ "$(hostname)" = "mini1.local" ]; then cd /Users/mini1/Movies/mlx_grpc_inference; else cd /Users/mini2/Movies/mlx_grpc_inference; fi && MLX_RANK=$OMPI_COMM_WORLD_RANK MLX_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE python3 server_moe.py' \
    --mca btl_tcp_if_include 192.168.5.0/24 \
    --mca btl tcp,self
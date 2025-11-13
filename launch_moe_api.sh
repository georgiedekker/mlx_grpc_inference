#!/bin/bash

# Launch distributed MoE inference API server

set -e

echo "🚀 MoE Distributed Inference API Server"
echo "======================================"

# Check if mini2 is reachable
echo "Checking connection to mini2..."
if ping -c 1 -W 1 192.168.5.2 > /dev/null 2>&1; then
    echo "✅ mini2 connected at 192.168.5.2"
    DISTRIBUTED=true
else
    echo "⚠️  mini2 not available, running single-device mode"
    DISTRIBUTED=false
fi

# Copy server to mini2 if distributed
if [ "$DISTRIBUTED" = true ]; then
    echo "Syncing server code to mini2..."
    ssh mini2@192.168.5.2 "mkdir -p /Users/mini2/Movies/mlx_grpc_inference" 2>/dev/null || true
    scp -q server_moe.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/
    
    # Copy MoE model files (now local to mlx_grpc_inference)
    scp -q shard.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/
    scp -q base.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/
    scp -q qwen_moe_mini.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/
    
    # Copy wrapper scripts
    scp -q run_remote.sh mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/
    ssh mini2@192.168.5.2 "chmod +x /Users/mini2/Movies/mlx_grpc_inference/run_remote.sh"
    
    echo "✅ Code synced to mini2"
fi

# Kill any existing servers
echo "Stopping any existing servers..."
pkill -f "server_moe.py" 2>/dev/null || true
ssh mini2@192.168.5.2 "pkill -f 'server_moe.py'" 2>/dev/null || true

# Launch based on mode
if [ "$DISTRIBUTED" = true ]; then
    echo ""
    echo "Starting distributed MoE server..."
    echo "- Rank 0 (mini1): API server + layers 0-7"
    echo "- Rank 1 (mini2): Worker + layers 8-15"
    echo ""
    
    # Use mpirun with wrapper scripts
    echo "Launching with mpirun..."
    
    # Run distributed with wrapper scripts
    mpirun \
        -n 1 -host localhost ./run_local.sh : \
        -n 1 -host 192.168.5.2 /Users/mini2/Movies/mlx_grpc_inference/run_remote.sh \
        --mca btl_tcp_if_include 192.168.5.0/24
else
    echo ""
    echo "Starting single-device MoE server..."
    echo ""
    
    # Run single device
    uv run python server_moe.py
fi
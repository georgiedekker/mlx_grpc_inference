#!/bin/bash

echo "🚀 MLX MoE Server with Thunderbolt Ring"
echo "======================================="

# Sync all files to mini2
echo "Syncing files to mini2..."
scp -q server_moe.py shard.py base.py qwen_moe_mini.py hosts_tb.json mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Kill existing servers
echo "Stopping existing servers..."
pkill -f server_moe.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null

echo ""
echo "Starting MLX distributed server with Thunderbolt ring..."
echo "======================================================="
echo ""

# Use mlx.launch with the hosts file targeting Thunderbolt IPs
mlx.launch --hostfile hosts_tb.json python3 server_moe.py
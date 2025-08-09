#!/bin/bash

echo "🚀 MoE Server with mlx.launch"
echo "=============================="

# Sync files to mini2
echo "Syncing to mini2..."
scp -q server_moe.py shard.py base.py qwen_moe_mini.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Kill existing servers
pkill -f server_moe.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null

echo ""
echo "Using mlx.launch CLI with hosts..."
echo ""

# Create hosts file for mlx.launch
cat > mlx_hosts.json << EOF
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF

# Use mlx.launch CLI
mlx.launch --hostfile mlx_hosts.json python3 server_moe.py
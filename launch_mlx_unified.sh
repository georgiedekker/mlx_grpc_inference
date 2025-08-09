#!/bin/bash

echo "🚀 MLX Unified Distributed Launch"
echo "=================================="

# Sync files to mini2
echo "Syncing files to mini2..."
scp -q server_moe.py shard.py base.py qwen_moe_mini.py hosts_tb.json mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Kill existing servers
pkill -f server_moe.py 2>/dev/null
ssh mini2@192.168.5.2 "pkill -f server_moe.py" 2>/dev/null

echo ""
echo "Using mlx.launch for unified distributed job..."
echo "==============================================="

# Create a proper hosts file for mlx.launch
cat > hosts_mlx.json << 'EOF'
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "mini2@192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF

# Copy hosts file to mini2
scp -q hosts_mlx.json mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

echo "Launching unified distributed job..."
echo ""

# Use mlx.launch to create a unified distributed job
mlx.launch --hostfile hosts_mlx.json --backend ring python3 server_moe.py
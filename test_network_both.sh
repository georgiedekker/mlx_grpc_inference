#!/bin/bash
# Test basic network connectivity

# Copy script to mini2
scp test_network_debug.py mini2@mini2.local:/Users/mini2/Movies/mlx_grpc_inference/

# Start client on mini2 in background
echo "Starting client on mini2..."
ssh mini2@mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && /Users/mini2/.local/bin/uv run python test_network_debug.py client" &

# Start server on mini1
echo "Starting server on mini1..."
uv run python test_network_debug.py server

# Wait
wait
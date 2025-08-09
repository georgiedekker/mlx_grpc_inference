#!/bin/bash
# Wrapper script to run MoE server with proper environment

# Change to the right directory
cd /Users/$(whoami)/Movies/mlx_grpc_inference

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the server
exec python server_moe.py
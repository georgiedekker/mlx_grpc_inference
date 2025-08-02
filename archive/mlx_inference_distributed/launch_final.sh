#!/bin/bash
# Final launch script for distributed MLX

echo "Starting Distributed MLX Inference System..."
echo "Nodes: mini1.local, mini2.local"
echo ""

# Set environment
export DISTRIBUTED_CONFIG="distributed_config.json"
export PYTHONUNBUFFERED=1

# Launch with MPI using full uv path
exec mpirun -np 2 --hostfile hostfile.txt \
    sh -c 'cd ~/Movies/mlx_distributed && /opt/homebrew/bin/uv run python distributed_api.py'
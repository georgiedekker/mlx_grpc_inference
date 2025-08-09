#!/bin/bash
# Start rank 0 (API server) on mini1

echo "Starting Rank 0 (API Server) on mini1..."
echo "This will handle layers 0-7 and serve the API on port 8100"

export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0

# Use system python3 with MLX installed
python3 server_moe.py
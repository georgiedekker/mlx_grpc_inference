#!/bin/bash

echo "============================================"
echo "DISTRIBUTED MLX INFERENCE - WORKING SOLUTION"
echo "============================================"
echo ""
echo "This uses manual process management to ensure"
echo "both mini1 and mini2 GPUs are used."
echo ""

# Clean up any existing processes
echo "Cleaning up..."
pkill -f server.py 2>/dev/null
ssh 192.168.5.2 "pkill -f server.py" 2>/dev/null
sleep 2

# Copy server to mini2
echo "Syncing server.py to mini2..."
scp -q server.py 192.168.5.2:/Users/mini2/

# Start both processes with MPI coordination
echo "Starting distributed inference..."
echo "  Mini1: Rank 0 (layers 0-13)"
echo "  Mini2: Rank 1 (layers 14-27)"
echo ""

# Use mpirun with specific executable paths
mpirun -n 1 -host localhost /Users/mini1/Movies/mlx_grpc_inference/.venv/bin/python /Users/mini1/Movies/mlx_grpc_inference/server.py : \
       -n 1 -host 192.168.5.2 /Users/mini2/.venv/bin/python /Users/mini2/server.py
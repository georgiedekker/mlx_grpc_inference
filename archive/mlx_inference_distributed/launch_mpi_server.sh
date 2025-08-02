#!/bin/bash
# Launch the MPI-based distributed inference server

echo "Starting MPI-based Distributed MLX Inference System..."
echo "This uses MPI for coordination with a single API endpoint on master."
echo ""

cd ~/Movies/mlx_distributed

# Copy the script to mini2
echo "Copying files to mini2..."
scp mpi_inference_server.py distributed_api.py mini2.local:~/Movies/mlx_distributed/

# Launch with MPI
echo "Launching distributed system..."
exec mpirun -np 2 --hostfile hostfile.txt \
    sh -c 'cd ~/Movies/mlx_distributed && /opt/homebrew/bin/uv run python mpi_inference_server.py'
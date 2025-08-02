#!/bin/bash
# Test GPU on both nodes
cd ~/Movies/mlx_distributed
exec mpirun -np 2 --hostfile hostfile.txt sh -c 'cd ~/Movies/mlx_distributed && /opt/homebrew/bin/uv run python test_gpu_usage.py'
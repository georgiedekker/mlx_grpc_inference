#!/bin/bash
# Test basic distributed setup with MLX

# First, let's test on single machine
echo "Testing single device inference..."
python simple_distributed.py --prompt "What is 2+2?"

# For distributed across multiple machines, we need to set up MLX distributed
# MLX uses MPI-style setup
echo -e "\n\nFor distributed inference across machines:"
echo "1. On mini1: MLX_RANK=0 MLX_WORLD_SIZE=3 python simple_distributed.py"
echo "2. On mini2: MLX_RANK=1 MLX_WORLD_SIZE=3 python simple_distributed.py" 
echo "3. On m4:    MLX_RANK=2 MLX_WORLD_SIZE=3 python simple_distributed.py"
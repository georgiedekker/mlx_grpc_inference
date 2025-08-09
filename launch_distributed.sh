#!/bin/bash

echo "============================================"
echo "MLX DISTRIBUTED INFERENCE (mini1 + mini2)"
echo "============================================"
echo ""

# Ensure we're in the right directory
cd /Users/mini1/Movies/mlx_grpc_inference

# Copy server to mini2
echo "Syncing server.py to mini2..."
scp -q server.py 192.168.5.2:/Users/mini2/

# Create a wrapper script that uses local Python on each machine
cat > run_server_wrapper.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server
server.main()
EOF

# Copy wrapper to mini2
scp -q run_server_wrapper.py 192.168.5.2:/Users/mini2/

echo "Starting distributed server..."
echo "  Rank 0: mini1 (layers 0-13)"
echo "  Rank 1: mini2 (layers 14-27)"
echo "  API: http://localhost:8100"
echo ""

# Run with mpirun directly to avoid path issues
mpirun -n 2 \
  --host mini1.local,mini2.local \
  --map-by rankfile:file=rankfile \
  bash -c 'cd $(hostname -s | grep -q mini1 && echo /Users/mini1/Movies/mlx_grpc_inference || echo /Users/mini2) && .venv/bin/python server.py'
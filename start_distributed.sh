#!/bin/bash

echo "============================================"
echo "STARTING DISTRIBUTED MLX INFERENCE"
echo "============================================"
echo ""

# Kill any existing servers
pkill -f server.py 2>/dev/null
ssh 192.168.5.2 "pkill -f server.py" 2>/dev/null
sleep 2

# Copy server to mini2
echo "Copying server.py to mini2..."
scp -q server.py 192.168.5.2:/Users/mini2/

# Start rank 1 on mini2 in background
echo "Starting rank 1 on mini2..."
ssh 192.168.5.2 "cd /Users/mini2 && MLX_RANK=1 MLX_WORLD_SIZE=2 .venv/bin/python server.py" > mini2.log 2>&1 &
MINI2_PID=$!

# Start rank 0 on mini1
echo "Starting rank 0 on mini1..."
echo ""
echo "API will be at: http://localhost:8100"
echo "Press Ctrl+C to stop both servers"
echo ""

# Set environment for rank 0
export MLX_RANK=0
export MLX_WORLD_SIZE=2

# Run rank 0 with uv
uv run python server.py

# Cleanup when stopped
echo "Stopping servers..."
kill $MINI2_PID 2>/dev/null
ssh 192.168.5.2 "pkill -f server.py" 2>/dev/null
pkill -f server.py 2>/dev/null
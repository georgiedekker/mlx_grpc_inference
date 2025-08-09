#!/bin/bash

# Test the distributed sharded server

echo "Testing Distributed Sharded Server"
echo "==================================="

# Clean up any existing processes
pkill -f distributed_server.py 2>/dev/null || true
ssh mini2@192.168.5.2 "pkill -f distributed_server.py" 2>/dev/null || true
sleep 2

# Copy the new server files to mini2
echo "Copying files to mini2..."
scp -q distributed_server.py shard.py sharded_model_loader.py mini2@192.168.5.2:/Users/mini2/

# Start the distributed server with explicit host specification
echo "Starting distributed sharded server..."
echo "Using bridge0 interface for Thunderbolt communication..."

# Use the approach from working launch.sh
nohup mpirun \
    -n 1 -host localhost /Users/mini1/Movies/mlx_grpc_inference/.venv/bin/python /Users/mini1/Movies/mlx_grpc_inference/distributed_server.py : \
    -n 1 -host mini2@192.168.5.2 /Users/mini2/.venv/bin/python /Users/mini2/distributed_server.py \
    > distributed_server.log 2>&1 &

PID=$!
echo "Server PID: $PID"

# Wait for server to start
echo "Waiting for server to initialize (20 seconds)..."
sleep 20

# Test the health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8100/health | jq '.' || echo "Health check failed"

# Test inference
echo ""
echo "Testing inference..."
curl -X POST http://localhost:8100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 20,
        "temperature": 0.7
    }' | jq '.' || echo "Inference test failed"

echo ""
echo "Press Ctrl+C to stop the server"
wait $PID
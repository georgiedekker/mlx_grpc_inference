#!/bin/bash

# Test the pipeline parallel server

echo "Testing Pipeline Parallel Server"
echo "================================"

# Clean up any existing processes
pkill -f pipeline_server.py 2>/dev/null || true
ssh 192.168.5.2 "pkill -f pipeline_server.py" 2>/dev/null || true
sleep 2

# Copy the new server files to mini2
echo "Copying files to mini2..."
scp -q pipeline_server.py shard.py sharded_model_loader.py 192.168.5.2:/Users/mini2/

# Start the pipeline server
echo "Starting pipeline parallel server..."
mpirun \
    -n 1 -host 192.168.5.1 bash -c "cd /Users/mini1/Movies/mlx_grpc_inference && source .venv/bin/activate && python pipeline_server.py" : \
    -n 1 -host 192.168.5.2 bash -c "cd /Users/mini2 && source .venv/bin/activate && python pipeline_server.py" \
    --mca btl_tcp_if_include bridge0 \
    --mca btl self,tcp &

PID=$!
echo "Server PID: $PID"

# Wait for server to start
echo "Waiting for server to initialize (15 seconds)..."
sleep 15

# Test the health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8100/health | jq '.'

# Test inference
echo ""
echo "Testing inference..."
curl -X POST http://localhost:8100/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 20,
        "temperature": 0.7
    }' | jq '.'

echo ""
echo "Press Ctrl+C to stop the server"
wait $PID
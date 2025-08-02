#!/bin/bash
# Start a local-only test with all 3 processes on mini1

set -e

echo "🚀 STARTING LOCAL 3-PROCESS TEST ON MINI1"
echo "========================================="

# Clean up
echo "🧹 Cleaning up old processes..."
pkill -f "python.*distributed" || true
pkill -f "python.*worker" || true
pkill -f "python.*grpc_server" || true
sleep 2

cd /Users/mini1/Movies/mlx_inference_distributed

# Start worker processes locally (simulating distributed)
echo ""
echo "📍 Starting local workers to simulate 3-device cluster..."

# Worker 1 (simulating mini2) 
echo "Starting worker 1 (rank 1) on port 50051..."
PYTHONUNBUFFERED=1 GRPC_DNS_RESOLVER=native LOCAL_RANK=1 WORKER_PORT=50051 \
    DISTRIBUTED_CONFIG=distributed_config.json \
    uv run python grpc_server.py --rank=1 --port=50051 > logs/worker1_local.log 2>&1 &
WORKER1_PID=$!
echo "Worker 1 PID: $WORKER1_PID"

# Worker 2 (simulating master)
echo "Starting worker 2 (rank 2) on port 50052..."
PYTHONUNBUFFERED=1 GRPC_DNS_RESOLVER=native LOCAL_RANK=2 WORKER_PORT=50052 \
    DISTRIBUTED_CONFIG=distributed_config.json \
    uv run python grpc_server.py --rank=2 --port=50052 > logs/worker2_local.log 2>&1 &
WORKER2_PID=$!
echo "Worker 2 PID: $WORKER2_PID"

echo "Waiting for workers to start..."
sleep 10

# Check workers
echo ""
echo "🔍 Checking worker status..."
if nc -z localhost 50051 2>/dev/null; then
    echo "✅ Worker 1 (port 50051) is responding"
else
    echo "❌ Worker 1 not responding"
fi

if nc -z localhost 50052 2>/dev/null; then
    echo "✅ Worker 2 (port 50052) is responding"
else
    echo "❌ Worker 2 not responding"
fi

# Start API server
echo ""
echo "📍 Starting API server (coordinator)..."
PYTHONUNBUFFERED=1 GRPC_DNS_RESOLVER=native LOCAL_RANK=0 \
    DISTRIBUTED_CONFIG=distributed_config.json API_PORT=8100 \
    uv run python distributed_api.py > logs/api_local.log 2>&1 &
API_PID=$!
echo "API server PID: $API_PID"

echo "Waiting for API server to start..."
sleep 15

# Test the system
echo ""
echo "🧪 Testing system health..."
if curl -s -f http://localhost:8100/health > /dev/null; then
    echo "✅ API server is healthy"
    curl -s http://localhost:8100/health | python -m json.tool
else
    echo "❌ API server not responding"
fi

echo ""
echo "🧪 Testing inference..."
curl -X POST http://localhost:8100/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mlx-community/Qwen3-1.7B-8bit", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20}' \
     -w "\nHTTP Status: %{http_code}\n" | python -m json.tool || true

echo ""
echo "📋 Process IDs:"
echo "   Worker 1: $WORKER1_PID"
echo "   Worker 2: $WORKER2_PID"
echo "   API Server: $API_PID"
echo ""
echo "📋 Logs:"
echo "   tail -f logs/worker1_local.log"
echo "   tail -f logs/worker2_local.log"
echo "   tail -f logs/api_local.log"
echo ""
echo "🛑 To stop all: kill $WORKER1_PID $WORKER2_PID $API_PID"
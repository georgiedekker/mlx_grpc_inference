#!/bin/bash
# Simple test startup for debugging

set -e

echo "🚀 STARTING SIMPLE 3-DEVICE TEST"
echo "================================"

# Clean up
echo "🧹 Cleaning up old processes..."
pkill -f "python.*distributed" || true
pkill -f "python.*worker" || true
sleep 2

# Test 1: Just the API server in standalone mode
echo ""
echo "📍 TEST 1: Starting API server in standalone mode..."
cd /Users/mini1/Movies/mlx_inference_distributed

echo "Starting API server on mini1..."
PYTHONUNBUFFERED=1 GRPC_DNS_RESOLVER=native LOCAL_RANK=0 DISTRIBUTED_CONFIG=distributed_config.json API_PORT=8100 \
    uv run python distributed_api.py 2>&1 | tee api_debug.log &

API_PID=$!
echo "API PID: $API_PID"

echo "Waiting for startup..."
sleep 15

# Test health
echo ""
echo "🧪 Testing health endpoint..."
if curl -s -f http://localhost:8100/health > /dev/null; then
    echo "✅ Health check passed"
    curl -s http://localhost:8100/health | python -m json.tool
else
    echo "❌ Health check failed"
    echo "Check api_debug.log for errors"
    exit 1
fi

# Test inference
echo ""
echo "🧪 Testing inference..."
curl -X POST http://localhost:8100/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mlx-community/Qwen3-1.7B-8bit", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}' \
     -w "\nHTTP Status: %{http_code}\n"

echo ""
echo "📋 Check api_debug.log for detailed error messages"
echo "🛑 To stop: kill $API_PID"
#!/bin/bash
# Start the dynamic MLX distributed cluster with auto-discovery

echo "🚀 Starting Dynamic MLX Distributed Cluster"
echo "============================================"
echo ""
echo "This cluster supports:"
echo "  ✅ Automatic device discovery (mDNS/Bonjour)"
echo "  ✅ Dynamic shard rebalancing"
echo "  ✅ Thunderbolt networking"
echo "  ✅ Maximum RAM utilization"
echo ""

# Kill any existing processes
echo "🛑 Cleaning up existing processes..."
pkill -f "distributed_api_dynamic.py" || true
pkill -f "worker_simple.py" || true

# Install zeroconf if needed
echo "📦 Checking dependencies..."
source .venv/bin/activate
pip show zeroconf > /dev/null 2>&1 || pip install zeroconf

# Start the API server with dynamic discovery
echo "🌐 Starting API server with dynamic discovery..."
cd /Users/mini1/Movies/mlx_inference_distributed
nohup python distributed_api_dynamic.py > logs/api_dynamic.log 2>&1 &
API_PID=$!

echo "⏳ Waiting for API server to start..."
sleep 5

# Check if API is running
if curl -s http://localhost:8100/health > /dev/null 2>&1; then
    echo "✅ API server started successfully"
else
    echo "❌ API server failed to start. Check logs/api_dynamic.log"
    exit 1
fi

echo ""
echo "📡 Cluster Status:"
curl -s http://localhost:8100/distributed/cluster-info | python -m json.tool

echo ""
echo "======================================"
echo "🎯 To add workers to this cluster:"
echo ""
echo "On any Mac with Apple Silicon, run:"
echo "  python worker_simple.py"
echo ""
echo "Workers will be automatically discovered!"
echo ""
echo "To test inference:"
echo '  curl -X POST http://localhost:8100/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model": "mlx-community/Qwen3-1.7B-8bit", "messages": [{"role": "user", "content": "Hello!"}]}'"'"
echo ""
echo "To monitor cluster:"
echo "  curl http://localhost:8100/distributed/cluster-info"
echo ""
echo "To stop:"
echo "  pkill -f distributed_api_dynamic.py"
echo "======================================"
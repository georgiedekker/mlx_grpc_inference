#!/bin/bash
# Start the 3-device distributed MLX cluster

echo "🚀 Starting 3-device distributed MLX cluster..."

# Kill any existing processes
echo "🛑 Cleaning up existing processes..."
pkill -f "python.*worker.py" || true
pkill -f "python.*distributed_api.py" || true
ssh mini2.local "pkill -f 'python.*worker.py'" || true
ssh georgedekker@master.local "pkill -f 'python.*worker.py'" || true

# Start worker on mini2 (rank 1)
echo "🟢 Starting worker on mini2 (rank 1)..."
ssh mini2.local "cd Movies/mlx_inference_distributed && source .venv/bin/activate && GRPC_DNS_RESOLVER=native LOCAL_RANK=1 DISTRIBUTED_CONFIG=distributed_config.json nohup python worker.py --rank=1 > logs/worker_rank1.log 2>&1 &"

# Start worker on master.local (rank 2)
echo "🟢 Starting worker on master.local (rank 2)..."
ssh georgedekker@master.local "cd ~/Movies/mlx_inference_distributed && source .venv/bin/activate && GRPC_DNS_RESOLVER=native LOCAL_RANK=2 DISTRIBUTED_CONFIG=distributed_config.json nohup python worker.py --rank=2 > logs/worker_rank2.log 2>&1 &"

# Wait for workers to start
echo "⏳ Waiting for workers to initialize..."
sleep 10

# Check workers are running
echo "🔍 Checking worker status..."
nc -zv mini2.local 50101 || echo "Warning: mini2 worker not responding on port 50101"
nc -zv master.local 50102 || echo "Warning: master worker not responding on port 50102"

# Start master on mini1 (rank 0)
echo "🟢 Starting master on mini1 (rank 0)..."
cd /Users/mini1/Movies/mlx_inference_distributed
source .venv/bin/activate && GRPC_DNS_RESOLVER=native LOCAL_RANK=0 DISTRIBUTED_CONFIG=distributed_config.json API_PORT=8100 nohup python distributed_api.py > logs/api_server.log 2>&1 &

echo "⏳ Waiting for API server to start..."
sleep 10

# Test the cluster
echo "🧪 Testing 3-device cluster..."
echo ""
echo "Health check:"
curl -s http://localhost:8100/health | python -m json.tool

echo ""
echo "GPU info (should show 3 devices):"
curl -s http://localhost:8100/distributed/gpu-info | python -m json.tool

echo ""
echo "To test inference:"
echo "curl -X POST http://localhost:8100/v1/chat/completions -H \"Content-Type: application/json\" -d '{\"model\": \"mlx-community/Qwen3-1.7B-8bit\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"

echo ""
echo "To monitor logs:"
echo "  mini1: tail -f logs/api_server.log"
echo "  mini2: ssh mini2.local 'tail -f Movies/mlx_distributed/logs/worker_rank1.log'"
echo "  master: ssh georgedekker@master.local 'tail -f ~/Movies/mlx_distributed/logs/worker_rank2.log'"
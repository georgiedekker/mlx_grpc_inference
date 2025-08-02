#!/bin/bash
# PRODUCTION MLX Distributed Inference Cluster Startup
# Supports flexible coordinator selection and proper 3-device operation

set -e

echo "🚀 MLX DISTRIBUTED INFERENCE CLUSTER"
echo "==================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Device configuration
MINI1_HOST="mini1.local"
MINI2_HOST="mini2.local"
MASTER_HOST="master.local"

# Create logs directory
mkdir -p logs

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "python.*distributed_api" || true
pkill -f "python.*grpc_server" || true
pkill -f "python.*worker" || true

# Also clean remote hosts
ssh mini2@mini2.local "pkill -f 'python.*grpc_server' || true" 2>/dev/null || true
ssh georgedekker@master.local "pkill -f 'python.*grpc_server' || true" 2>/dev/null || true

sleep 3

# OPTION 1: LOCAL TEST MODE (all on mini1)
if [[ "$1" == "--local" ]]; then
    echo "📍 Running in LOCAL TEST MODE (all processes on mini1)"
    echo
    
    # Start local workers
    echo "Starting worker mini2 (local simulation)..."
    GRPC_DNS_RESOLVER=native DISTRIBUTED_CONFIG=distributed_config.json \
        uv run python grpc_server.py --device-id mini2 --port 50051 > logs/worker_mini2.log 2>&1 &
    WORKER1_PID=$!
    
    echo "Starting worker master (local simulation)..."
    GRPC_DNS_RESOLVER=native DISTRIBUTED_CONFIG=distributed_config.json \
        uv run python grpc_server.py --device-id master --port 50051 > logs/worker_master.log 2>&1 &
    WORKER2_PID=$!
    
    echo "Workers started: PIDs $WORKER1_PID, $WORKER2_PID"
    
    # Wait for workers
    sleep 10
    
    # Start API server
    echo "Starting API server (coordinator)..."
    GRPC_DNS_RESOLVER=native LOCAL_RANK=0 DISTRIBUTED_CONFIG=distributed_config.json \
        API_PORT=8100 uv run python distributed_api.py > logs/api_server.log 2>&1 &
    API_PID=$!
    
    echo "API server started: PID $API_PID"
    
    # Wait for API server
    sleep 15
    
    # Test health
    echo ""
    echo "🧪 Testing cluster health..."
    if curl -s -f http://localhost:8100/health >/dev/null; then
        echo "✅ Cluster is healthy!"
        echo
        curl -s http://localhost:8100/health | python -m json.tool
    else
        echo "❌ Cluster health check failed"
        echo "Check logs/api_server.log for details"
    fi
    
    echo ""
    echo "📋 Local test cluster running:"
    echo "   Worker 1 (mini2): PID $WORKER1_PID"
    echo "   Worker 2 (master): PID $WORKER2_PID"
    echo "   API Server: PID $API_PID"
    echo ""
    echo "🛑 To stop: kill $WORKER1_PID $WORKER2_PID $API_PID"
    
# OPTION 2: DISTRIBUTED MODE (across actual devices)
else
    echo "📍 Running in DISTRIBUTED MODE (across 3 devices)"
    echo
    
    # Start workers on remote devices
    echo "Starting worker on mini2.local..."
    ssh mini2@mini2.local "cd Movies/mlx_inference_distributed && \
        source .venv/bin/activate && \
        GRPC_DNS_RESOLVER=native DISTRIBUTED_CONFIG=distributed_config.json \
        nohup python grpc_server.py --device-id mini2 --port 50051 > logs/grpc_server.log 2>&1 &"
    
    echo "Starting worker on master.local..."  
    ssh georgedekker@master.local "cd ~/Movies/mlx_inference_distributed && \
        source .venv/bin/activate && \
        GRPC_DNS_RESOLVER=native DISTRIBUTED_CONFIG=distributed_config.json \
        nohup python grpc_server.py --device-id master --port 50051 > logs/grpc_server.log 2>&1 &"
    
    echo "Waiting for workers to start..."
    sleep 15
    
    # Check worker status
    echo ""
    echo "🔍 Checking worker status..."
    if nc -z mini2.local 50051 2>/dev/null; then
        echo "✅ mini2.local worker is responding"
    else
        echo "⚠️  mini2.local worker not responding"
    fi
    
    if nc -z master.local 50051 2>/dev/null; then
        echo "✅ master.local worker is responding"
    else
        echo "⚠️  master.local worker not responding"
    fi
    
    # Start API server on mini1
    echo ""
    echo "Starting API server on mini1..."
    GRPC_DNS_RESOLVER=native LOCAL_RANK=0 DISTRIBUTED_CONFIG=distributed_config.json \
        API_PORT=8100 nohup uv run python distributed_api.py > logs/api_server.log 2>&1 &
    
    API_PID=$!
    echo "API server started: PID $API_PID"
    
    # Wait for API server
    sleep 15
    
    # Test the distributed cluster
    echo ""
    echo "🧪 Testing distributed cluster..."
    if curl -s -f http://localhost:8100/health >/dev/null; then
        echo "✅ Distributed cluster is healthy!"
        echo
        curl -s http://localhost:8100/health | python -m json.tool
        echo
        echo "📊 GPU cluster info:"
        curl -s http://localhost:8100/distributed/gpu-info | python -m json.tool | head -20
    else
        echo "❌ Cluster health check failed"
        echo "Check logs/api_server.log for details"
    fi
    
    echo ""
    echo "📋 Distributed cluster running across:"
    echo "   mini1.local: API server (coordinator)"
    echo "   mini2.local: Worker (grpc_server)"
    echo "   master.local: Worker (grpc_server)"
fi

echo ""
echo "🎯 CLUSTER READY FOR INFERENCE!"
echo ""
echo "Test with:"
echo 'curl -X POST http://localhost:8100/v1/chat/completions \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"model": "mlx-community/Qwen3-1.7B-8bit", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'"'"''
echo ""
echo "Monitor logs:"
echo "  tail -f logs/api_server.log"
echo "  tail -f logs/worker_*.log"
echo ""
echo "Stop cluster: ./stop_cluster.sh"
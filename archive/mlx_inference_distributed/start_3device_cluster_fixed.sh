#!/bin/bash
# Fixed 3-device distributed MLX cluster startup script
# Flexible coordinator selection and proper port configuration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/distributed_config.json"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Default device hostnames and users
MINI1_HOST="mini1.local"
MINI2_HOST="mini2.local" 
MASTER_HOST="master.local"
MASTER_USER="georgedekker"

# Detect which device we're running from
CURRENT_HOST=$(hostname -s)
echo "🖥️  Running from: ${CURRENT_HOST}"

# Determine coordinator based on where script is run
if [[ "$CURRENT_HOST" == "master" ]]; then
    COORDINATOR_HOST="$MASTER_HOST"
    COORDINATOR_USER="$MASTER_USER"
    WORKER1_HOST="$MINI1_HOST"
    WORKER1_USER="mini1"
    WORKER2_HOST="$MINI2_HOST" 
    WORKER2_USER="mini2"
    echo "🎯 Coordinator: master.local"
elif [[ "$CURRENT_HOST" == "mini2" ]]; then
    COORDINATOR_HOST="$MINI2_HOST"
    COORDINATOR_USER="mini2"
    WORKER1_HOST="$MINI1_HOST"
    WORKER1_USER="mini1"
    WORKER2_HOST="$MASTER_HOST"
    WORKER2_USER="$MASTER_USER"
    echo "🎯 Coordinator: mini2.local"
else
    # Default: mini1 is coordinator
    COORDINATOR_HOST="$MINI1_HOST"
    COORDINATOR_USER="mini1"
    WORKER1_HOST="$MINI2_HOST"
    WORKER1_USER="mini2"
    WORKER2_HOST="$MASTER_HOST"
    WORKER2_USER="$MASTER_USER"
    echo "🎯 Coordinator: mini1.local (default)"
fi

echo "🚀 Starting 3-device distributed MLX cluster..."
echo "   Coordinator: $COORDINATOR_HOST"
echo "   Worker 1: $WORKER1_HOST"
echo "   Worker 2: $WORKER2_HOST"

# Create logs directory
mkdir -p "$LOGS_DIR"

# Kill any existing processes
echo "🛑 Cleaning up existing processes..."
pkill -f "python.*worker.py" || true
pkill -f "python.*distributed_api.py" || true
pkill -f "python.*grpc_server.py" || true

# Clean up on remote devices
if [[ "$WORKER1_HOST" != "$COORDINATOR_HOST" ]]; then
    ssh ${WORKER1_USER}@${WORKER1_HOST} "pkill -f 'python.*worker.py' || true" || echo "Warning: Could not cleanup on $WORKER1_HOST"
fi

if [[ "$WORKER2_HOST" != "$COORDINATOR_HOST" ]]; then
    ssh ${WORKER2_USER}@${WORKER2_HOST} "pkill -f 'python.*worker.py' || true" || echo "Warning: Could not cleanup on $WORKER2_HOST"
fi

# Function to start worker
start_worker() {
    local host=$1
    local user=$2
    local rank=$3
    local port=$4
    
    echo "🟢 Starting worker on $host (rank $rank, port $port)..."
    
    if [[ "$host" == "$COORDINATOR_HOST" ]]; then
        # Local worker
        cd "$SCRIPT_DIR"
        GRPC_DNS_RESOLVER=native LOCAL_RANK=$rank WORKER_PORT=$port DISTRIBUTED_CONFIG=distributed_config.json \
        nohup python worker.py --rank=$rank --port=$port > logs/worker_rank${rank}.log 2>&1 &
        local worker_pid=$!
        echo "   Started local worker PID: $worker_pid"
    else
        # Remote worker
        ssh ${user}@${host} "cd Movies/mlx_inference_distributed && \
            export GRPC_DNS_RESOLVER=native && \
            export LOCAL_RANK=$rank && \
            export WORKER_PORT=$port && \
            export DISTRIBUTED_CONFIG=distributed_config.json && \
            mkdir -p logs && \
            nohup python worker.py --rank=$rank --port=$port > logs/worker_rank${rank}.log 2>&1 &" || {
                echo "❌ Failed to start worker on $host"
                return 1
            }
        echo "   Started remote worker on $host"
    fi
}

# Start workers with proper port configuration
# Use ports that match gRPC configuration: 50051, 50052, 50053
start_worker "$WORKER1_HOST" "$WORKER1_USER" 1 50052
start_worker "$WORKER2_HOST" "$WORKER2_USER" 2 50053

# Wait for workers to start
echo "⏳ Waiting for workers to initialize..."
sleep 15

# Check workers are responding
echo "🔍 Checking worker status..."
check_worker() {
    local host=$1
    local port=$2
    local rank=$3
    
    if nc -z -w5 "$host" "$port" 2>/dev/null; then
        echo "   ✅ Worker rank $rank ($host:$port) is responding"
        return 0
    else
        echo "   ❌ Worker rank $rank ($host:$port) is not responding"
        return 1
    fi
}

worker1_ok=false
worker2_ok=false

check_worker "$WORKER1_HOST" 50052 1 && worker1_ok=true
check_worker "$WORKER2_HOST" 50053 2 && worker2_ok=true

if [[ "$worker1_ok" == false ]] && [[ "$worker2_ok" == false ]]; then
    echo "❌ Both workers failed to start. Aborting."
    exit 1
elif [[ "$worker1_ok" == false ]] || [[ "$worker2_ok" == false ]]; then
    echo "⚠️  One worker failed, but continuing with available workers..."
fi

# Start coordinator/API server on rank 0
echo "🟢 Starting coordinator on $COORDINATOR_HOST (rank 0)..."
cd "$SCRIPT_DIR"
GRPC_DNS_RESOLVER=native \
LOCAL_RANK=0 \
COORDINATOR_PORT=50051 \
DISTRIBUTED_CONFIG=distributed_config.json \
API_PORT=8100 \
nohup python distributed_api.py > logs/api_server.log 2>&1 &

API_PID=$!
echo "   Started API server PID: $API_PID"

echo "⏳ Waiting for API server to start..."
sleep 10

# Test the cluster
echo "🧪 Testing cluster health..."

# Check API server
if curl -s -f http://localhost:8100/health >/dev/null 2>&1; then
    echo "   ✅ API server is responding"
    
    echo ""
    echo "📊 Cluster Status:"
    curl -s http://localhost:8100/health | python -m json.tool 2>/dev/null || echo "Could not parse health response"
    
    echo ""
    echo "🔧 GPU Information:"
    curl -s http://localhost:8100/distributed/gpu-info | python -m json.tool 2>/dev/null || echo "Could not get GPU info"
    
else
    echo "   ❌ API server is not responding"
    echo "   Check logs: tail -f logs/api_server.log"
    exit 1
fi

echo ""
echo "🎉 Cluster started successfully!"
echo ""
echo "📚 Usage:"
echo "  Health check: curl http://localhost:8100/health"
echo "  Test inference:"
echo "    curl -X POST http://localhost:8100/v1/chat/completions \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"model\": \"mlx-community/Qwen3-1.7B-8bit\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"
echo ""
echo "📋 Monitoring:"
echo "  API server: tail -f logs/api_server.log"
echo "  Worker 1: tail -f logs/worker_rank1.log"
echo "  Worker 2: tail -f logs/worker_rank2.log"
echo ""
echo "🛑 To stop cluster: ./stop_cluster.sh"
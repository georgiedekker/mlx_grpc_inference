#!/bin/bash
# Launch distributed MLX server cluster with UV
# Runs on agreed port 8100 for coordinator

set -e

echo "🚀 Starting Distributed MLX Cluster with UV"
echo "========================================"

# Kill any existing servers
echo "🧹 Cleaning up existing processes..."
pkill -f "distributed_server.py" || true
pkill -f "simple_server.py" || true
sleep 2

# Determine which device we're on
HOSTNAME=$(hostname)
echo "📍 Current hostname: ${HOSTNAME}"

# Function to launch server
launch_server() {
    local device=$1
    local role=$2
    
    echo ""
    echo "🚀 Launching $role on $device..."
    
    if [[ "$device" == "local" ]]; then
        # Launch locally
        uv run python distributed_server.py &
        PID=$!
        echo "✅ Started $role (PID: $PID)"
    else
        # Launch on remote device
        echo "📡 Starting $role on $device..."
        ssh $device "cd /Users/mini1/Movies/mlx_training_distributed && uv run python distributed_server.py" &
        echo "✅ Started $role on $device"
    fi
}

# Launch based on hostname
if [[ "${HOSTNAME}" == *"mini1"* ]]; then
    echo "🎯 This is mini1 - launching full cluster"
    
    # Start coordinator on mini1 (port 8100)
    launch_server "local" "coordinator"
    sleep 5
    
    # Start workers on other devices
    echo ""
    echo "🔄 Starting workers on remote devices..."
    
    # mini2 worker (port 8101)
    launch_server "mini2.local" "worker"
    
    # master worker (port 8102)
    launch_server "master.local" "worker"
    
    echo ""
    echo "⏳ Waiting for cluster to initialize..."
    sleep 10
    
    # Check cluster status
    echo ""
    echo "📊 Checking cluster status..."
    curl -s http://localhost:8100/cluster-status | python -m json.tool
    
    echo ""
    echo "✅ Distributed MLX cluster is running!"
    echo ""
    echo "🔗 Endpoints:"
    echo "   http://localhost:8100/              - Root"
    echo "   http://localhost:8100/v1/models     - List models"
    echo "   http://localhost:8100/cluster-status - Cluster status"
    echo "   http://localhost:8100/health        - Health check"
    echo ""
    echo "🧪 Test with:"
    echo '   curl -X POST http://localhost:8100/v1/chat/completions \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"model": "qwen", "messages": [{"role": "user", "content": "Hello!"}]}'"'"
    
elif [[ "${HOSTNAME}" == *"mini2"* ]] || [[ "${HOSTNAME}" == *"master"* ]]; then
    echo "🎯 This is a worker node - starting worker service"
    launch_server "local" "worker"
    
else
    echo "❌ Unknown hostname: ${HOSTNAME}"
    echo "   This script should be run from mini1, mini2, or master"
    exit 1
fi

# Keep script running
echo ""
echo "Press Ctrl+C to stop the cluster..."
wait
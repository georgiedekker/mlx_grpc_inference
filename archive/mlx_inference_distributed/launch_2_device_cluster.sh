#!/bin/bash
# Launch 2-device distributed MLX inference cluster
# Team A - Production distributed setup

set -e

echo "🚀 Starting 2-Device MLX Distributed Cluster"
echo "============================================="

# Configuration
MASTER_HOST="mini1.local"
MASTER_PORT=8100
WORKER_HOST="mini2.local"
GRPC_PORT_MASTER=50100
GRPC_PORT_WORKER=50101

echo "📋 Cluster Configuration:"
echo "  Master: ${MASTER_HOST}:${MASTER_PORT} (gRPC: ${GRPC_PORT_MASTER})"
echo "  Worker: ${WORKER_HOST} (gRPC: ${GRPC_PORT_WORKER})"
echo ""

# Check if we're on the master node
HOSTNAME=$(hostname)
echo "🔍 Current hostname: ${HOSTNAME}"

if [[ "${HOSTNAME}" == "mini1"* ]]; then
    echo "🎯 Running on MASTER node (mini1)"
    
    # Step 1: Start the distributed API server (coordinator)
    echo "1️⃣ Starting distributed API server..."
    cd /Users/mini1/Movies/mlx_distributed
    
    # Set environment variables for 2-device setup
    export WORLD_SIZE=2
    export RANK=0
    export MASTER_ADDR="mini1.local"
    export MASTER_PORT=50100
    
    # Start the API server which will coordinate both devices
    echo "🌟 Launching API server with 2-device coordination..."
    uv run python -m src.mlx_distributed.distributed_api &
    API_PID=$!
    
    echo "✅ API server started (PID: ${API_PID})"
    echo "📡 Coordinator will manage both mini1 and mini2"
    echo ""
    echo "🔗 Test endpoints:"
    echo "  http://mini1.local:8100/v1/models"
    echo "  http://mini1.local:8100/distributed/gpu-info" 
    echo "  http://mini1.local:8100/distributed/cluster-status"
    echo ""
    echo "🎉 2-device cluster should now show both mini1 AND mini2!"
    
    # Keep the script running
    wait $API_PID

elif [[ "${HOSTNAME}" == "mini2"* ]]; then
    echo "🎯 Running on WORKER node (mini2)"
    echo "⚠️  This script should be run from mini1 (master)"
    echo "   The master will coordinate both devices automatically"
    exit 1
    
else
    echo "❌ Unknown hostname: ${HOSTNAME}"
    echo "   Expected: mini1 or mini2"
    exit 1
fi
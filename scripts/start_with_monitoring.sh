#!/bin/bash
# Start the distributed inference system with comprehensive monitoring

set -e

echo "=== Starting MLX Distributed Inference with Monitoring ==="
echo ""

# Enable comprehensive monitoring
export GRPC_TRACE=all
export GRPC_VERBOSITY=DEBUG
export MLX_METAL_DEBUG=ON

# Log monitoring settings
echo "Monitoring enabled:"
echo "  - GRPC_TRACE: all"
echo "  - GRPC_VERBOSITY: DEBUG"
echo "  - MLX_METAL_DEBUG: ON"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to start a service with monitoring
start_service() {
    local service_name=$1
    local command=$2
    local log_file=$3
    
    echo "Starting $service_name..."
    echo "  Command: $command"
    echo "  Log: $log_file"
    
    # Start with environment variables and redirect output
    GRPC_TRACE=all GRPC_VERBOSITY=DEBUG MLX_METAL_DEBUG=ON \
        $command > "$log_file" 2>&1 &
    
    echo "  PID: $!"
    echo ""
}

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "worker_server" || true
pkill -f "working_api_server" || true
sleep 2

# Start workers on remote devices
echo "=== Starting Workers ==="
ssh mini2.local "cd ~/Movies/mlx_grpc_inference && \
    GRPC_TRACE=all GRPC_VERBOSITY=DEBUG MLX_METAL_DEBUG=ON \
    python3 -m src.worker.worker_server --device-id mini2 > logs/worker_mini2_debug.log 2>&1 &"

ssh master.local "cd ~/Movies/mlx_grpc_inference && \
    GRPC_TRACE=all GRPC_VERBOSITY=DEBUG MLX_METAL_DEBUG=ON \
    python3 -m src.worker.worker_server --device-id master > logs/worker_master_debug.log 2>&1 &"

echo "Workers started on mini2 and master"
echo ""

# Wait for workers to initialize
echo "Waiting for workers to initialize..."
sleep 5

# Start coordinator
echo "=== Starting Coordinator ==="
start_service "Coordinator API Server" \
    "uv run python working_api_server_fixed.py" \
    "logs/coordinator_debug.log"

echo "=== All services started with monitoring ==="
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/coordinator_debug.log"
echo "  ssh mini2.local 'tail -f ~/Movies/mlx_grpc_inference/logs/worker_mini2_debug.log'"
echo "  ssh master.local 'tail -f ~/Movies/mlx_grpc_inference/logs/worker_master_debug.log'"
echo ""
echo "To analyze gRPC traffic:"
echo "  grep 'grpc' logs/coordinator_debug.log | less"
echo ""
echo "To stop all services:"
echo "  ./scripts/cluster-manager.sh stop"
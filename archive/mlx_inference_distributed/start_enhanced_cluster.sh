#!/bin/bash
# Start Enhanced Distributed MLX Inference Cluster
# Uses the new model splitting and tensor communication

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="distributed_config.json"
LOG_DIR="logs"

# Device configuration
MINI1_HOST="mini1.local"
MINI2_HOST="mini2.local"
MASTER_HOST="master.local"
MASTER_USER="georgedekker"

echo -e "${GREEN}🚀 Enhanced Distributed MLX Inference Cluster${NC}"
echo "=============================================="

# Create log directory
mkdir -p "$LOG_DIR"

# Clean up existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "distributed_server_enhanced" || true
pkill -f "grpc_tensor_service" || true
pkill -f "distributed_api" || true
pkill -f "grpc_server.py" || true

# Clean up remote hosts
ssh mini2@$MINI2_HOST "pkill -f 'distributed_server_enhanced' || true" 2>/dev/null || true
ssh $MASTER_USER@$MASTER_HOST "pkill -f 'distributed_server_enhanced' || true" 2>/dev/null || true

sleep 2

# Function to start a device
start_device() {
    local device_id=$1
    local hostname=$2
    local user=$3
    local api_port=$4
    local is_local=$5
    
    echo -e "${YELLOW}Starting $device_id on $hostname...${NC}"
    
    if [ "$is_local" = true ]; then
        # Start locally
        DISTRIBUTED_CONFIG=$CONFIG_FILE API_PORT=$api_port \
            uv run python distributed_server_enhanced.py > "$LOG_DIR/${device_id}_enhanced.log" 2>&1 &
        echo $! > "$LOG_DIR/${device_id}_enhanced.pid"
        echo -e "  ${GREEN}✓${NC} Started locally (PID: $(cat $LOG_DIR/${device_id}_enhanced.pid))"
    else
        # Start remotely
        ssh $user@$hostname "cd ~/mlx_inference_distributed && \
            DISTRIBUTED_CONFIG=$CONFIG_FILE API_PORT=$api_port \
            uv run python distributed_server_enhanced.py > logs/${device_id}_enhanced.log 2>&1 & \
            echo \$! > logs/${device_id}_enhanced.pid" 2>/dev/null
        echo -e "  ${GREEN}✓${NC} Started on $hostname"
    fi
}

# Function to check if server is ready
check_server() {
    local hostname=$1
    local port=$2
    local device_id=$3
    local max_attempts=30
    local attempt=1
    
    echo -n "  Waiting for $device_id API"
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://$hostname:$port/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Test mode selection
if [[ "$1" == "--local-test" ]]; then
    echo -e "${YELLOW}Running in LOCAL TEST MODE (all on mini1)${NC}"
    echo
    
    # Start all servers locally with different ports
    start_device "mini1" "localhost" "" 8100 true
    sleep 5
    start_device "mini2" "localhost" "" 8101 true
    sleep 5
    start_device "master" "localhost" "" 8102 true
    
    # Check all servers
    echo
    echo -e "${YELLOW}Checking server health...${NC}"
    check_server "localhost" 8100 "mini1"
    check_server "localhost" 8101 "mini2"
    check_server "localhost" 8102 "master"
    
else
    echo -e "${YELLOW}Running in DISTRIBUTED MODE${NC}"
    echo
    
    # Start coordinator (mini1) locally
    start_device "mini1" "localhost" "" 8100 true
    sleep 5
    
    # Check coordinator is ready
    echo -e "${YELLOW}Checking coordinator...${NC}"
    if ! check_server "localhost" 8100 "mini1"; then
        echo -e "${RED}Coordinator failed to start!${NC}"
        exit 1
    fi
    
    # Start workers on remote hosts
    start_device "mini2" $MINI2_HOST "mini2" 8101 false
    start_device "master" $MASTER_HOST $MASTER_USER 8102 false
    
    sleep 10
    
    # Check workers (optional, as they might not have API endpoints)
    echo
    echo -e "${YELLOW}Checking workers...${NC}"
    check_server $MINI2_HOST 8101 "mini2" || echo "  Note: Worker might not expose API"
    check_server $MASTER_HOST 8102 "master" || echo "  Note: Worker might not expose API"
fi

# Test the cluster
echo
echo -e "${YELLOW}Testing cluster status...${NC}"
CLUSTER_STATUS=$(curl -s "http://localhost:8100/cluster-status" | python -m json.tool 2>/dev/null || echo "Failed")

if [[ "$CLUSTER_STATUS" != "Failed" ]]; then
    echo -e "${GREEN}✅ Cluster status:${NC}"
    echo "$CLUSTER_STATUS" | head -20
else
    echo -e "${RED}❌ Failed to get cluster status${NC}"
fi

# Show device info
echo
echo -e "${YELLOW}Coordinator device info:${NC}"
curl -s "http://localhost:8100/device-info" | python -m json.tool 2>/dev/null || echo "Failed"

echo
echo -e "${GREEN}🎉 Enhanced Distributed MLX Cluster Started!${NC}"
echo "=============================================="
echo
echo "Access points:"
echo "  - API Endpoint: http://localhost:8100/v1/chat/completions"
echo "  - Health Check: http://localhost:8100/health"
echo "  - Cluster Status: http://localhost:8100/cluster-status"
echo "  - GPU Stats: http://localhost:8100/gpu-stats"
echo "  - API Docs: http://localhost:8100/docs"
echo
echo "Logs:"
echo "  - Coordinator: $LOG_DIR/mini1_enhanced.log"
echo "  - Worker 1: $LOG_DIR/mini2_enhanced.log"
echo "  - Worker 2: $LOG_DIR/master_enhanced.log"
echo
echo "To monitor GPU activity:"
echo "  uv run python monitor_gpu_activity.py"
echo
echo "To stop the cluster:"
echo "  pkill -f distributed_server_enhanced"
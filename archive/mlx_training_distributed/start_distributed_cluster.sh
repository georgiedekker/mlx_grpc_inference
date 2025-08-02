#!/bin/bash
# start_distributed_cluster.sh - Start the complete distributed MLX inference cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MINI1_HOST="mini1.local"
MINI2_HOST="mini2.local" 
MASTER_HOST="master.local"

MINI1_USER="mini1"
MINI2_USER="mini2"
MASTER_USER="georgedekker"

PROJECT_DIR="/Users/mini1/Movies/mlx_training_distributed"
LOG_DIR="$PROJECT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${GREEN}🚀 Starting Distributed MLX Inference Cluster${NC}"
echo "================================================"

# Function to check if a host is reachable
check_host() {
    local host=$1
    local user=$2
    echo -n "Checking $host... "
    
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$user@$host" exit 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Function to start a device
start_device() {
    local host=$1
    local user=$2
    local device_name=$3
    local log_file="$LOG_DIR/${device_name}.log"
    
    echo -e "${YELLOW}Starting $device_name on $host...${NC}"
    
    if [ "$host" == "localhost" ] || [ "$host" == "mini1.local" ]; then
        # Start locally
        echo "  Starting local server..."
        cd "$PROJECT_DIR"
        nohup uv run python distributed_server.py > "$log_file" 2>&1 &
        echo $! > "$LOG_DIR/${device_name}.pid"
    else
        # Start remotely
        echo "  Starting remote server on $host..."
        ssh "$user@$host" "cd ~/mlx_distributed && nohup uv run python distributed_server.py > logs/${device_name}.log 2>&1 & echo \$! > logs/${device_name}.pid"
    fi
    
    echo -e "  ${GREEN}✓${NC} Started (log: $log_file)"
}

# Function to check if a server is ready
wait_for_server() {
    local host=$1
    local port=$2
    local device_name=$3
    local max_attempts=30
    local attempt=1
    
    echo -n "  Waiting for $device_name API on $host:$port"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://$host:$port/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e " ${RED}✗${NC} (timeout)"
    return 1
}

# Function to check gRPC health
check_grpc_health() {
    local host=$1
    local port=$2
    local device_name=$3
    
    echo -n "  Checking gRPC health on $host:$port"
    
    # Use grpc_health_probe if available
    if command -v grpc_health_probe > /dev/null; then
        if grpc_health_probe -addr="$host:$port" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
    else
        # Fallback to Python check
        python3 -c "
import grpc
from tensor_service_pb2_grpc import TensorServiceStub
from tensor_service_pb2 import HealthRequest
try:
    channel = grpc.insecure_channel('$host:$port')
    stub = TensorServiceStub(channel)
    response = stub.HealthCheck(HealthRequest(device_id='test'), timeout=5)
    exit(0 if response.healthy else 1)
except:
    exit(1)
" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
    fi
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Main execution
echo -e "\n${YELLOW}1. Checking device connectivity...${NC}"
echo "------------------------------------"

HOSTS_OK=true
check_host "$MINI1_HOST" "$MINI1_USER" || HOSTS_OK=false
check_host "$MINI2_HOST" "$MINI2_USER" || HOSTS_OK=false
check_host "$MASTER_HOST" "$MASTER_USER" || HOSTS_OK=false

if [ "$HOSTS_OK" = false ]; then
    echo -e "\n${RED}❌ Not all hosts are reachable. Please check SSH configuration.${NC}"
    echo "Ensure you can SSH to all devices without password prompts."
    exit 1
fi

echo -e "\n${YELLOW}2. Stopping any existing servers...${NC}"
echo "------------------------------------"

# Kill existing processes
pkill -f "distributed_server.py" || true
ssh "$MINI2_USER@$MINI2_HOST" "pkill -f distributed_server.py || true" 2>/dev/null || true
ssh "$MASTER_USER@$MASTER_HOST" "pkill -f distributed_server.py || true" 2>/dev/null || true

sleep 2

echo -e "\n${YELLOW}3. Starting distributed servers...${NC}"
echo "------------------------------------"

# Start coordinator (mini1)
start_device "localhost" "$MINI1_USER" "mini1"
sleep 5

# Start workers
start_device "$MINI2_HOST" "$MINI2_USER" "mini2"
start_device "$MASTER_HOST" "$MASTER_USER" "master"

echo -e "\n${YELLOW}4. Waiting for services to be ready...${NC}"
echo "------------------------------------"

# Check REST APIs
wait_for_server "$MINI1_HOST" 8100 "mini1"
wait_for_server "$MINI2_HOST" 8101 "mini2"
wait_for_server "$MASTER_HOST" 8102 "master"

echo -e "\n${YELLOW}5. Checking gRPC services...${NC}"
echo "------------------------------------"

sleep 5  # Give gRPC servers time to start

check_grpc_health "$MINI1_HOST" 9100 "mini1"
check_grpc_health "$MINI2_HOST" 9101 "mini2"
check_grpc_health "$MASTER_HOST" 9102 "master"

echo -e "\n${YELLOW}6. Verifying cluster status...${NC}"
echo "------------------------------------"

# Get cluster status from coordinator
echo "Fetching cluster status from coordinator..."
CLUSTER_STATUS=$(curl -s "http://$MINI1_HOST:8100/cluster-status" | python -m json.tool)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Cluster status:"
    echo "$CLUSTER_STATUS" | grep -E "(device_id|status|role)" | head -20
else
    echo -e "${RED}✗${NC} Failed to get cluster status"
fi

echo -e "\n${YELLOW}7. Running health checks...${NC}"
echo "------------------------------------"

# Test inference endpoint
echo "Testing inference endpoint..."
TEST_RESPONSE=$(curl -s -X POST "http://$MINI1_HOST:8100/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mlx-community/Qwen3-1.7B-8bit",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }' | python -m json.tool 2>/dev/null)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Inference endpoint responding"
else
    echo -e "${RED}✗${NC} Inference endpoint not responding"
fi

echo -e "\n${GREEN}🎉 Distributed MLX Inference Cluster Started!${NC}"
echo "================================================"
echo
echo "Access points:"
echo "  - Coordinator API: http://$MINI1_HOST:8100"
echo "  - Worker 1 API: http://$MINI2_HOST:8101"
echo "  - Worker 2 API: http://$MASTER_HOST:8102"
echo "  - Cluster Status: http://$MINI1_HOST:8100/cluster-status"
echo "  - Distributed Stats: http://$MINI1_HOST:8100/distributed-stats"
echo "  - API Documentation: http://$MINI1_HOST:8100/docs"
echo
echo "Logs:"
echo "  - mini1: $LOG_DIR/mini1.log"
echo "  - mini2: $LOG_DIR/mini2.log"
echo "  - master: $LOG_DIR/master.log"
echo
echo "To stop the cluster, run: ./stop_distributed_cluster.sh"
echo

# Optional: Start monitoring
read -p "Start performance monitoring? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Starting performance monitoring...${NC}"
    
    # Start asitop in new terminal windows if available
    if command -v osascript > /dev/null; then
        osascript -e 'tell app "Terminal" to do script "cd '$PROJECT_DIR' && asitop"'
        echo "  - Started asitop for GPU monitoring"
    fi
    
    # Start metrics collector
    nohup python collect_metrics.py > "$LOG_DIR/metrics.log" 2>&1 &
    echo "  - Started metrics collector (Prometheus endpoint: http://localhost:9090)"
fi

echo -e "\n${GREEN}✨ Setup complete!${NC}"
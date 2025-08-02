#!/bin/bash
# stop_distributed_cluster.sh - Stop the distributed MLX inference cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MINI2_HOST="mini2.local"
MASTER_HOST="master.local"
MINI2_USER="mini2"
MASTER_USER="georgedekker"
LOG_DIR="/Users/mini1/Movies/mlx_training_distributed/logs"

echo -e "${YELLOW}🛑 Stopping Distributed MLX Inference Cluster${NC}"
echo "================================================"

# Function to stop a device
stop_device() {
    local host=$1
    local user=$2
    local device_name=$3
    local pid_file="$LOG_DIR/${device_name}.pid"
    
    echo -n "Stopping $device_name on $host... "
    
    if [ "$host" == "localhost" ] || [ "$host" == "mini1.local" ]; then
        # Stop locally
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm -f "$pid_file"
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${YELLOW}(not running)${NC}"
            fi
        else
            # Try pkill as fallback
            if pkill -f "distributed_server.py"; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${YELLOW}(not found)${NC}"
            fi
        fi
    else
        # Stop remotely
        ssh "$user@$host" "pkill -f distributed_server.py || true" 2>/dev/null
        echo -e "${GREEN}✓${NC}"
    fi
}

# Stop all devices
echo -e "\n${YELLOW}Stopping servers...${NC}"
echo "------------------------------------"

stop_device "localhost" "mini1" "mini1"
stop_device "$MINI2_HOST" "$MINI2_USER" "mini2"
stop_device "$MASTER_HOST" "$MASTER_USER" "master"

# Stop metrics collector
echo -n "Stopping metrics collector... "
if pkill -f "collect_metrics.py"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}(not running)${NC}"
fi

# Clean up gRPC processes
echo -n "Cleaning up gRPC processes... "
pkill -f "grpc.*tensor" || true
echo -e "${GREEN}✓${NC}"

echo -e "\n${GREEN}✅ Cluster stopped successfully!${NC}"
echo
echo "Log files preserved in: $LOG_DIR"
echo "To restart the cluster, run: ./start_distributed_cluster.sh"
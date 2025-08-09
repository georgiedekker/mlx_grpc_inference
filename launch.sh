#!/bin/bash

# Enhanced MLX Distributed Inference Launch Script
# Always runs distributed across mini1 and mini2 via Thunderbolt

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen3-1.7B-8bit}"
MEMORY_LIMIT_MB=25600  # 25GB for 32GB M4 Mac minis
PID_FILE="/tmp/mlx_distributed.pid"
LOG_FILE="server.log"

# Parse command line arguments
COMMAND="${1:-start}"  # Default to 'start'

# Function to print header
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}MLX DISTRIBUTED INFERENCE LAUNCHER v2.0${NC}"
    echo -e "${BLUE}Distributed across mini1 + mini2${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Function to configure system
configure_system() {
    # Check if running on macOS and set memory limits
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${GREEN}✓ Running on macOS${NC}"
        
        # Set memory limits on both machines
        echo -e "${YELLOW}Setting GPU memory limit to ${MEMORY_LIMIT_MB}MB on mini1...${NC}"
        sudo sysctl iogpu.wired_limit_mb=$MEMORY_LIMIT_MB 2>/dev/null || {
            echo -e "${YELLOW}Note: Memory limit requires admin privileges${NC}"
        }
        
        echo -e "${YELLOW}Setting GPU memory limit to ${MEMORY_LIMIT_MB}MB on mini2...${NC}"
        ssh 192.168.5.2 "sudo sysctl iogpu.wired_limit_mb=$MEMORY_LIMIT_MB" 2>/dev/null || {
            echo -e "${YELLOW}Note: Could not set memory limit on mini2${NC}"
        }
    fi

    # Network configuration
    echo -e "${YELLOW}Configuring network...${NC}"

    # Auto-detect Thunderbolt interface
    if ifconfig bridge100 &>/dev/null; then
        NETWORK_INTERFACE="bridge100"
        echo -e "${GREEN}✓ Found Thunderbolt interface: bridge100${NC}"
    else
        # Fallback to finding any 192.168.5.x interface
        NETWORK_INTERFACE=$(ifconfig | grep -B4 "192.168.5" | grep "^[a-z]" | cut -d: -f1 | head -1)
        if [ -n "$NETWORK_INTERFACE" ]; then
            echo -e "${GREEN}✓ Found network interface: $NETWORK_INTERFACE${NC}"
        else
            echo -e "${RED}✗ Could not detect Thunderbolt interface${NC}"
            echo -e "${YELLOW}Please ensure Thunderbolt cable is connected${NC}"
            exit 1
        fi
    fi

    # Set MPI network interface
    export OMPI_MCA_btl_tcp_if_include=$NETWORK_INTERFACE
    echo -e "${GREEN}✓ MPI using interface: $NETWORK_INTERFACE${NC}"

    # Export model name
    export MODEL_NAME
}

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to stop server
stop_server() {
    echo -e "${YELLOW}Stopping MLX Distributed Inference...${NC}"
    
    # Kill processes on both machines
    pkill -f server.py 2>/dev/null || true
    ssh 192.168.5.2 "pkill -f server.py" 2>/dev/null || true
    
    # Remove PID file
    rm -f "$PID_FILE"
    
    sleep 2
    echo -e "${GREEN}✓ Server stopped on both mini1 and mini2${NC}"
}

# Function to start server
start_server() {
    if is_running; then
        echo -e "${YELLOW}Server is already running (PID: $(cat $PID_FILE))${NC}"
        echo -e "${YELLOW}Use './launch.sh restart' to restart${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Starting MLX Distributed Inference...${NC}"
    echo ""
    
    # Check connectivity FIRST
    echo -e "${YELLOW}Checking Thunderbolt connectivity to mini2...${NC}"
    if ! ping -c 1 -W 1 192.168.5.2 &>/dev/null; then
        echo -e "${RED}✗ Cannot reach mini2 at 192.168.5.2${NC}"
        echo -e "${YELLOW}Please ensure:${NC}"
        echo -e "  1. Thunderbolt cable is connected"
        echo -e "  2. Network is configured on both machines"
        echo -e "  3. mini2 is powered on"
        exit 1
    fi
    echo -e "${GREEN}✓ mini2 (192.168.5.2) is reachable${NC}"
    
    # Check SSH access
    if ! ssh -o ConnectTimeout=2 192.168.5.2 "echo 'OK'" &>/dev/null; then
        echo -e "${RED}✗ Cannot SSH to mini2${NC}"
        echo -e "${YELLOW}Please setup passwordless SSH:${NC}"
        echo -e "  ssh-copy-id 192.168.5.2"
        exit 1
    fi
    echo -e "${GREEN}✓ SSH access to mini2 confirmed${NC}"
    
    # Configure system
    configure_system
    
    # Clean up any stray processes
    echo -e "${YELLOW}Cleaning up any existing processes...${NC}"
    pkill -f server.py 2>/dev/null || true
    ssh 192.168.5.2 "pkill -f server.py" 2>/dev/null || true
    sleep 2
    
    # Ensure rankfile exists
    if [ ! -f "rankfile" ]; then
        echo -e "${YELLOW}Creating rankfile...${NC}"
        cat > rankfile << 'EOF'
rank 0=mini1.local slot=0
rank 1=mini2.local slot=0
EOF
    fi
    
    # Ensure hosts.json exists
    if [ ! -f "hosts.json" ]; then
        echo -e "${YELLOW}Creating hosts.json...${NC}"
        cat > hosts.json << 'EOF'
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF
    fi
    
    # Copy server to mini2
    echo -e "${YELLOW}Syncing server.py to mini2...${NC}"
    scp -q server.py 192.168.5.2:/Users/mini2/ || {
        echo -e "${RED}✗ Failed to copy server.py to mini2${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Server synced to mini2${NC}"
    
    echo ""
    echo -e "${GREEN}=== DISTRIBUTED CONFIGURATION ===${NC}"
    echo -e "  ${YELLOW}Model:${NC} $MODEL_NAME"
    echo -e "  ${YELLOW}Memory Limit:${NC} ${MEMORY_LIMIT_MB}MB per device"
    echo -e "  ${YELLOW}Network:${NC} Thunderbolt (${NETWORK_INTERFACE})"
    echo -e "  ${YELLOW}mini1:${NC} Rank 0 (layers 0-13) @ 192.168.5.1"
    echo -e "  ${YELLOW}mini2:${NC} Rank 1 (layers 14-27) @ 192.168.5.2"
    echo ""
    echo -e "${BLUE}Starting distributed inference...${NC}"
    echo ""
    
    # Use direct mpirun to have more control over Python paths
    # mlx.launch doesn't handle different Python paths well
    nohup mpirun \
        -n 1 -host localhost /Users/mini1/Movies/mlx_grpc_inference/.venv/bin/python /Users/mini1/Movies/mlx_grpc_inference/server.py : \
        -n 1 -host 192.168.5.2 /Users/mini2/.venv/bin/python /Users/mini2/server.py \
        > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    
    # Wait and check if started successfully
    echo -e "${YELLOW}Waiting for servers to initialize...${NC}"
    sleep 8
    
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo ""
        echo -e "${GREEN}✅ SUCCESS: Distributed inference is running${NC}"
        echo -e "${GREEN}✅ Process ID: $PID${NC}"
        echo -e "${GREEN}✅ API endpoint: http://localhost:8100${NC}"
        echo -e "${GREEN}✅ Dashboard: http://localhost:8100${NC}"
        echo ""
        
        # Check if API is responding
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is responding${NC}"
            
            # Show GPU usage
            HEALTH=$(curl -s http://localhost:8100/health)
            if [ -n "$HEALTH" ]; then
                echo ""
                echo -e "${BLUE}System Status:${NC}"
                echo "$HEALTH" | jq -r '
                    "  GPUs: \(.world_size)",
                    "  Model: \(.model)",
                    "  GPU Memory: \(.memory.gpu_gb) GB",
                    "  Pipeline: \(.pipeline)"
                ' 2>/dev/null || echo "$HEALTH"
            fi
        else
            echo -e "${YELLOW}⏳ API is still starting up...${NC}"
            echo -e "${YELLOW}   Check status with: ./launch.sh status${NC}"
        fi
        
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo -e "  ${GREEN}./launch.sh status${NC}  - Check server status"
        echo -e "  ${GREEN}./launch.sh logs${NC}    - View live logs"
        echo -e "  ${GREEN}./launch.sh restart${NC} - Restart servers"
        echo -e "  ${GREEN}./launch.sh stop${NC}    - Stop servers"
        echo ""
    else
        echo -e "${RED}✗ Failed to start distributed inference${NC}"
        echo -e "${YELLOW}Checking logs for errors...${NC}"
        echo ""
        tail -20 "$LOG_FILE"
        echo ""
        echo -e "${YELLOW}Full logs: tail -f $LOG_FILE${NC}"
        return 1
    fi
}

# Function to show status
show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}✅ Distributed inference is running (PID: $PID)${NC}"
        echo ""
        
        # Check mini1 and mini2 processes
        echo -e "${BLUE}Process Status:${NC}"
        echo -e "  ${YELLOW}mini1:${NC}"
        ps aux | grep -E "rank.*0.*server.py" | grep -v grep | head -1 || echo "    Checking..."
        echo -e "  ${YELLOW}mini2:${NC}"
        ssh 192.168.5.2 "ps aux | grep server.py | grep -v grep | head -1" 2>/dev/null || echo "    Cannot check mini2"
        echo ""
        
        # Try to get health status
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            echo -e "${GREEN}API Health Check:${NC}"
            curl -s http://localhost:8100/health | jq '.' 2>/dev/null || curl -s http://localhost:8100/health
            echo ""
        else
            echo -e "${YELLOW}⚠ API not responding${NC}"
            echo -e "  The server may still be initializing..."
            echo ""
        fi
        
        # Show recent logs
        echo -e "${BLUE}Recent logs:${NC}"
        tail -10 "$LOG_FILE" | head -5
    else
        echo -e "${YELLOW}❌ Distributed inference is not running${NC}"
        echo ""
        echo -e "Start with: ${GREEN}./launch.sh start${NC}"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Following logs (Ctrl+C to exit):${NC}"
        echo ""
        tail -f "$LOG_FILE"
    else
        echo -e "${YELLOW}No log file found${NC}"
        echo -e "Start the server first: ${GREEN}./launch.sh start${NC}"
    fi
}

# Main command processing
print_header

case "$COMMAND" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        echo -e "${YELLOW}Restarting distributed inference...${NC}"
        stop_server
        sleep 2
        start_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo -e "${YELLOW}Usage:${NC}"
        echo -e "  ${GREEN}./launch.sh [command]${NC}"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo -e "  ${GREEN}start${NC}   - Start distributed inference (default)"
        echo -e "  ${GREEN}stop${NC}    - Stop distributed inference"
        echo -e "  ${GREEN}restart${NC} - Restart distributed inference"
        echo -e "  ${GREEN}status${NC}  - Show server status"
        echo -e "  ${GREEN}logs${NC}    - Show live logs"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  ${GREEN}./launch.sh${NC}          # Start distributed inference"
        echo -e "  ${GREEN}./launch.sh start${NC}    # Start distributed inference"
        echo -e "  ${GREEN}./launch.sh status${NC}   # Check if running"
        echo -e "  ${GREEN}./launch.sh restart${NC}  # Restart servers"
        echo -e "  ${GREEN}./launch.sh stop${NC}     # Stop servers"
        echo ""
        echo -e "${BLUE}This will run MLX inference distributed across:${NC}"
        echo -e "  • mini1 (192.168.5.1) - Rank 0, layers 0-13"
        echo -e "  • mini2 (192.168.5.2) - Rank 1, layers 14-27"
        exit 1
        ;;
esac
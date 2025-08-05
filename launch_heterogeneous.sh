#!/bin/bash
# Launch script for heterogeneous PyTorch distributed inference
# Supports 2 or 3 device configurations with capability-based sharding

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config/heterogeneous_cluster.json}"
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen3-1.7B-8bit}"
API_PORT="${API_PORT:-8000}"

# Device configuration - automatically detect from config file
if [ -f "$CONFIG_FILE" ]; then
    # Extract devices from config file using python
    DEVICES=($(uv run python -c "import json; c=json.load(open('$CONFIG_FILE')); print(' '.join(c['cluster']['devices']))"))
    MASTER_ADDR=$(uv run python -c "import json; c=json.load(open('$CONFIG_FILE')); print(c['communication']['master_addr'])")
    MASTER_PORT=$(uv run python -c "import json; c=json.load(open('$CONFIG_FILE')); print(c['communication']['master_port'])")
else
    # Default configuration
    DEVICES=("mini1" "mini2")
    MASTER_ADDR="10.0.0.80"
    MASTER_PORT="29501"
fi

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to check if process is running
is_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to stop services
stop_services() {
    print_status "Stopping heterogeneous distributed services..."
    
    # Stop API server
    if [ -f "$LOG_DIR/api_server.pid" ]; then
        local api_pid=$(cat "$LOG_DIR/api_server.pid")
        if ps -p "$api_pid" > /dev/null 2>&1; then
            print_status "Stopping API server (PID: $api_pid)"
            kill "$api_pid" 2>/dev/null || true
            sleep 2
        fi
        rm -f "$LOG_DIR/api_server.pid"
    fi
    
    # Stop distributed workers
    for device in "${DEVICES[@]}"; do
        local pid_file="$LOG_DIR/${device}_worker.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if [ "$device" == "mini1" ]; then
                # Local process
                if ps -p "$pid" > /dev/null 2>&1; then
                    print_status "Stopping local worker $device (PID: $pid)"
                    kill "$pid" 2>/dev/null || true
                fi
            else
                # Remote process
                print_status "Stopping remote worker on $device"
                ssh "$device" "kill $pid 2>/dev/null || true" || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    # Clean up any remaining Python processes
    pkill -f "pytorch_heterogeneous_server.py" 2>/dev/null || true
    
    print_status "All services stopped"
}

# Function to launch worker on device
launch_worker() {
    local rank=$1
    local device=$2
    local world_size=${#DEVICES[@]}
    
    if [ "$device" == "mini1" ]; then
        # Launch locally
        print_status "Launching local worker (rank $rank) on $device"
        
        PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/pytorch_heterogeneous_server.py" \
            --rank "$rank" \
            --world-size "$world_size" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --model-name "$MODEL_NAME" \
            --config "$CONFIG_FILE" \
            > "$LOG_DIR/${device}_worker.log" 2>&1 &
        
        local pid=$!
        echo "$pid" > "$LOG_DIR/${device}_worker.pid"
        print_status "Local worker started with PID $pid"
        
    else
        # Launch remotely via SSH
        print_status "Launching remote worker (rank $rank) on $device"
        
        # Copy necessary files if needed
        print_status "Syncing code to $device..."
        rsync -az --exclude='logs' --exclude='__pycache__' --exclude='*.pyc' \
            "$SCRIPT_DIR"/ "$device:$SCRIPT_DIR/" || {
            print_error "Failed to sync code to $device"
            return 1
        }
        
        # Launch remote worker
        ssh "$device" "cd $SCRIPT_DIR && \
            PYTHONUNBUFFERED=1 uv run python pytorch_heterogeneous_server.py \
            --rank $rank \
            --world-size $world_size \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT \
            --model-name '$MODEL_NAME' \
            --config '$CONFIG_FILE' \
            > '$LOG_DIR/${device}_worker.log' 2>&1 & \
            echo \$!" > "$LOG_DIR/${device}_worker.pid"
        
        local pid=$(cat "$LOG_DIR/${device}_worker.pid")
        print_status "Remote worker started on $device with PID $pid"
    fi
}

# Function to verify worker is ready
verify_worker() {
    local device=$1
    local log_file="$LOG_DIR/${device}_worker.log"
    local max_wait=60
    local waited=0
    
    print_status "Waiting for $device to be ready..."
    
    while [ $waited -lt $max_wait ]; do
        if [ -f "$log_file" ] && grep -q "ready for inference" "$log_file"; then
            print_status "$device is ready"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    
    print_error "$device failed to start within $max_wait seconds"
    return 1
}

# Function to launch API server
launch_api_server() {
    print_status "Launching API server on port $API_PORT"
    
    PYTHONUNBUFFERED=1 MODEL_NAME="$MODEL_NAME" \
        CONFIG_FILE="$CONFIG_FILE" \
        uv run python "$SCRIPT_DIR/pytorch_heterogeneous_api.py" \
        --port "$API_PORT" \
        > "$LOG_DIR/api_server.log" 2>&1 &
    
    local pid=$!
    echo "$pid" > "$LOG_DIR/api_server.pid"
    
    # Wait for API server to start
    local max_wait=30
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            print_status "API server is ready at http://localhost:$API_PORT"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    
    print_error "API server failed to start"
    return 1
}

# Function to show cluster status
show_status() {
    print_status "Heterogeneous Cluster Status:"
    echo "================================="
    
    # Check workers
    for i in "${!DEVICES[@]}"; do
        local device="${DEVICES[$i]}"
        local pid_file="$LOG_DIR/${device}_worker.pid"
        
        if is_running "$pid_file"; then
            local pid=$(cat "$pid_file")
            echo -e "Worker $device (rank $i): ${GREEN}Running${NC} (PID: $pid)"
            
            # Show assignment from log
            if [ -f "$LOG_DIR/${device}_worker.log" ]; then
                local assignment=$(grep "assignment:" "$LOG_DIR/${device}_worker.log" | tail -1)
                if [ -n "$assignment" ]; then
                    echo "  $assignment"
                fi
            fi
        else
            echo -e "Worker $device (rank $i): ${RED}Stopped${NC}"
        fi
    done
    
    # Check API server
    if is_running "$LOG_DIR/api_server.pid"; then
        local pid=$(cat "$LOG_DIR/api_server.pid")
        echo -e "API Server: ${GREEN}Running${NC} (PID: $pid) at http://localhost:$API_PORT"
    else
        echo -e "API Server: ${RED}Stopped${NC}"
    fi
    
    echo "================================="
}

# Function to test the cluster
test_cluster() {
    print_status "Testing heterogeneous cluster..."
    
    # Test with a simple prompt
    local response=$(curl -s -X POST "http://localhost:$API_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.7
        }')
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        print_status "Test successful!"
        echo "Response: $response"
    else
        print_error "Test failed"
        return 1
    fi
}

# Main execution
case "${1:-start}" in
    start)
        print_status "Starting heterogeneous distributed inference cluster"
        print_status "Configuration: ${#DEVICES[@]} devices, model: $MODEL_NAME"
        print_status "Using sharding strategy from: $CONFIG_FILE"
        
        # Stop any existing services
        stop_services
        
        # Launch workers in order (rank = array index)
        for i in "${!DEVICES[@]}"; do
            launch_worker "$i" "${DEVICES[$i]}" || {
                print_error "Failed to launch worker on ${DEVICES[$i]}"
                stop_services
                exit 1
            }
            
            # Give first worker time to initialize
            if [ $i -eq 0 ]; then
                sleep 5
            else
                sleep 2
            fi
        done
        
        # Verify all workers are ready
        for device in "${DEVICES[@]}"; do
            verify_worker "$device" || {
                print_error "Worker verification failed"
                stop_services
                exit 1
            }
        done
        
        # Launch API server
        launch_api_server || {
            print_error "Failed to launch API server"
            stop_services
            exit 1
        }
        
        print_status "Heterogeneous cluster started successfully!"
        show_status
        
        # Run test if requested
        if [ "${RUN_TEST:-false}" == "true" ]; then
            sleep 2
            test_cluster
        fi
        ;;
        
    stop)
        stop_services
        ;;
        
    restart)
        stop_services
        sleep 2
        exec "$0" start
        ;;
        
    status)
        show_status
        ;;
        
    test)
        test_cluster
        ;;
        
    logs)
        device="${2:-mini1}"
        if [ -f "$LOG_DIR/${device}_worker.log" ]; then
            tail -f "$LOG_DIR/${device}_worker.log"
        else
            print_error "No log file found for $device"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|test|logs [device]}"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_NAME    - Model to load (default: mlx-community/Qwen3-1.7B-8bit)"
        echo "  CONFIG_FILE   - Cluster configuration file (default: config/heterogeneous_cluster.json)"
        echo "  API_PORT      - API server port (default: 8000)"
        echo "  RUN_TEST      - Run test after start (default: false)"
        echo ""
        echo "Example:"
        echo "  MODEL_NAME='microsoft/phi-2' $0 start"
        echo "  $0 logs master  # View logs from master device"
        exit 1
        ;;
esac
#!/bin/bash
# Master control script for distributed MLX inference
# Controls all devices from mini1

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/distributed_config.json"
LOG_DIR="${SCRIPT_DIR}/logs"
VENV_PATH="${SCRIPT_DIR}/.venv"

# Device configurations from distributed_config.json
# mini1: 16GB RAM, 10 GPU cores (master)
# mini2: 16GB RAM, 10 GPU cores (worker)
# master.local: 48GB RAM, 16 GPU cores (worker)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

# Function to check if a host is reachable
check_host() {
    local host=$1
    
    # Special case for master.local - use SSH since ping doesn't work
    if [ "$host" = "master.local" ]; then
        if ssh -o ConnectTimeout=3 -o BatchMode=yes georgedekker@master.local "echo 'ok'" >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        if ping -c 1 -W 2 $host >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
}

# Function to check if a port is open
check_port() {
    local host=$1
    local port=$2
    nc -z -w 2 $host $port >/dev/null 2>&1
}

# Function to get device info
get_device_info() {
    local host=$1
    local user=$2
    local ssh_prefix=""
    
    if [ "$host" != "mini1.local" ]; then
        if [ -n "$user" ]; then
            ssh_prefix="ssh $user@$host"
        else
            ssh_prefix="ssh $host"
        fi
    fi
    
    # Get available memory
    if [ -n "$ssh_prefix" ]; then
        $ssh_prefix "sysctl -n hw.memsize" 2>/dev/null || echo "0"
    else
        sysctl -n hw.memsize 2>/dev/null || echo "0"
    fi
}

# Function to start a worker on a remote device
start_worker() {
    local host=$1
    local rank=$2
    local port=$3
    local user=$4
    
    print_status "Starting worker on $host (rank $rank, port $port)..."
    
    local ssh_cmd="ssh"
    if [ -n "$user" ]; then
        ssh_cmd="ssh $user@$host"
    else
        ssh_cmd="ssh $host"
    fi
    
    # Kill any existing worker
    $ssh_cmd "pkill -f 'python.*worker.py.*rank=$rank' || true" 2>/dev/null
    
    # Determine the correct path based on the host
    local work_dir="Movies/mlx_inference_distributed"
    if [ "$host" = "master.local" ]; then
        work_dir="~/Movies/mlx_inference_distributed"
    fi
    
    # Start the worker
    $ssh_cmd "cd $work_dir && source .venv/bin/activate && \
        GRPC_DNS_RESOLVER=native \
        LOCAL_RANK=$rank \
        DISTRIBUTED_CONFIG=distributed_config.json \
        nohup python worker.py --rank=$rank > logs/worker_rank${rank}.log 2>&1 &" || {
        print_error "Failed to start worker on $host"
        return 1
    }
    
    # Wait and verify
    sleep 3
    if check_port $host $port; then
        print_status "✓ Worker on $host is listening on port $port"
        return 0
    else
        print_warning "Worker on $host may not be ready yet"
        return 1
    fi
}

# Function to stop all processes
stop_all() {
    print_status "Stopping all distributed MLX processes..."
    
    # Stop local processes
    pkill -f "python.*distributed_api.py" || true
    pkill -f "python.*worker.py" || true
    
    # Stop remote processes
    ssh mini2.local "pkill -f 'python.*worker.py'" 2>/dev/null || true
    ssh georgedekker@master.local "pkill -f 'python.*worker.py'" 2>/dev/null || true
    
    print_status "All processes stopped"
}

# Function to check system status
check_status() {
    print_status "Checking distributed MLX system status..."
    
    # Check if API server is running
    if check_port localhost 8100; then
        echo -e "  ${GREEN}✓${NC} API server is running on port 8100"
        
        # Try to get health status
        health=$(curl -s http://localhost:8100/health 2>/dev/null || echo "{}")
        if [ -n "$health" ] && [ "$health" != "{}" ]; then
            echo -e "  ${GREEN}✓${NC} API server is healthy"
        fi
    else
        echo -e "  ${RED}✗${NC} API server is not running"
    fi
    
    # Check workers
    if check_port mini2.local 50101; then
        echo -e "  ${GREEN}✓${NC} Worker on mini2.local (rank 1) is running"
    else
        echo -e "  ${RED}✗${NC} Worker on mini2.local is not running"
    fi
    
    if check_port master.local 50102; then
        echo -e "  ${GREEN}✓${NC} Worker on master.local (rank 2) is running"
    else
        echo -e "  ${RED}✗${NC} Worker on master.local is not running"
    fi
    
    # Show GPU info if available
    if check_port localhost 8100; then
        echo ""
        print_status "Fetching GPU information..."
        curl -s http://localhost:8100/distributed/gpu-info 2>/dev/null | python -m json.tool 2>/dev/null || echo "  Unable to fetch GPU info"
    fi
}

# Function to start the distributed system
start_all() {
    print_status "Starting distributed MLX inference system..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Check if all hosts are reachable
    print_status "Checking device connectivity..."
    
    local available_devices=()
    
    if check_host mini2.local; then
        available_devices+=("mini2")
        print_status "✓ mini2.local is reachable"
    else
        print_warning "✗ mini2.local is not reachable"
    fi
    
    if check_host master.local; then
        available_devices+=("master")
        print_status "✓ master.local is reachable"
    else
        print_warning "✗ master.local is not reachable"
    fi
    
    if [ ${#available_devices[@]} -eq 0 ]; then
        print_error "No worker devices are reachable. At least one worker is required."
        return 1
    fi
    
    print_status "Found ${#available_devices[@]} available device(s)"
    
    # Select appropriate config file based on available devices
    if [ ${#available_devices[@]} -eq 2 ]; then
        CONFIG_FILE="${SCRIPT_DIR}/distributed_config.json"  # 3-device config
    elif [ ${#available_devices[@]} -eq 1 ]; then
        CONFIG_FILE="${SCRIPT_DIR}/distributed_config_2device.json"  # 2-device config
    fi
    print_status "Using config: $(basename $CONFIG_FILE)"
    
    # Show device capacities from config
    print_status "Device capacities (from config):"
    echo "  - mini1.local: 16GB RAM, 10 GPU cores (master)"
    echo "  - mini2.local: 16GB RAM, 10 GPU cores (worker)" 
    echo "  - master.local: 48GB RAM, 16 GPU cores (worker)"
    echo ""
    print_status "Model sharding will be distributed based on these capacities"
    
    # Start workers first (only available ones)
    if [[ " ${available_devices[@]} " =~ " mini2 " ]]; then
        start_worker "mini2.local" 1 50101 ""
    fi
    
    if [[ " ${available_devices[@]} " =~ " master " ]]; then
        start_worker "master.local" 2 50102 "georgedekker"
    fi
    
    # Wait for workers to be fully initialized
    print_status "Waiting for workers to initialize..."
    sleep 5
    
    # Start master API server
    print_status "Starting master API server on mini1..."
    cd "$SCRIPT_DIR"
    source "$VENV_PATH/bin/activate"
    
    GRPC_DNS_RESOLVER=native \
    LOCAL_RANK=0 \
    DISTRIBUTED_CONFIG="$CONFIG_FILE" \
    API_PORT=8100 \
    nohup python distributed_api.py > "$LOG_DIR/api_server.log" 2>&1 &
    
    # Wait for API server
    print_status "Waiting for API server to start..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if check_port localhost 8100; then
            print_status "✓ API server is ready"
            break
        fi
        sleep 1
        retries=$((retries - 1))
    done
    
    if [ $retries -eq 0 ]; then
        print_error "API server failed to start"
        return 1
    fi
    
    # Final status check
    sleep 2
    check_status
    
    echo ""
    print_status "Distributed MLX inference system is ready!"
    echo ""
    echo "Test with:"
    echo '  curl -X POST http://localhost:8100/v1/chat/completions \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"model": "mlx-community/Qwen3-1.7B-8bit", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'"'"
    echo ""
    echo "Monitor logs:"
    echo "  - API server: tail -f $LOG_DIR/api_server.log"
    echo "  - Worker mini2: ssh mini2.local 'tail -f Movies/mlx_inference_distributed/logs/worker_rank1.log'"
    echo "  - Worker master: ssh georgedekker@master.local 'tail -f ~/Movies/mlx_inference_distributed/logs/worker_rank2.log'"
}

# Function to test inference
test_inference() {
    print_status "Testing distributed inference..."
    
    if ! check_port localhost 8100; then
        print_error "API server is not running"
        return 1
    fi
    
    # Simple test
    echo ""
    print_status "Running inference test..."
    curl -X POST http://localhost:8100/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [{"role": "user", "content": "What is 2+2? Please answer with just the number."}],
            "max_tokens": 10,
            "temperature": 0.1
        }' 2>/dev/null | python -m json.tool || print_error "Inference test failed"
}

# Function to show logs
show_logs() {
    local device=$1
    
    case $device in
        "api"|"master"|"mini1")
            tail -f "$LOG_DIR/api_server.log"
            ;;
        "mini2"|"worker1")
            ssh mini2.local "tail -f Movies/mlx_inference_distributed/logs/worker_rank1.log"
            ;;
        "master.local"|"worker2")
            ssh georgedekker@master.local "tail -f ~/Movies/mlx_inference_distributed/logs/worker_rank2.log"
            ;;
        "all")
            # Show all logs in parallel
            echo "Showing all logs (Ctrl+C to stop)..."
            tail -f "$LOG_DIR/api_server.log" | sed 's/^/[mini1] /' &
            ssh mini2.local "tail -f Movies/mlx_inference_distributed/logs/worker_rank1.log" | sed 's/^/[mini2] /' &
            ssh georgedekker@master.local "tail -f ~/Movies/mlx_inference_distributed/logs/worker_rank2.log" | sed 's/^/[master] /' &
            wait
            ;;
        *)
            print_error "Unknown device: $device"
            echo "Usage: $0 logs [api|mini1|mini2|master.local|all]"
            ;;
    esac
}

# Main command handling
case "$1" in
    "start")
        start_all
        ;;
    "stop")
        stop_all
        ;;
    "restart")
        stop_all
        sleep 2
        start_all
        ;;
    "status")
        check_status
        ;;
    "test")
        test_inference
        ;;
    "logs")
        show_logs "$2"
        ;;
    *)
        echo "Distributed MLX Inference Control Script"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|test|logs}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all distributed components"
        echo "  stop     - Stop all distributed components"
        echo "  restart  - Restart all distributed components"
        echo "  status   - Check system status"
        echo "  test     - Run a simple inference test"
        echo "  logs [device] - Show logs (device: api|mini1|mini2|master.local|all)"
        echo ""
        echo "This script manages the distributed MLX inference system across:"
        echo "  - mini1.local (16GB, 10 GPU cores) - Master"
        echo "  - mini2.local (16GB, 10 GPU cores) - Worker"
        echo "  - master.local (48GB, 16 GPU cores) - Worker"
        exit 1
        ;;
esac
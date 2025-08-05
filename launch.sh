#!/bin/bash
# Launcher for distributed inference using file-based coordination
# Manages mini1 (coordinator) and mini2 (worker) over Thunderbolt Bridge
# Works around Gloo backend issues on macOS

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen3-1.7B-8bit}"
CONFIG_PATH="$SCRIPT_DIR/config/cluster_config.yaml"
COORD_DIR="/tmp/mlx_coordination"
MINI2_DIR="/Users/mini2/Movies/mlx_grpc_inference"
# No backend selection needed - using simplified server.py

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"
}

# Function to check if a process is running
check_process() {
    if ps -p "$1" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Stop all services
stop() {
    print_status "Stopping all services..."
    
    # Kill local processes
    pkill -f "python.*server.py" 2>/dev/null || true
    pkill -f "python.*worker.py" 2>/dev/null || true
    
    # Kill processes on mini2
    ssh mini2@mini2.local "pkill -f 'python.*server.py' 2>/dev/null || true"
    ssh mini2@mini2.local "pkill -f 'python.*worker.py' 2>/dev/null || true"
    
    # Clean up any stale port bindings
    lsof -ti:8100 | xargs kill -9 2>/dev/null || true
    lsof -ti:8101 | xargs kill -9 2>/dev/null || true
    lsof -ti:50051 | xargs kill -9 2>/dev/null || true
    lsof -ti:50052 | xargs kill -9 2>/dev/null || true
    
    # Clean up coordination directory and FileStore
    rm -rf "$COORD_DIR"
    ssh mini2@mini2.local "rm -rf $COORD_DIR"
    # Clean up any stale TCPStore processes
    lsof -ti:29501 | xargs kill -9 2>/dev/null || true
    
    print_status "All services stopped"
}

# Start distributed inference
start() {
    print_status "Starting TRUE DISTRIBUTED inference (layers split across devices)"
    print_status "Model: $MODEL_NAME"
    print_status "Coordinator: mini1 (192.168.5.1)"
    print_status "Worker: mini2 (192.168.5.2)"
    
    # Stop any existing services
    stop
    
    # Create coordination directory
    mkdir -p "$COORD_DIR"
    
    # Sync files to mini2
    print_status "Syncing files to mini2..."
    ssh mini2@mini2.local "mkdir -p $MINI2_DIR/logs $MINI2_DIR/src $MINI2_DIR/config"
    
    # Sync all necessary files
    rsync -az --exclude='logs' --exclude='__pycache__' --exclude='.venv' \
        "$SCRIPT_DIR"/{server.py,server_distributed.py,server_single.py,worker.py,pyproject.toml,uv.lock} \
        mini2@mini2.local:"$MINI2_DIR/"
    
    # Sync src directory
    rsync -az --exclude='__pycache__' \
        "$SCRIPT_DIR/src/" \
        mini2@mini2.local:"$MINI2_DIR/src/"
    
    # Sync config directory
    rsync -az "$SCRIPT_DIR/config/" mini2@mini2.local:"$MINI2_DIR/config/"
    
    # Start rank 1 on mini2
    print_status "Starting worker on mini2 (rank 1)..."
    ssh mini2@mini2.local "cd $MINI2_DIR && \
        PYTHONUNBUFFERED=1 \
        DISTRIBUTED=true \
        RANK=1 \
        WORLD_SIZE=2 \
        MASTER_ADDR=192.168.5.1 \
        MASTER_PORT=29501 \
        MODEL_NAME='$MODEL_NAME' \
        /Users/mini2/.local/bin/uv run python worker.py \
        > logs/worker.log 2>&1 & \
        echo \$!" > "$LOG_DIR/mini2_pid"
    
    MINI2_PID=$(cat "$LOG_DIR/mini2_pid")
    print_status "Worker started on mini2 (PID: $MINI2_PID)"
    
    # Give worker time to start
    sleep 3
    
    # Start rank 0 on mini1
    print_status "Starting coordinator on mini1 (rank 0)..."
    PYTHONUNBUFFERED=1 \
    DISTRIBUTED=true \
    RANK=0 \
    WORLD_SIZE=2 \
    MASTER_ADDR=192.168.5.1 \
    MASTER_PORT=29501 \
    MODEL_NAME="$MODEL_NAME" \
    uv run python "$SCRIPT_DIR/server.py" \
        > "$LOG_DIR/coordinator.log" 2>&1 &
    
    MINI1_PID=$!
    echo $MINI1_PID > "$LOG_DIR/mini1_pid"
    print_status "Coordinator started on mini1 (PID: $MINI1_PID)"
    
    # Wait for services to initialize
    print_status "Waiting for services to initialize..."
    sleep 10
    
    # Check if services are running
    if check_process $MINI1_PID; then
        print_status "✓ Coordinator is running"
    else
        print_error "✗ Coordinator failed to start"
        tail -20 "$LOG_DIR/coordinator.log"
        stop
        return 1
    fi
    
    # Check mini2
    if ssh mini2@mini2.local "ps -p $MINI2_PID > /dev/null 2>&1"; then
        print_status "✓ Worker is running"
    else
        print_error "✗ Worker failed to start"
        ssh mini2@mini2.local "tail -20 $MINI2_DIR/logs/worker.log"
        stop
        return 1
    fi
    
    # Test API
    print_status "Testing API endpoint..."
    if curl -s http://localhost:8100/health > /dev/null; then
        print_status "✓ API server is ready"
        print_status "Health check:"
        curl -s http://localhost:8100/health | python -m json.tool
    else
        print_error "✗ API server not responding"
        print_status "Check logs with: $0 logs"
    fi
    
    print_status "Distributed inference is running!"
    print_status "API: http://localhost:8100"
    print_status "Logs: $0 logs [coordinator|worker]"
}

# Show logs
logs() {
    case "${1:-coordinator}" in
        coordinator|mini1)
            tail -f "$LOG_DIR/coordinator.log"
            ;;
        worker|mini2)
            ssh mini2@mini2.local "tail -f $MINI2_DIR/logs/worker.log"
            ;;
        *)
            echo "Usage: $0 logs [coordinator|worker]"
            ;;
    esac
}

# Show status
status() {
    echo "=== Distributed Inference Status ==="
    
    # Check coordinator
    if [ -f "$LOG_DIR/mini1_pid" ]; then
        PID=$(cat "$LOG_DIR/mini1_pid")
        if check_process $PID; then
            echo "Coordinator (mini1): Running (PID: $PID)"
        else
            echo "Coordinator (mini1): Stopped"
        fi
    else
        echo "Coordinator (mini1): Not started"
    fi
    
    # Check worker
    if [ -f "$LOG_DIR/mini2_pid" ]; then
        PID=$(cat "$LOG_DIR/mini2_pid")
        if ssh mini2@mini2.local "ps -p $PID > /dev/null 2>&1"; then
            echo "Worker (mini2): Running (PID: $PID)"
        else
            echo "Worker (mini2): Stopped"
        fi
    else
        echo "Worker (mini2): Not started"
    fi
    
    # Check API
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "API Server: Running on port 8100"
    else
        echo "API Server: Not responding"
    fi
}

# Test inference
test() {
    print_status "Testing inference..."
    
    curl -X POST http://localhost:8100/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.7
        }' | python -m json.tool
}

# Main command handler
case "${1:-help}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    test)
        test
        ;;
    help|*)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start    - Start distributed inference"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - Show logs (coordinator|worker)"
        echo "  test     - Test inference API"
        echo ""
        echo "Environment:"
        echo "  MODEL_NAME - Model to use (default: mlx-community/Qwen3-1.7B-8bit)"
        ;;
esac
#!/bin/bash
# Single-machine launcher for MLX inference
# Fallback option if distributed doesn't work

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen3-1.7B-8bit}"

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

# Stop service
stop() {
    print_status "Stopping single-machine server..."
    pkill -f "python.*server.py" 2>/dev/null || true
    lsof -ti:8100 | xargs kill -9 2>/dev/null || true
    rm -f /tmp/mlx_distributed_store
    print_status "Server stopped"
}

# Start single-machine inference
start() {
    print_status "Starting single-machine inference"
    print_status "Model: $MODEL_NAME"
    
    # Stop any existing services
    stop
    
    # Start server in single-machine mode
    print_status "Starting server..."
    PYTHONUNBUFFERED=1 \
    RANK=0 \
    WORLD_SIZE=1 \
    MODEL_NAME="$MODEL_NAME" \
    uv run python "$SCRIPT_DIR/server.py" \
        > "$LOG_DIR/single_machine.log" 2>&1 &
    
    SERVER_PID=$!
    echo $SERVER_PID > "$LOG_DIR/server_pid"
    print_status "Server started (PID: $SERVER_PID)"
    
    # Wait for service to initialize
    print_status "Waiting for service to initialize..."
    sleep 10
    
    # Check if service is running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        print_status "✓ Server is running"
    else
        print_error "✗ Server failed to start"
        tail -20 "$LOG_DIR/single_machine.log"
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
    fi
    
    print_status "Single-machine inference is running!"
    print_status "API: http://localhost:8100"
    print_status "Logs: tail -f $LOG_DIR/single_machine.log"
}

# Show logs
logs() {
    tail -f "$LOG_DIR/single_machine.log"
}

# Show status
status() {
    echo "=== Single-Machine Inference Status ==="
    
    if [ -f "$LOG_DIR/server_pid" ]; then
        PID=$(cat "$LOG_DIR/server_pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Server: Running (PID: $PID)"
        else
            echo "Server: Stopped"
        fi
    else
        echo "Server: Not started"
    fi
    
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "API: Running on port 8100"
    else
        echo "API: Not responding"
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
        logs
        ;;
    test)
        test
        ;;
    help|*)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Single-machine MLX inference launcher"
        echo "Use this if distributed setup is not working"
        ;;
esac
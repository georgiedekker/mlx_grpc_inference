#!/bin/bash
# Launch distributed MLX inference with performance monitoring dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Kill any existing processes
cleanup() {
    print_message "Stopping all processes..."
    
    # Kill dashboard
    pkill -f "start_dashboard.py" 2>/dev/null || true
    
    # Kill server and workers
    pkill -f "python server.py" 2>/dev/null || true
    pkill -f "python worker.py" 2>/dev/null || true
    
    # Kill remote workers
    ssh mini2.local "pkill -f 'python worker.py'" 2>/dev/null || true
    
    sleep 2
    print_message "All processes stopped"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Start monitoring dashboard
start_dashboard() {
    print_info "Starting performance monitoring dashboard..."
    python start_dashboard.py > dashboard.log 2>&1 &
    local dashboard_pid=$!
    
    sleep 2
    
    if ps -p $dashboard_pid > /dev/null; then
        print_message "✅ Dashboard started (PID: $dashboard_pid)"
        print_info "📊 Dashboard available at: http://localhost:8888"
        return 0
    else
        print_error "Failed to start dashboard"
        return 1
    fi
}

# Start server
start_server() {
    print_info "Starting distributed inference server..."
    python server.py > server_monitored.log 2>&1 &
    local server_pid=$!
    
    sleep 3
    
    if ps -p $server_pid > /dev/null; then
        print_message "✅ Server started (PID: $server_pid)"
        print_info "🚀 API available at: http://localhost:8000"
        return 0
    else
        print_error "Failed to start server"
        cat server_monitored.log | tail -20
        return 1
    fi
}

# Start remote worker
start_remote_worker() {
    print_info "Starting worker on mini2..."
    
    # Sync files first
    print_message "Syncing files to mini2..."
    rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
        --exclude='*.pyc' --exclude='models' \
        /Users/mini1/Movies/mlx_grpc_inference/ \
        mini2.local:~/Movies/mlx_grpc_inference/ > /dev/null 2>&1
    
    # Generate protos on mini2
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && ./protos/generate_protos.sh" > /dev/null 2>&1
    
    # Start worker
    ssh mini2.local "cd ~/Movies/mlx_grpc_inference && python worker.py" > worker_mini2.log 2>&1 &
    
    sleep 3
    print_message "✅ Worker started on mini2"
}

# Main execution
main() {
    print_message "🚀 Starting MLX Distributed Inference System with Monitoring"
    print_message "Model: mlx-community/Qwen3-1.7B-8bit"
    print_message "Devices: mini1 (coordinator) + mini2 (worker)"
    echo ""
    
    # Clean up first
    cleanup
    
    # Start dashboard
    if ! start_dashboard; then
        print_error "Failed to start dashboard"
        exit 1
    fi
    
    # Start remote worker
    start_remote_worker
    
    # Start server (includes local worker)
    if ! start_server; then
        print_error "Failed to start server"
        exit 1
    fi
    
    echo ""
    print_message "=== System Successfully Started ==="
    print_info "📊 Dashboard: http://localhost:8888"
    print_info "🚀 API: http://localhost:8000"
    print_info "📝 API Docs: http://localhost:8000/docs"
    echo ""
    print_message "Performance metrics:"
    print_info "  • Real-time token generation rate"
    print_info "  • GPU memory usage per device"
    print_info "  • Network bandwidth utilization"
    print_info "  • Cache hit rates"
    print_info "  • Per-layer latency breakdown"
    echo ""
    print_message "Test with:"
    print_info '  curl -X POST "http://localhost:8000/generate" \'
    print_info '    -H "Content-Type: application/json" \'
    print_info '    -d '\''{"prompt": "What is tensor parallelism?", "max_tokens": 100}'\'''
    echo ""
    print_message "Logs:"
    print_info "  • Dashboard: tail -f dashboard.log"
    print_info "  • Server: tail -f server_monitored.log"
    print_info "  • Worker: tail -f worker_mini2.log"
    echo ""
    
    # Keep running and show server logs
    print_message "Following server logs (Ctrl+C to stop)..."
    tail -f server_monitored.log
}

# Run main
main
#!/bin/bash

# MLX Distributed Inference Cluster Manager
# Unified script for managing the entire cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config/cluster_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to check if a service is running
is_service_running() {
    local service_name="$1"
    pgrep -f "$service_name" > /dev/null 2>&1
}

# Function to get cluster status
get_cluster_status() {
    log_info "📊 Cluster Status"
    echo "=================="
    
    # Check coordinator
    if is_service_running "api_server_modular.py"; then
        log_info "✅ Coordinator: Running"
        
        # Try to get detailed status via API
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            echo "🌐 API Health:"
            curl -s http://localhost:8100/health | jq '.' 2>/dev/null || echo "API responding but no jq available"
        else
            log_warn "⚠️  API not responding"
        fi
    else
        log_error "❌ Coordinator: Not running"
    fi
    
    # Check workers
    if is_service_running "worker.py"; then
        log_info "✅ Worker: Running"
    else
        log_warn "❌ Worker: Not running"
    fi
    
    # Check device configuration
    if [ -f "$CONFIG_FILE" ]; then
        local device_count=$(grep -c "device_id:" "$CONFIG_FILE" 2>/dev/null || echo "0")
        log_info "📱 Configured devices: $device_count"
    else
        log_warn "⚠️  No cluster configuration found"
    fi
}

# Function to start services
start_services() {
    log_info "🚀 Starting MLX Distributed Inference Services"
    
    cd "$PROJECT_DIR"
    
    # Determine role from device config
    local role="unknown"
    if [ -f "config/device_config.yaml" ]; then
        role=$(grep "role:" config/device_config.yaml | awk '{print $2}' | tr -d '"' || echo "unknown")
    fi
    
    log_info "🎭 Device role: $role"
    
    case "$role" in
        "coordinator")
            log_info "Starting coordinator with modular API server..."
            if ! is_service_running "api_server_modular.py"; then
                nohup uv run python working_api_server.py > logs/coordinator.log 2>&1 &
                log_info "✅ Coordinator started (PID: $!)"
            else
                log_warn "⚠️  Coordinator already running"
            fi
            ;;
        "worker")
            log_info "Starting worker..."
            if ! is_service_running "worker.py"; then
                nohup uv run python src/distributed/worker.py > logs/worker.log 2>&1 &
                log_info "✅ Worker started (PID: $!)"
            else
                log_warn "⚠️  Worker already running"
            fi
            ;;
        *)
            log_error "❌ Unknown role: $role. Please check config/device_config.yaml"
            exit 1
            ;;
    esac
    
    # Wait a moment and check if services started successfully
    sleep 3
    get_cluster_status
}

# Function to stop services
stop_services() {
    log_info "🛑 Stopping MLX Distributed Inference Services"
    
    # Stop coordinator
    if is_service_running "api_server_modular.py"; then
        pkill -f "api_server_modular.py" && log_info "✅ Coordinator stopped" || log_warn "⚠️  Failed to stop coordinator"
    fi
    
    # Stop worker
    if is_service_running "worker.py"; then
        pkill -f "worker.py" && log_info "✅ Worker stopped" || log_warn "⚠️  Failed to stop worker"
    fi
    
    # Stop legacy API server if running
    if is_service_running "working_api_server.py"; then
        pkill -f "working_api_server.py" && log_info "✅ Legacy API server stopped"
    fi
}

# Function to restart services
restart_services() {
    log_info "🔄 Restarting MLX Distributed Inference Services"
    stop_services
    sleep 2
    start_services
}

# Function to view logs
view_logs() {
    local service="${1:-all}"
    
    cd "$PROJECT_DIR"
    mkdir -p logs
    
    case "$service" in
        "coordinator")
            if [ -f "logs/coordinator.log" ]; then
                tail -f logs/coordinator.log
            else
                log_warn "No coordinator logs found"
            fi
            ;;
        "worker")
            if [ -f "logs/worker.log" ]; then
                tail -f logs/worker.log
            else
                log_warn "No worker logs found"
            fi
            ;;
        "all"|*)
            log_info "📋 Available log files:"
            ls -la logs/ 2>/dev/null || log_warn "No logs directory found"
            
            if [ -f "logs/coordinator.log" ] && [ -f "logs/worker.log" ]; then
                log_info "Showing combined logs (Ctrl+C to exit):"
                tail -f logs/*.log
            elif [ -f "logs/coordinator.log" ]; then
                log_info "Showing coordinator logs:"
                tail -f logs/coordinator.log
            elif [ -f "logs/worker.log" ]; then
                log_info "Showing worker logs:"
                tail -f logs/worker.log
            else
                log_warn "No log files found"
            fi
            ;;
    esac
}

# Function to add a device to the cluster
add_device() {
    local device_hostname="$1"
    local device_role="${2:-worker}"
    
    if [ -z "$device_hostname" ]; then
        log_error "Usage: $0 add-device <hostname> [role]"
        exit 1
    fi
    
    log_info "➕ Adding device: $device_hostname (role: $device_role)"
    
    # Use API to add device if coordinator is running
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        local device_id=$(echo "$device_hostname" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
        
        curl -X POST http://localhost:8100/cluster/devices \
            -H "Content-Type: application/json" \
            -d "{
                \"device_id\": \"$device_id\",
                \"hostname\": \"$device_hostname\",
                \"role\": \"$device_role\",
                \"capabilities\": {
                    \"model\": \"Apple M4\",
                    \"memory_gb\": 16,
                    \"gpu_cores\": 10,
                    \"cpu_cores\": 10,
                    \"cpu_performance_cores\": 4,
                    \"cpu_efficiency_cores\": 6,
                    \"neural_engine_cores\": 16,
                    \"bandwidth_gbps\": 120.0,
                    \"mlx_metal_available\": true,
                    \"max_recommended_model_size_gb\": 10.0
                }
            }"
        
        log_info "✅ Device addition request sent"
    else
        log_error "❌ Coordinator not available. Please start the coordinator first."
        exit 1
    fi
}

# Function to migrate coordinator
migrate_coordinator() {
    local new_coordinator="$1"
    
    if [ -z "$new_coordinator" ]; then
        log_error "Usage: $0 migrate-coordinator <device_id>"
        exit 1
    fi
    
    log_info "🔄 Migrating coordinator to: $new_coordinator"
    
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        curl -X POST http://localhost:8100/cluster/migrate-coordinator \
            -H "Content-Type: application/json" \
            -d "{\"new_coordinator_id\": \"$new_coordinator\"}"
        
        log_info "✅ Coordinator migration request sent"
    else
        log_error "❌ Coordinator not available"
        exit 1
    fi
}

# Function to install on remote device
install_remote() {
    local hostname="$1"
    local role="${2:-worker}"
    
    if [ -z "$hostname" ]; then
        log_error "Usage: $0 install-remote <hostname> [role]"
        exit 1
    fi
    
    log_info "📦 Installing on remote device: $hostname"
    
    # Copy installer script to remote device
    scp "$SCRIPT_DIR/install-client.sh" "$hostname:/tmp/"
    
    # Run installer on remote device
    ssh "$hostname" "bash /tmp/install-client.sh auto $role"
    
    log_info "✅ Remote installation complete"
}

# Function to show help
show_help() {
    echo "MLX Distributed Inference Cluster Manager"
    echo "========================================"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start                    Start services based on device role"
    echo "  stop                     Stop all services"
    echo "  restart                  Restart all services"
    echo "  status                   Show cluster status"
    echo "  logs [service]           View logs (coordinator|worker|all)"
    echo "  add-device <host> [role] Add device to cluster"
    echo "  migrate-coordinator <id> Migrate coordinator to device"
    echo "  install-remote <host>    Install on remote device"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # Start local services"
    echo "  $0 status                          # Check cluster status"
    echo "  $0 add-device mini3.local worker   # Add a worker device"
    echo "  $0 migrate-coordinator master      # Migrate coordinator to master"
    echo "  $0 install-remote mini4.local      # Install on mini4"
}

# Main command handling
case "${1:-help}" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        get_cluster_status
        ;;
    "logs")
        view_logs "$2"
        ;;
    "add-device")
        add_device "$2" "$3"
        ;;
    "migrate-coordinator")
        migrate_coordinator "$2"
        ;;
    "install-remote")
        install_remote "$2" "$3"
        ;;
    "help"|*)
        show_help
        ;;
esac
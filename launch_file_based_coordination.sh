#!/bin/bash
#
# File-Based Distributed Launcher for Thunderbolt Bridge Networks  
# Bypasses PyTorch distributed Gloo backend limitations
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config/cluster_config.yaml"
LOG_DIR="${SCRIPT_DIR}/logs"
COORD_DIR="/tmp/mlx_coordination"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Clean up coordination directory
cleanup_coordination() {
    log "Cleaning up coordination directory..."
    rm -rf "${COORD_DIR}"
    mkdir -p "${COORD_DIR}"
}

# Check if coordination directory is accessible
check_coordination_dir() {
    if [[ ! -d "${COORD_DIR}" ]]; then
        mkdir -p "${COORD_DIR}"
    fi
    
    if [[ ! -w "${COORD_DIR}" ]]; then
        error "Coordination directory ${COORD_DIR} is not writable"
        return 1
    fi
    
    log "Coordination directory ready: ${COORD_DIR}"
    return 0
}

# Get device configuration
get_device_config() {
    local hostname=$(hostname -s)
    
    # Parse YAML config to get our rank and world size
    # This is a simple approach - in production you might use a proper YAML parser
    local rank world_size
    
    if [[ "${hostname}" == "mini1" ]]; then
        rank=0
        world_size=2  # mini1 + mini2
    elif [[ "${hostname}" == "mini2" ]]; then
        rank=1
        world_size=2
    elif [[ "${hostname}" == "master" ]]; then
        rank=2
        world_size=3  # mini1 + mini2 + master
    else
        # Error message to stderr
        >&2 echo "Unknown hostname: ${hostname}"
        return 1
    fi
    
    # Return values without any formatting/colors or logging
    echo "${rank} ${world_size}"
}

# Start single node
start_single_node() {
    local rank=0
    local world_size=1
    
    log "Starting single-node file-based server..."
    
    export RANK=${rank}
    export WORLD_SIZE=${world_size}
    export CONFIG_PATH="${CONFIG_PATH}"
    
    local log_file="${LOG_DIR}/file_based_single_$(date +%Y%m%d_%H%M%S).log"
    
    uv run python server_file_based.py 2>&1 | tee "${log_file}" &
    local pid=$!
    
    echo ${pid} > "${LOG_DIR}/single.pid"
    log "Single-node server started with PID ${pid}"
    log "Logs: ${log_file}"
    
    return 0
}

# Start distributed cluster
start_distributed() {
    # Get device config
    local config=$(get_device_config)
    if [[ $? -ne 0 ]]; then
        error "Failed to get device configuration"
        return 1
    fi
    
    local rank=$(echo "${config}" | cut -d' ' -f1)
    local world_size=$(echo "${config}" | cut -d' ' -f2)
    
    log "Starting file-based distributed server..."
    log "Rank: ${rank}, World Size: ${world_size}"
    
    # Clean up coordination directory
    if [[ "${rank}" -eq 0 ]]; then
        cleanup_coordination
    fi
    
    # Check coordination directory
    if ! check_coordination_dir; then
        return 1
    fi
    
    export RANK=${rank}
    export WORLD_SIZE=${world_size}
    export CONFIG_PATH="${CONFIG_PATH}"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="${LOG_DIR}/file_based_rank${rank}_${timestamp}.log"
    
    # Start server
    uv run python server_file_based.py 2>&1 | tee "${log_file}" &
    local pid=$!
    
    echo ${pid} > "${LOG_DIR}/rank${rank}.pid"
    log "Rank ${rank} server started with PID ${pid}"
    log "Logs: ${log_file}"
    
    # If coordinator, show additional info
    if [[ "${rank}" -eq 0 ]]; then
        info "Coordinator started - API will be available at http://localhost:8100"
        info "Waiting for all ${world_size} nodes to join..."
        
        # Monitor coordination status
        monitor_coordination &
    fi
    
    return 0
}

# Monitor coordination status
monitor_coordination() {
    local max_wait=60  # 60 seconds
    local waited=0
    
    while [[ ${waited} -lt ${max_wait} ]]; do
        if [[ -f "${COORD_DIR}/nodes.json" ]]; then
            local node_count=$(jq '. | length' "${COORD_DIR}/nodes.json" 2>/dev/null || echo "0")
            local world_size=${WORLD_SIZE:-2}
            
            if [[ "${node_count}" -eq "${world_size}" ]]; then
                log "All ${world_size} nodes joined the cluster!"
                info "Testing API endpoint..."
                sleep 2
                
                if curl -s http://localhost:8100/health > /dev/null; then
                    log "API server is responding"
                    info "You can now test with: curl http://localhost:8100/health"
                else
                    warn "API server not yet responding"
                fi
                return 0
            else
                info "Nodes joined: ${node_count}/${world_size}"
            fi
        fi
        
        sleep 2
        waited=$((waited + 2))
    done
    
    warn "Timeout waiting for all nodes to join"
}

# Stop all processes
stop_all() {
    log "Stopping all file-based servers..."
    
    # Kill processes by PID files
    for pid_file in "${LOG_DIR}"/*.pid; do
        if [[ -f "${pid_file}" ]]; then
            local pid=$(cat "${pid_file}")
            if ps -p ${pid} > /dev/null 2>&1; then
                log "Stopping process ${pid}"
                kill ${pid}
                sleep 1
                
                # Force kill if still running
                if ps -p ${pid} > /dev/null 2>&1; then
                    kill -9 ${pid}
                fi
            fi
            rm -f "${pid_file}"
        fi
    done
    
    # Clean up coordination directory
    cleanup_coordination
    
    log "All processes stopped"
}

# Show status
show_status() {
    log "File-based distributed inference status:"
    
    # Check running processes
    local running=0
    for pid_file in "${LOG_DIR}"/*.pid; do
        if [[ -f "${pid_file}" ]]; then
            local pid=$(cat "${pid_file}")
            local name=$(basename "${pid_file}" .pid)
            
            if ps -p ${pid} > /dev/null 2>&1; then
                info "✓ ${name} (PID: ${pid}) - Running"
                running=$((running + 1))
            else
                warn "✗ ${name} (PID: ${pid}) - Not running"
                rm -f "${pid_file}"
            fi
        fi
    done
    
    if [[ "${running}" -eq 0 ]]; then
        warn "No processes running"
        return 1
    fi
    
    # Check coordination status
    if [[ -f "${COORD_DIR}/nodes.json" ]]; then
        local node_count=$(jq '. | length' "${COORD_DIR}/nodes.json" 2>/dev/null || echo "0")
        info "Coordination: ${node_count} nodes registered"
        
        # Show node details
        if command -v jq > /dev/null; then
            jq -r '.[] | "  - Rank \(.rank): \(.hostname) (\(.status))"' "${COORD_DIR}/nodes.json" 2>/dev/null || true
        fi
    else
        warn "No coordination file found"
    fi
    
    return 0
}

# Test the cluster
test_cluster() {
    log "Testing file-based distributed inference..."
    
    # Check if coordinator is running
    if ! curl -s http://localhost:8100/health > /dev/null; then
        error "Coordinator API not responding on http://localhost:8100"
        return 1
    fi
    
    # Test health endpoint
    log "Testing health endpoint..."
    local health_response=$(curl -s http://localhost:8100/health)
    echo "${health_response}" | jq . 2>/dev/null || echo "${health_response}"
    
    # Test generation endpoint
    log "Testing generation endpoint..."
    local gen_response=$(curl -s -X POST http://localhost:8100/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "The future of AI is", "max_tokens": 20}')
    echo "${gen_response}" | jq . 2>/dev/null || echo "${gen_response}"
    
    log "Cluster test complete"
}

# Show logs
show_logs() {
    local rank=${1:-0}
    local log_pattern="${LOG_DIR}/file_based_rank${rank}_*.log"
    local latest_log=$(ls -t ${log_pattern} 2>/dev/null | head -1)
    
    if [[ -f "${latest_log}" ]]; then
        log "Showing logs for rank ${rank}: ${latest_log}"
        tail -f "${latest_log}"
    else
        error "No log file found for rank ${rank}"
        return 1
    fi
}

# Main command handling
case "${1:-help}" in
    "single")
        start_single_node
        ;;
    "start")
        start_distributed
        ;;
    "stop")
        stop_all
        ;;
    "status")
        show_status
        ;;
    "test")
        test_cluster
        ;;
    "logs")
        show_logs "${2:-0}"
        ;;
    "clean")
        stop_all
        cleanup_coordination
        ;;
    "help"|*)
        echo "File-Based Distributed MLX Inference Launcher"
        echo "Bypasses PyTorch distributed Gloo backend limitations"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  single    - Start single-node server"
        echo "  start     - Start distributed cluster"
        echo "  stop      - Stop all servers"
        echo "  status    - Show cluster status"
        echo "  test      - Test cluster functionality"
        echo "  logs [rank] - Show logs for specific rank (default: 0)"
        echo "  clean     - Stop all and clean coordination files"
        echo "  help      - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 single              # Single-node testing"
        echo "  $0 start               # Start distributed cluster"
        echo "  $0 status              # Check cluster status"
        echo "  $0 test                # Test inference API"
        echo "  $0 logs 1              # Show worker logs"
        echo ""
        ;;
esac
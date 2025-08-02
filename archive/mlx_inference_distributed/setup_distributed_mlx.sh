#!/bin/bash
# Setup script for distributed MLX inference
# Discovers devices and installs dependencies

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$SCRIPT_DIR"
PYTHON_VERSION="3.11"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to detect network interfaces
detect_network_interfaces() {
    print_status "Detecting network interfaces..."
    
    # Check for Thunderbolt interfaces
    local tb_interfaces=$(networksetup -listallhardwareports | grep -A2 "Thunderbolt" | grep "Device:" | awk '{print $2}')
    
    if [ -n "$tb_interfaces" ]; then
        print_info "Found Thunderbolt interfaces: $tb_interfaces"
    fi
    
    # Check for Ethernet interfaces
    local eth_interfaces=$(networksetup -listallhardwareports | grep -A2 "Ethernet" | grep "Device:" | awk '{print $2}')
    
    if [ -n "$eth_interfaces" ]; then
        print_info "Found Ethernet interfaces: $eth_interfaces"
    fi
}

# Function to discover devices on network
discover_devices() {
    print_status "Discovering devices on network..."
    
    # Method 1: Check known hostnames
    local known_hosts=("mini2.local" "master.local" "studio.local")
    local discovered_hosts=()
    
    for host in "${known_hosts[@]}"; do
        if ping -c 1 -W 2 $host >/dev/null 2>&1; then
            discovered_hosts+=($host)
            print_info "Found device: $host"
            
            # Try to get device info
            local device_info=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $host "hostname && sysctl -n hw.model 2>/dev/null" 2>/dev/null || echo "Unknown")
            if [ "$device_info" != "Unknown" ]; then
                print_info "  Device info: $device_info"
            fi
        fi
    done
    
    # Method 2: Use mDNS to discover _ssh._tcp services
    print_status "Scanning for SSH services via mDNS..."
    local mdns_hosts=$(dns-sd -B _ssh._tcp local. 2>/dev/null | grep -E "mini|master|studio" | awk '{print $NF}' | sort -u || true)
    
    # Method 3: Scan local subnet
    print_status "Scanning local subnet for Apple devices..."
    local my_ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    if [ -n "$my_ip" ]; then
        local subnet=$(echo $my_ip | cut -d. -f1-3)
        print_info "Scanning subnet: $subnet.0/24"
        
        # Quick scan for common Apple device ports
        for i in {1..254}; do
            (nc -z -w 1 $subnet.$i 22 2>/dev/null && echo "$subnet.$i") &
        done | while read ip; do
            print_info "Found device at: $ip"
        done
    fi
    
    echo ""
    print_status "Discovered devices: ${discovered_hosts[@]}"
    echo ""
    
    return 0
}

# Function to check device prerequisites
check_device_prerequisites() {
    local host=$1
    local user=$2
    
    print_status "Checking prerequisites on $host..."
    
    local ssh_prefix=""
    if [ "$host" != "localhost" ] && [ "$host" != "$(hostname)" ]; then
        if [ -n "$user" ]; then
            ssh_prefix="ssh $user@$host"
        else
            ssh_prefix="ssh $host"
        fi
    fi
    
    # Check macOS version
    local macos_version=$($ssh_prefix sw_vers -productVersion 2>/dev/null || echo "Unknown")
    print_info "macOS version: $macos_version"
    
    # Check if Apple Silicon
    local chip=$($ssh_prefix sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    if [[ $chip == *"Apple"* ]]; then
        print_info "✓ Apple Silicon detected: $chip"
    else
        print_error "Not Apple Silicon: $chip"
        return 1
    fi
    
    # Check available memory
    local mem_bytes=$($ssh_prefix sysctl -n hw.memsize 2>/dev/null || echo "0")
    local mem_gb=$((mem_bytes / 1024 / 1024 / 1024))
    print_info "Available memory: ${mem_gb}GB"
    
    # Check Python
    local python_version=$($ssh_prefix python3 --version 2>/dev/null || echo "Not installed")
    print_info "Python: $python_version"
    
    # Check if mlx is available
    local has_mlx=$($ssh_prefix "python3 -c 'import mlx' 2>/dev/null && echo 'yes' || echo 'no'" 2>/dev/null)
    if [ "$has_mlx" = "yes" ]; then
        print_info "✓ MLX is installed"
    else
        print_info "✗ MLX not installed"
    fi
    
    return 0
}

# Function to setup remote device
setup_remote_device() {
    local host=$1
    local user=$2
    
    print_status "Setting up MLX distributed on $host..."
    
    local ssh_cmd="ssh"
    if [ -n "$user" ]; then
        ssh_cmd="ssh $user@$host"
    else
        ssh_cmd="ssh $host"
    fi
    
    # Create directory structure
    print_info "Creating directory structure..."
    $ssh_cmd "mkdir -p ~/Movies/mlx_inference_distributed/logs"
    
    # Copy necessary files
    print_info "Copying project files..."
    local target_dir="Movies/mlx_inference_distributed"
    if [ "$host" = "master.local" ] && [ -n "$user" ]; then
        target_dir="~/Movies/mlx_inference_distributed"
    fi
    
    # Use rsync to copy files
    if [ -n "$user" ]; then
        rsync -avz --exclude='.venv' --exclude='logs' --exclude='__pycache__' \
            "$REPO_PATH/" "$user@$host:$target_dir/"
    else
        rsync -avz --exclude='.venv' --exclude='logs' --exclude='__pycache__' \
            "$REPO_PATH/" "$host:$target_dir/"
    fi
    
    # Setup Python environment
    print_info "Setting up Python virtual environment..."
    $ssh_cmd "cd $target_dir && python3 -m venv .venv"
    
    # Install dependencies
    print_info "Installing dependencies..."
    $ssh_cmd "cd $target_dir && source .venv/bin/activate && pip install --upgrade pip"
    $ssh_cmd "cd $target_dir && source .venv/bin/activate && pip install -r requirements.txt"
    
    # Verify installation
    print_info "Verifying installation..."
    local mlx_check=$($ssh_cmd "cd $target_dir && source .venv/bin/activate && python -c 'import mlx; print(\"MLX OK\")'" 2>&1)
    if [[ $mlx_check == *"MLX OK"* ]]; then
        print_info "✓ Installation verified on $host"
    else
        print_error "Installation verification failed on $host"
        return 1
    fi
    
    return 0
}

# Function to generate device configuration
generate_device_config() {
    local hosts=("$@")
    
    print_status "Generating device configuration..."
    
    # TODO: Dynamically generate distributed_config.json based on discovered devices
    # For now, we'll use the existing configuration
    
    print_info "Using existing distributed_config.json"
}

# Main setup flow
main() {
    echo "Distributed MLX Inference Setup"
    echo "==============================="
    echo ""
    
    # Detect network interfaces
    detect_network_interfaces
    echo ""
    
    # Discover devices
    discover_devices
    echo ""
    
    # Check if we should setup specific devices
    if [ "$1" = "install" ]; then
        # Setup mini2
        if ping -c 1 -W 2 mini2.local >/dev/null 2>&1; then
            print_status "Setting up mini2.local..."
            check_device_prerequisites "mini2.local" ""
            setup_remote_device "mini2.local" ""
        else
            print_error "Cannot reach mini2.local"
        fi
        
        echo ""
        
        # Setup master.local
        if ping -c 1 -W 2 master.local >/dev/null 2>&1; then
            print_status "Setting up master.local..."
            check_device_prerequisites "master.local" "georgedekker"
            setup_remote_device "master.local" "georgedekker"
        else
            print_error "Cannot reach master.local"
        fi
        
        echo ""
        print_status "Setup complete!"
        echo ""
        echo "You can now run: ./control_distributed_mlx.sh start"
    else
        print_info "Run '$0 install' to setup remote devices"
        echo ""
        
        # Just check prerequisites
        print_status "Checking prerequisites on known devices..."
        
        if ping -c 1 -W 2 mini2.local >/dev/null 2>&1; then
            check_device_prerequisites "mini2.local" ""
            echo ""
        fi
        
        if ping -c 1 -W 2 master.local >/dev/null 2>&1; then
            check_device_prerequisites "master.local" "georgedekker"
            echo ""
        fi
    fi
}

# Run main function
main "$@"
#!/bin/bash
# Setup network configuration for heterogeneous cluster

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if running as root for hosts file modification
check_sudo() {
    if [ "$EUID" -ne 0 ]; then 
        print_error "This script needs sudo privileges to modify /etc/hosts"
        print_status "Please run: sudo $0"
        exit 1
    fi
}

# Setup hosts file
setup_hosts() {
    print_status "Setting up /etc/hosts entries..."
    
    # Backup hosts file
    cp /etc/hosts /etc/hosts.backup.$(date +%Y%m%d_%H%M%S)
    
    # Remove existing entries
    sed -i '' '/# Heterogeneous MLX Cluster/,/# End Heterogeneous MLX Cluster/d' /etc/hosts
    
    # Add new entries
    cat >> /etc/hosts << EOF

# Heterogeneous MLX Cluster
10.0.0.80   mini1
10.0.0.16   mini2
127.0.0.1   master
# End Heterogeneous MLX Cluster
EOF
    
    print_status "Added cluster hosts to /etc/hosts"
}

# Setup SSH keys
setup_ssh_keys() {
    print_status "Setting up SSH keys..."
    
    # Drop sudo for SSH operations
    REAL_USER=$(logname)
    REAL_HOME=$(eval echo ~$REAL_USER)
    
    # Generate SSH key if it doesn't exist
    if [ ! -f "$REAL_HOME/.ssh/id_rsa" ]; then
        print_status "Generating SSH key..."
        sudo -u $REAL_USER ssh-keygen -t rsa -b 4096 -f "$REAL_HOME/.ssh/id_rsa" -N ""
    fi
    
    # Create SSH config
    print_status "Creating SSH config..."
    sudo -u $REAL_USER tee "$REAL_HOME/.ssh/config" > /dev/null << EOF
Host mini1
    HostName 10.0.0.80
    User mini1
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host mini2
    HostName 10.0.0.16
    User mini2
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host master
    HostName 127.0.0.1
    User $REAL_USER
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    
    chmod 600 "$REAL_HOME/.ssh/config"
    chown $REAL_USER:staff "$REAL_HOME/.ssh/config"
    
    print_status "SSH config created"
}

# Test connectivity
test_connectivity() {
    print_status "Testing connectivity..."
    
    # Drop sudo for connectivity tests
    REAL_USER=$(logname)
    
    # Test local resolution
    if ping -c 1 mini1 > /dev/null 2>&1; then
        print_status "✓ mini1 (local) is reachable"
    else
        print_error "✗ Cannot reach mini1"
    fi
    
    if ping -c 1 mini2 > /dev/null 2>&1; then
        print_status "✓ mini2 is reachable"
    else
        print_error "✗ Cannot reach mini2"
        print_warning "Make sure mini2 is powered on and connected"
    fi
}

# Setup SSH access to mini2
setup_mini2_ssh() {
    print_status "Setting up SSH access to mini2..."
    
    REAL_USER=$(logname)
    REAL_HOME=$(eval echo ~$REAL_USER)
    
    print_warning "You'll need to:"
    print_warning "1. Make sure mini2 is powered on"
    print_warning "2. Copy your SSH key to mini2"
    echo ""
    print_status "Run this command to copy your SSH key:"
    echo "  ssh-copy-id mini2@10.0.0.16"
    echo ""
    print_status "Or manually on mini2:"
    echo "  1. Create ~/.ssh directory: mkdir -p ~/.ssh"
    echo "  2. Add this key to ~/.ssh/authorized_keys:"
    echo ""
    cat "$REAL_HOME/.ssh/id_rsa.pub"
    echo ""
}

# Create device-specific configs
create_device_configs() {
    print_status "Creating device-specific configurations..."
    
    REAL_USER=$(logname)
    SCRIPT_DIR="/Users/$REAL_USER/Movies/mlx_grpc_inference"
    
    # Create config for 2-device setup
    sudo -u $REAL_USER tee "$SCRIPT_DIR/config/heterogeneous_2device.json" > /dev/null << EOF
{
  "cluster": {
    "name": "mac-heterogeneous-cluster-2device",
    "devices": ["mini1", "mini2"]
  },
  "model": {
    "name": "mlx-community/Qwen3-1.7B-8bit",
    "sharding_strategy": "equal",
    "total_layers": 28
  },
  "devices": {
    "mini1": {
      "hostname": "mini1",
      "memory_gb": 16.0,
      "gpu_cores": 10,
      "gpu_memory_gb": 12.0,
      "bandwidth_gbps": 10.0
    },
    "mini2": {
      "hostname": "mini2", 
      "memory_gb": 16.0,
      "gpu_cores": 10,
      "gpu_memory_gb": 12.0,
      "bandwidth_gbps": 10.0
    }
  },
  "communication": {
    "master_addr": "10.0.0.80",
    "master_port": 29501,
    "backend": "gloo"
  }
}
EOF

    # Update main config to use IP addresses
    sudo -u $REAL_USER tee "$SCRIPT_DIR/config/heterogeneous_cluster.json" > /dev/null << EOF
{
  "cluster": {
    "name": "mac-heterogeneous-cluster",
    "devices": ["mini1", "mini2", "master"]
  },
  "model": {
    "name": "mlx-community/Qwen3-1.7B-8bit",
    "sharding_strategy": "capability_based",
    "total_layers": 28
  },
  "devices": {
    "mini1": {
      "hostname": "10.0.0.80",
      "memory_gb": 16.0,
      "gpu_cores": 10,
      "gpu_memory_gb": 12.0,
      "bandwidth_gbps": 10.0
    },
    "mini2": {
      "hostname": "10.0.0.16", 
      "memory_gb": 16.0,
      "gpu_cores": 10,
      "gpu_memory_gb": 12.0,
      "bandwidth_gbps": 10.0
    },
    "master": {
      "hostname": "127.0.0.1",
      "memory_gb": 48.0,
      "gpu_cores": 30,
      "gpu_memory_gb": 36.0,
      "bandwidth_gbps": 10.0
    }
  },
  "communication": {
    "master_addr": "10.0.0.80",
    "master_port": 29501,
    "backend": "gloo"
  }
}
EOF
    
    print_status "Configuration files created"
}

# Main setup
main() {
    print_status "Heterogeneous Cluster Network Setup"
    print_status "===================================="
    
    # Check sudo
    check_sudo
    
    # Setup hosts
    setup_hosts
    
    # Setup SSH
    setup_ssh_keys
    
    # Create configs
    create_device_configs
    
    # Test connectivity
    test_connectivity
    
    # SSH setup instructions
    setup_mini2_ssh
    
    print_status "===================================="
    print_status "Setup complete!"
    print_warning "Next steps:"
    print_warning "1. Set up SSH access to mini2 using the instructions above"
    print_warning "2. Test with: ssh mini2 'echo Connected successfully'"
    print_warning "3. Run: CONFIG_FILE=config/heterogeneous_2device.json ./launch_heterogeneous.sh"
}

# Run main
main
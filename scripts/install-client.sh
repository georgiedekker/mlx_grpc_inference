#!/bin/bash
set -e

# MLX Distributed Inference Client Installer
# Usage: curl -sSL https://your-domain.com/install.sh | bash
# Or: ./install-client.sh [coordinator_host] [device_role]

COORDINATOR_HOST="${1:-auto}"
DEVICE_ROLE="${2:-auto}"
INSTALL_DIR="${HOME}/mlx_distributed"
SERVICE_NAME="mlx-distributed"

echo "🚀 MLX Distributed Inference Client Installer"
echo "================================================"

# Detect system information
detect_system() {
    echo "🔍 Detecting system capabilities..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        echo "✅ Detected macOS"
    else
        echo "❌ Error: Only macOS is currently supported"
        exit 1
    fi
    
    # Detect Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        ARCH="arm64"
        echo "✅ Detected Apple Silicon (ARM64)"
    else
        echo "❌ Error: Only Apple Silicon is currently supported"
        exit 1
    fi
    
    # Detect memory
    MEMORY_GB=$(( $(sysctl hw.memsize | awk '{print $2}') / 1024 / 1024 / 1024 ))
    echo "✅ Detected ${MEMORY_GB}GB RAM"
    
    # Detect hostname
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    echo "✅ Hostname: ${HOSTNAME}"
}

# Install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Python 3.13 if not present
    if ! command -v python3.13 &> /dev/null; then
        echo "Installing Python 3.13..."
        brew install python@3.13
    fi
    
    # Install UV if not present
    if ! command -v uv &> /dev/null; then
        echo "Installing UV package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    echo "✅ Dependencies installed"
}

# Download and setup application
setup_application() {
    echo "⬇️  Setting up MLX Distributed Inference..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Download application (for now, we'll assume it's copied)
    if [ ! -d "mlx_grpc_inference" ]; then
        echo "📁 Please copy the mlx_grpc_inference directory to $INSTALL_DIR"
        echo "   Or implement git clone here for production"
        # git clone https://github.com/your-repo/mlx_grpc_inference.git
        return 1
    fi
    
    cd mlx_grpc_inference
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    uv sync
    
    echo "✅ Application setup complete"
}

# Auto-detect device role
detect_device_role() {
    if [ "$DEVICE_ROLE" = "auto" ]; then
        echo "🤖 Auto-detecting device role..."
        
        # Simple heuristic: devices with more memory become coordinators
        if [ "$MEMORY_GB" -ge 32 ]; then
            DETECTED_ROLE="coordinator"
            echo "✅ High memory detected - suggesting coordinator role"
        else
            DETECTED_ROLE="worker"
            echo "✅ Standard memory - suggesting worker role"
        fi
        
        echo "Detected role: $DETECTED_ROLE"
        echo "Press Enter to accept, or type 'coordinator' or 'worker' to override:"
        read -r user_input
        
        if [ -n "$user_input" ]; then
            DEVICE_ROLE="$user_input"
        else
            DEVICE_ROLE="$DETECTED_ROLE"
        fi
    fi
    
    echo "✅ Device role: $DEVICE_ROLE"
}

# Generate device configuration
generate_config() {
    echo "⚙️  Generating device configuration..."
    
    # Generate unique device ID
    DEVICE_ID=$(echo "$HOSTNAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
    
    # Generate device configuration
    cat > config/device_config.yaml << EOF
device:
  device_id: "${DEVICE_ID}"
  hostname: "${HOSTNAME}"
  role: "${DEVICE_ROLE}"
  capabilities:
    model: "Apple M4"
    memory_gb: ${MEMORY_GB}
    gpu_cores: 10  # Default for M4
    cpu_cores: 10
    mlx_metal_available: true
    max_recommended_model_size_gb: $((MEMORY_GB * 6 / 10))

network:
  grpc_port: 50051
  api_port: 8100
  coordinator_host: "${COORDINATOR_HOST}"

performance:
  batch_size: 1
  max_sequence_length: 2048
  tensor_compression: false
  connection_pool_size: 10
EOF

    echo "✅ Configuration generated: config/device_config.yaml"
}

# Create service scripts
create_service_scripts() {
    echo "🔧 Creating service scripts..."
    
    # Create start script
    cat > scripts/start-service.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

DEVICE_CONFIG="config/device_config.yaml"
ROLE=$(grep "role:" "$DEVICE_CONFIG" | awk '{print $2}')

echo "🚀 Starting MLX Distributed Inference ($ROLE)..."

if [ "$ROLE" = "coordinator" ]; then
    echo "Starting coordinator with API server..."
    uv run python working_api_server.py
elif [ "$ROLE" = "worker" ]; then
    echo "Starting worker..."
    uv run python src/distributed/worker.py
else
    echo "❌ Unknown role: $ROLE"
    exit 1
fi
EOF

    # Create stop script
    cat > scripts/stop-service.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping MLX Distributed Inference..."
pkill -f "python.*worker.py" || true
pkill -f "python.*working_api_server.py" || true
echo "✅ Services stopped"
EOF

    # Create status script
    cat > scripts/status.sh << 'EOF'
#!/bin/bash
echo "📊 MLX Distributed Inference Status"
echo "=================================="

WORKER_PID=$(pgrep -f "python.*worker.py" || echo "")
API_PID=$(pgrep -f "python.*working_api_server.py" || echo "")

if [ -n "$WORKER_PID" ]; then
    echo "✅ Worker running (PID: $WORKER_PID)"
else
    echo "❌ Worker not running"
fi

if [ -n "$API_PID" ]; then
    echo "✅ API Server running (PID: $API_PID)"
else
    echo "❌ API Server not running"
fi

# Check if coordinator is reachable (if we're a worker)
DEVICE_CONFIG="config/device_config.yaml"
if [ -f "$DEVICE_CONFIG" ]; then
    COORDINATOR=$(grep "coordinator_host:" "$DEVICE_CONFIG" | awk '{print $2}')
    ROLE=$(grep "role:" "$DEVICE_CONFIG" | awk '{print $2}')
    
    if [ "$ROLE" = "worker" ] && [ "$COORDINATOR" != "auto" ]; then
        if curl -s "http://$COORDINATOR:8100/health" > /dev/null; then
            echo "✅ Coordinator reachable at $COORDINATOR"
        else
            echo "❌ Coordinator unreachable at $COORDINATOR"
        fi
    fi
fi
EOF

    # Make scripts executable
    chmod +x scripts/*.sh
    
    echo "✅ Service scripts created"
}

# Create launchd service (macOS)
create_system_service() {
    echo "⚙️  Creating system service..."
    
    SERVICE_PLIST="$HOME/Library/LaunchAgents/com.mlx.distributed.plist"
    
    cat > "$SERVICE_PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx.distributed</string>
    <key>ProgramArguments</key>
    <array>
        <string>$INSTALL_DIR/mlx_grpc_inference/scripts/start-service.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR/mlx_grpc_inference</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/mlx_grpc_inference/logs/service.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/mlx_grpc_inference/logs/service.error.log</string>
</dict>
</plist>
EOF

    # Create logs directory
    mkdir -p "$INSTALL_DIR/mlx_grpc_inference/logs"
    
    echo "✅ System service created: $SERVICE_PLIST"
    echo "To start service: launchctl load $SERVICE_PLIST"
    echo "To stop service: launchctl unload $SERVICE_PLIST"
}

# Main installation flow
main() {
    detect_system
    install_dependencies
    setup_application
    detect_device_role
    generate_config
    create_service_scripts
    create_system_service
    
    echo ""
    echo "🎉 Installation Complete!"
    echo "========================"
    echo "Device ID: $DEVICE_ID"
    echo "Role: $DEVICE_ROLE"
    echo "Install Directory: $INSTALL_DIR/mlx_grpc_inference"
    echo ""
    echo "Next steps:"
    echo "1. Start service: cd $INSTALL_DIR/mlx_grpc_inference && ./scripts/start-service.sh"
    echo "2. Check status: ./scripts/status.sh"
    echo "3. Auto-start on boot: launchctl load $HOME/Library/LaunchAgents/com.mlx.distributed.plist"
    echo ""
    echo "For coordinator migration and advanced features, see the documentation."
}

# Run main function
main "$@"
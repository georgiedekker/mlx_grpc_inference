#!/bin/bash
# Deploy worker files to mini2 and m4

echo "=== Deploying MLX gRPC Worker Files ==="

# Files to copy
FILES_TO_COPY=(
    "worker.py"
    "launch_worker.sh"
    "requirements.txt"
    "pyproject.toml"
    "src/communication/inference_pb2.py"
    "src/communication/inference_pb2_grpc.py"
    "src/communication/tensor_utils.py"
    "src/communication/__init__.py"
    "src/__init__.py"
)

# Function to deploy to a device
deploy_to_device() {
    local HOST=$1
    local USER=$2
    local DEVICE_NAME=$3
    
    echo ""
    echo "Deploying to $DEVICE_NAME ($HOST)..."
    
    # Create directory structure
    echo "Creating directories..."
    ssh $USER@$HOST "mkdir -p ~/mlx_grpc_worker/src/communication"
    
    # Copy files
    for file in "${FILES_TO_COPY[@]}"; do
        echo "Copying $file..."
        scp "$file" "$USER@$HOST:~/mlx_grpc_worker/$file"
    done
    
    # Create a simple setup script
    echo "Creating setup script..."
    ssh $USER@$HOST 'cat > ~/mlx_grpc_worker/setup.sh << "EOF"
#!/bin/bash
# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install uv
uv pip install grpcio grpcio-tools mlx mlx-lm fastapi uvicorn pydantic numpy lz4

echo "Setup complete!"
EOF'
    
    ssh $USER@$HOST "chmod +x ~/mlx_grpc_worker/setup.sh ~/mlx_grpc_worker/launch_worker.sh"
    
    echo "✓ Deployment to $DEVICE_NAME complete"
    echo "  Next steps on $DEVICE_NAME:"
    echo "  1. cd ~/mlx_grpc_worker"
    echo "  2. ./setup.sh  # Install dependencies"
    echo "  3. ./launch_worker.sh $4 3  # Start worker"
}

# Deploy to mini2 (192.168.5.2)
if ping -c 1 192.168.5.2 > /dev/null 2>&1; then
    deploy_to_device "192.168.5.2" "mini2" "mini2" "1"
else
    echo "⚠️  Cannot reach mini2 (192.168.5.2)"
fi

# Deploy to m4 (192.168.5.3)
if ping -c 1 192.168.5.3 > /dev/null 2>&1; then
    deploy_to_device "192.168.5.3" "georgedekker" "m4" "2"
else
    echo "⚠️  Cannot reach m4 (192.168.5.3)"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start workers:"
echo "1. On mini2: ssh mini2@192.168.5.2"
echo "   cd ~/mlx_grpc_worker && ./launch_worker.sh 1 3"
echo ""
echo "2. On m4: ssh georgedekker@192.168.5.3"
echo "   cd ~/mlx_grpc_worker && ./launch_worker.sh 2 3"
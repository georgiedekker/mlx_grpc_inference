# Distributed PyTorch Debugging Log

## Overview
Setting up distributed PyTorch inference between mini1 and mini2 over Thunderbolt network.

## Network Configuration
- mini1: 192.168.5.1 (bridge0 interface)
- mini2: 192.168.5.2 (Thunderbolt bridge)
- Basic TCP connectivity confirmed working

## Key Learnings

### 1. Network Interface Issues
- **Problem**: PyTorch Gloo backend couldn't find the right network interface
- **Symptom**: `Unable to find address for: en0`
- **Fix**: Don't specify `GLOO_SOCKET_IFNAME`, let PyTorch auto-detect

### 2. Port Conflicts
- **Problem**: Previous processes holding onto ports
- **Symptom**: Process hangs at initialization
- **Fix**: Always kill old processes, use unique ports

### 3. File Sync Issues
- **Problem**: Wrong directory paths on mini2
- **Symptom**: Scripts not found, wrong locations
- **Fix**: Ensure correct paths: `/Users/mini2/Movies/mlx_grpc_inference`

### 4. Environment Issues
- **Problem**: uv not in PATH when using SSH
- **Fix**: Use full path: `/Users/mini2/.local/bin/uv`

### 5. Port Mismatch
- **Problem**: Different ports in launch script
- **Symptom**: Timeout waiting for clients
- **Fix**: Ensure MASTER_PORT is same for all ranks

## Working Configuration
```bash
# Network
MASTER_ADDR=192.168.5.1
MASTER_PORT=29501

# Distributed init
Rank 0: tcp://0.0.0.0:29501  # Binds to all interfaces
Rank 1: tcp://192.168.5.1:29501  # Connects to master
```

## Clean Implementation Plan
1. Single launch.sh script ✓
2. Single server.py that handles both coordinator and worker roles ✓
3. Simple worker.py for additional nodes ✓
4. Use MLX adapter for model loading ✓
5. Clear logging and error handling ✓

## Final Implementation
- **server.py**: Unified server handling both coordinator (rank 0) and worker roles
- **worker.py**: Simple wrapper that calls server.py with worker role
- **launch.sh**: Clean launcher with proper error handling and status checks
- **File sync**: Ensures correct paths on mini2 (/Users/mini2/Movies/mlx_grpc_inference)
- **Port management**: Uses 29501 consistently, cleans up stale connections
- **Logging**: Comprehensive logging with hostname and rank identification

## Usage
```bash
./launch.sh start     # Start distributed inference
./launch.sh status    # Check service status
./launch.sh logs      # Show logs
./launch.sh test      # Test inference API
./launch.sh stop      # Stop all services
```
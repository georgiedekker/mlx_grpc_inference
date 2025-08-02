# MLX Distributed Inference Deployment Guide

## Overview

This guide provides step-by-step procedures for deploying the MLX distributed inference system across multiple Apple Silicon devices. The system uses gRPC for communication and requires proper DNS resolution for .local hostnames.

## Prerequisites

### Hardware Requirements
- 3 Apple Silicon devices (M1/M2/M4) with at least 16GB RAM each
- Stable network connection between all devices (preferably Ethernet)
- SSH access configured between devices

### Software Requirements
- macOS with Xcode Command Line Tools installed
- UV package manager (recommended) or Python 3.11+
- SSH keys configured for remote access

### Network Requirements
- All devices on the same local network
- mDNS/Bonjour working for .local hostname resolution
- Firewall allowing gRPC communication on port 50051
- API access on port 8100 (coordinator only)

## Initial System Setup

### 1. Install UV Package Manager

On all devices, install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
uv --version
```

### 2. Configure SSH Access

Set up passwordless SSH between devices. On the coordinator (mini1):

```bash
# Generate SSH key if not exists
ssh-keygen -t rsa -b 4096 -C "mlx_cluster"

# Copy public key to mini2
ssh-copy-id mini2@mini2.local

# For master device with specific key
ssh-copy-id -i ~/.ssh/mlx_master_key.pub georgedekker@master.local
```

Test connectivity:
```bash
ssh mini2.local "echo 'Connection successful'"
ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "echo 'Connection successful'"
```

### 3. DNS Resolution Test

Verify .local hostname resolution works:
```bash
ping -c 1 mini2.local
ping -c 1 master.local
```

If DNS resolution fails, see the Troubleshooting section.

## Project Deployment

### 1. Copy Project to All Devices

On mini1 (coordinator):
```bash
cd /Users/mini1/Movies/mlx_grpc_inference
```

Copy to mini2:
```bash
rsync -av --exclude='.venv' --exclude='logs' --exclude='__pycache__' \
  /Users/mini1/Movies/mlx_grpc_inference/ \
  mini2.local:/Users/mini2/Movies/mlx_grpc_inference/
```

Copy to master:
```bash
rsync -av --exclude='.venv' --exclude='logs' --exclude='__pycache__' \
  -e "ssh -i ~/.ssh/mlx_master_key" \
  /Users/mini1/Movies/mlx_grpc_inference/ \
  georgedekker@master.local:/Users/georgedekker/Movies/mlx_grpc_inference/
```

### 2. Set Up Virtual Environments with UV

On each device, run:
```bash
cd ~/Movies/mlx_grpc_inference

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

Verify MLX installation:
```bash
python -c "import mlx.core as mx; print('MLX available:', hasattr(mx, 'array'))"
```

### 3. Generate Protocol Buffers

On each device:
```bash
cd ~/Movies/mlx_grpc_inference
chmod +x protos/generate_protos.sh
./protos/generate_protos.sh
```

### 4. Verify Configuration

Check cluster configuration:
```bash
cat config/cluster_config.yaml
```

Ensure device hostnames and network settings are correct.

## Starting the Cluster

### Option 1: Automated Start (Recommended)

From mini1 (coordinator):
```bash
cd /Users/mini1/Movies/mlx_grpc_inference
./scripts/start_cluster.sh
```

This script will:
1. Start workers on remote devices via SSH
2. Wait for workers to initialize
3. Start the coordinator API server locally
4. Verify cluster health

### Option 2: Manual Start

If automated start fails, use manual startup:

**Step 1: Start workers first**

On mini2:
```bash
cd ~/Movies/mlx_grpc_inference
source .venv/bin/activate
python -m src.worker.worker_server --config config/cluster_config.yaml
```

On master:
```bash
cd ~/Movies/mlx_grpc_inference
source .venv/bin/activate
python -m src.worker.worker_server --config config/cluster_config.yaml
```

**Step 2: Start coordinator**

On mini1:
```bash
cd ~/Movies/mlx_grpc_inference
source .venv/bin/activate
python -m src.coordinator.api_server --host 0.0.0.0 --port 8100
```

## Testing and Validation

### 1. Basic Health Check
```bash
curl http://localhost:8100/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device_id": "mini1",
  "role": "coordinator",
  "workers_connected": 2,
  "timestamp": 1234567890
}
```

### 2. Cluster Status Check
```bash
curl http://localhost:8100/cluster/status | jq .
```

### 3. DNS Resolution Test
```bash
cd /Users/mini1/Movies/mlx_grpc_inference
python src/communication/dns_resolver.py
```

### 4. Comprehensive Test Suite
```bash
cd /Users/mini1/Movies/mlx_grpc_inference
source .venv/bin/activate
python scripts/test_inference.py
```

### 5. GPU Activity Monitoring
```bash
sudo python scripts/monitor_gpus.py
```

### 6. Inference Test
```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-1.7B-8bit",
    "messages": [{"role": "user", "content": "What is distributed computing?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Monitoring and Logging

### Log Locations
- Coordinator: `logs/coordinator.log`
- Worker mini2: `logs/worker_mini2.log` (local), check remote logs via SSH
- Worker master: `logs/worker_master.log` (local), check remote logs via SSH

### Real-time Monitoring
```bash
# Watch coordinator logs
tail -f logs/coordinator.log

# Watch all local logs
tail -f logs/*.log

# Check remote worker logs
ssh mini2.local "tail -f /Users/mini2/Movies/mlx_grpc_inference/logs/worker_mini2.log"
ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "tail -f /Users/georgedekker/Movies/mlx_grpc_inference/logs/worker_master.log"
```

### Process Management
```bash
# Check running processes
ps aux | grep "worker_server\|api_server"

# Check process PIDs
cat logs/coordinator.pid

# Monitor system resources
htop
```

## Stopping the Cluster

### Automated Stop
```bash
./scripts/stop_cluster.sh
```

### Manual Stop
```bash
# Stop coordinator
kill $(cat logs/coordinator.pid) 2>/dev/null || true

# Stop remote workers
ssh mini2.local "pkill -f 'python -m src.worker.worker_server' || true"
ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "pkill -f 'python -m src.worker.worker_server' || true"
```

## Troubleshooting

### DNS Resolution Issues

**Problem**: `.local` hostnames not resolving
```
DNS resolution failed for mini2.local:50051: Domain name not found
```

**Solutions**:

1. **Check mDNS/Bonjour service**:
   ```bash
   sudo launchctl list | grep mdns
   dns-sd -B _services._dns-sd._udp local.
   ```

2. **Restart mDNS responder**:
   ```bash
   sudo dscacheutil -flushcache
   sudo killall -HUP mDNSResponder
   ```

3. **Add manual DNS entries** (temporary fix):
   ```bash
   # Add to /etc/hosts on all devices
   echo "192.168.2.15 mini2.local" | sudo tee -a /etc/hosts
   echo "192.168.2.106 master.local" | sudo tee -a /etc/hosts
   ```

4. **Test with IP addresses**:
   ```bash
   # Find IP addresses
   ping mini2.local
   ping master.local
   
   # Update cluster_config.yaml with IP addresses instead of hostnames
   ```

### gRPC Connection Issues

**Problem**: gRPC connections failing
```
status = StatusCode.UNAVAILABLE
details = "failed to connect to all addresses"
```

**Solutions**:

1. **Check firewall settings**:
   ```bash
   # Temporarily disable firewall for testing
   sudo pfctl -d
   
   # Or add specific rules
   sudo pfctl -f /etc/pf.conf
   ```

2. **Verify port accessibility**:
   ```bash
   # Test from coordinator
   nc -zv mini2.local 50051
   nc -zv master.local 50051
   ```

3. **Check worker processes**:
   ```bash
   ssh mini2.local "ps aux | grep worker_server"
   ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "ps aux | grep worker_server"
   ```

### Model Loading Issues

**Problem**: Model fails to load or download
```
FileNotFoundError: Model not found
```

**Solutions**:

1. **Check internet connectivity**:
   ```bash
   curl -I https://huggingface.co
   ```

2. **Clear model cache**:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

3. **Check disk space**:
   ```bash
   df -h
   ```

4. **Test model loading manually**:
   ```bash
   python -c "
   from mlx_lm import load
   model, tokenizer = load('mlx-community/Qwen3-1.7B-8bit')
   print('Model loaded successfully')
   "
   ```

### Performance Issues

**Problem**: Slow inference or high latency

**Solutions**:

1. **Check network latency**:
   ```bash
   ping mini2.local
   ping master.local
   ```

2. **Monitor GPU utilization**:
   ```bash
   sudo python scripts/monitor_gpus.py
   ```

3. **Check memory usage**:
   ```bash
   ps aux | grep python
   top -o mem
   ```

4. **Optimize network settings**:
   - Use wired Ethernet instead of WiFi
   - Enable tensor compression in config
   - Adjust batch size and sequence length

### SSH Connection Issues

**Problem**: Cannot connect to remote devices

**Solutions**:

1. **Test basic SSH connectivity**:
   ```bash
   ssh -v mini2.local
   ssh -v -i ~/.ssh/mlx_master_key georgedekker@master.local
   ```

2. **Check SSH key permissions**:
   ```bash
   chmod 600 ~/.ssh/mlx_master_key
   chmod 644 ~/.ssh/mlx_master_key.pub
   ```

3. **Verify SSH service on remote devices**:
   ```bash
   # On remote devices
   sudo systemsetup -getremotelogin
   sudo systemsetup -setremotelogin on
   ```

### Worker Health Check Failures

**Problem**: Workers appear disconnected in cluster status

**Solutions**:

1. **Check worker logs**:
   ```bash
   ssh mini2.local "tail -20 /Users/mini2/Movies/mlx_grpc_inference/logs/worker_mini2.log"
   ```

2. **Restart specific worker**:
   ```bash
   ssh mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && pkill -f worker_server && nohup python -m src.worker.worker_server --config config/cluster_config.yaml > logs/worker_mini2.log 2>&1 &"
   ```

3. **Verify worker configuration**:
   ```bash
   ssh mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && cat config/cluster_config.yaml"
   ```

## Known Limitations

### Current System Constraints

1. **Network Dependencies**:
   - Requires stable local network with mDNS support
   - `.local` hostname resolution can be unreliable across networks
   - No automatic failover for network partitions

2. **Model Distribution**:
   - Static layer assignment (no dynamic load balancing)
   - All devices must have the same model architecture
   - No support for heterogeneous device capabilities

3. **Error Handling**:
   - Limited retry mechanisms for failed connections
   - No automatic worker recovery
   - Manual intervention required for most failures

4. **Security**:
   - No authentication for API endpoints
   - Unencrypted gRPC communication
   - SSH keys required for cluster management

5. **Scalability**:
   - Fixed 3-device configuration
   - No support for adding/removing devices at runtime
   - Linear scaling only

6. **Monitoring**:
   - Basic health checks only
   - Limited performance metrics
   - No centralized logging

## Recovery Procedures

### Complete Cluster Recovery

If the entire cluster becomes unresponsive:

1. **Stop all processes**:
   ```bash
   ./scripts/stop_cluster.sh
   
   # If script fails, manual cleanup:
   pkill -f "python -m src"
   ssh mini2.local "pkill -f 'python -m src'"
   ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "pkill -f 'python -m src'"
   ```

2. **Clear logs and state**:
   ```bash
   rm -f logs/*.log logs/*.pid
   ```

3. **Restart cluster**:
   ```bash
   sleep 10
   ./scripts/start_cluster.sh
   ```

### Single Worker Recovery

To recover a single failed worker:

1. **Identify failed worker**:
   ```bash
   curl http://localhost:8100/cluster/status | jq '.workers[] | select(.status != "connected")'
   ```

2. **Restart worker on remote device**:
   ```bash
   # For mini2
   ssh mini2.local "cd /Users/mini2/Movies/mlx_grpc_inference && pkill -f worker_server || true && source .venv/bin/activate && nohup python -m src.worker.worker_server --config config/cluster_config.yaml > logs/worker_mini2.log 2>&1 &"
   
   # For master
   ssh -i ~/.ssh/mlx_master_key georgedekker@master.local "cd /Users/georgedekker/Movies/mlx_grpc_inference && pkill -f worker_server || true && source .venv/bin/activate && nohup python -m src.worker.worker_server --config config/cluster_config.yaml > logs/worker_master.log 2>&1 &"
   ```

3. **Verify recovery**:
   ```bash
   sleep 5
   curl http://localhost:8100/cluster/status
   ```

### Configuration Recovery

If configuration becomes corrupted:

1. **Backup current config**:
   ```bash
   cp config/cluster_config.yaml config/cluster_config.yaml.backup
   ```

2. **Reset to defaults** (update hostnames/IPs as needed):
   ```bash
   git checkout config/cluster_config.yaml
   ```

3. **Update network settings**:
   ```bash
   # Test DNS resolution
   python src/communication/dns_resolver.py
   
   # If DNS fails, update config with IP addresses
   sed -i '' 's/mini2.local/192.168.2.15/g' config/cluster_config.yaml
   sed -i '' 's/master.local/192.168.2.106/g' config/cluster_config.yaml
   ```

### Environment Recovery

To fix corrupted Python environments:

1. **Remove old environment**:
   ```bash
   rm -rf .venv
   ```

2. **Recreate with UV**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Regenerate protocol buffers**:
   ```bash
   ./protos/generate_protos.sh
   ```

4. **Sync to remote devices**:
   ```bash
   rsync -av --exclude='logs' .venv/ mini2.local:/Users/mini2/Movies/mlx_grpc_inference/.venv/
   rsync -av --exclude='logs' -e "ssh -i ~/.ssh/mlx_master_key" .venv/ georgedekker@master.local:/Users/georgedekker/Movies/mlx_grpc_inference/.venv/
   ```

## Performance Optimization

### Network Optimization
1. **Use wired Ethernet connections** for best performance and reliability
2. **Enable jumbo frames** if supported by network infrastructure
3. **Configure QoS** to prioritize gRPC traffic on port 50051

### Model Loading Optimization
1. **Pre-download models** to all devices to avoid download latency
2. **Use local model cache** to speed up subsequent loads
3. **Consider model quantization** for reduced memory usage

### System Optimization
1. **Increase batch size** in configuration for better throughput
2. **Adjust max_sequence_length** based on use case
3. **Enable tensor compression** for slower networks
4. **Monitor memory usage** and adjust layer distribution if needed

### Monitoring Optimization
1. **Set up centralized logging** for easier debugging
2. **Enable metrics export** to monitor performance trends
3. **Use automated health checks** to detect issues early

## Security Considerations

### Production Deployment
- Add API key authentication for public endpoints
- Use TLS encryption for gRPC communication
- Implement proper firewall rules
- Regular security updates for all dependencies
- Restrict network access to trusted devices only

### SSH Security
- Use dedicated SSH keys for cluster management
- Regularly rotate SSH keys
- Implement proper key management
- Monitor SSH access logs

This deployment guide provides a comprehensive framework for successfully deploying and maintaining the MLX distributed inference system. Follow the procedures carefully and refer to the troubleshooting section when issues arise.
# Phase 5 Testing Guide: Production-Ready Distributed MLX Inference

## 🚀 Overview
This guide addresses critical production concerns for deploying the distributed MLX inference system across mini1, mini2, and master devices.

## 1. Deployment & Environment Validation

### Container-Based Deployment
```dockerfile
# Dockerfile.mlx
FROM python:3.11-slim

# Install MLX dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY . /app
WORKDIR /app

# Expose ports
EXPOSE 8100-8102 9100-9102

CMD ["uv", "run", "python", "distributed_server.py"]
```

### Environment Verification Script
```bash
#!/bin/bash
# verify_environment.sh

echo "🔍 Verifying MLX Environment..."

# Check Python version
python --version

# Check MLX installation
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"

# Check gRPC
python -c "import grpc; print(f'gRPC version: {grpc.__version__}')"

# Verify GPU access
python -c "
import mlx.core as mx
print(f'Default device: {mx.default_device()}')
print(f'Metal available: {mx.metal.is_available() if hasattr(mx, \"metal\") else False}')
"

# Test layer assignments
uv run python model_splitter.py
```

## 2. Performance Monitoring & Metrics

### Install Monitoring Tools
```bash
# Install asitop for Apple Silicon monitoring
brew install asitop

# Create metrics collection script
cat > collect_metrics.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import json
import time
from prometheus_client import Gauge, start_http_server

# Define Prometheus metrics
gpu_utilization = Gauge('mlx_gpu_utilization', 'GPU utilization percentage', ['device'])
memory_usage = Gauge('mlx_memory_usage_mb', 'Memory usage in MB', ['device'])
inference_latency = Gauge('mlx_inference_latency_ms', 'Inference latency in ms', ['device'])

def collect_asitop_metrics():
    """Collect metrics from asitop"""
    try:
        result = subprocess.run(['asitop', '--json'], capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # Extract GPU metrics
        gpu_util = data.get('gpu_utilization', 0)
        mem_usage = data.get('memory_usage_mb', 0)
        
        gpu_utilization.labels(device='local').set(gpu_util)
        memory_usage.labels(device='local').set(mem_usage)
        
    except Exception as e:
        print(f"Error collecting metrics: {e}")

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(9090)
    
    while True:
        collect_asitop_metrics()
        time.sleep(5)
EOF
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "MLX Distributed Inference",
    "panels": [
      {
        "title": "GPU Utilization by Device",
        "targets": [
          {
            "expr": "mlx_gpu_utilization",
            "legendFormat": "{{device}}"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "targets": [
          {
            "expr": "mlx_inference_latency_ms",
            "legendFormat": "{{device}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "mlx_memory_usage_mb",
            "legendFormat": "{{device}}"
          }
        ]
      }
    ]
  }
}
```

## 3. gRPC Reliability & Tuning

### Enhanced gRPC Configuration
```python
# grpc_config.py
import grpc

def create_secure_channel(target: str, max_message_size: int = 1024 * 1024 * 1024):  # 1GB
    """Create gRPC channel with production settings"""
    
    options = [
        ('grpc.max_send_message_length', max_message_size),
        ('grpc.max_receive_message_length', max_message_size),
        ('grpc.keepalive_time_ms', 10000),  # 10 seconds
        ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.enable_retries', 1),
        ('grpc.service_config', json.dumps({
            "methodConfig": [{
                "name": [{"service": "tensor_service.TensorService"}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "10s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"]
                }
            }]
        }))
    ]
    
    # Use TLS in production
    credentials = grpc.ssl_channel_credentials()
    return grpc.secure_channel(target, credentials, options=options)
```

### Health Check Implementation
```python
# health_check_service.py
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

class HealthServicer(health.HealthServicer):
    def __init__(self, distributed_engine):
        self.distributed_engine = distributed_engine
        
    async def Check(self, request, context):
        # Check if engine is healthy
        if self.distributed_engine and self.distributed_engine.is_healthy():
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        else:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )
```

## 4. Fault-Tolerance Testing

### Chaos Testing Script
```python
#!/usr/bin/env python3
# chaos_test.py
import asyncio
import random
import subprocess
import time
import logging

logger = logging.getLogger(__name__)

class ChaosTestRunner:
    def __init__(self, devices):
        self.devices = devices
        self.test_results = []
        
    async def simulate_device_failure(self, device_id: str, duration: int = 30):
        """Simulate device failure by killing gRPC process"""
        logger.info(f"🔥 Simulating failure on {device_id} for {duration}s")
        
        # Kill gRPC process
        subprocess.run(f"pkill -f 'grpc.*{device_id}'", shell=True)
        
        # Wait
        await asyncio.sleep(duration)
        
        # Restart service
        logger.info(f"🔧 Restarting {device_id}")
        # Restart logic here
        
    async def simulate_network_partition(self, device_id: str, duration: int = 20):
        """Simulate network partition using firewall rules"""
        logger.info(f"🔌 Simulating network partition for {device_id}")
        
        # Block traffic (macOS pfctl)
        subprocess.run(f"sudo pfctl -t blocklist -T add {device_id}.local", shell=True)
        
        await asyncio.sleep(duration)
        
        # Restore
        subprocess.run(f"sudo pfctl -t blocklist -T delete {device_id}.local", shell=True)
        
    async def test_fallback_behavior(self):
        """Test coordinator fallback to local processing"""
        logger.info("🧪 Testing fallback behavior...")
        
        # Make inference request
        start_time = time.time()
        response = await self.make_inference_request("Test prompt")
        latency = time.time() - start_time
        
        self.test_results.append({
            "test": "fallback",
            "success": response is not None,
            "latency": latency
        })
        
    async def run_chaos_suite(self):
        """Run complete chaos test suite"""
        tests = [
            ("device_failure", self.simulate_device_failure("mini2", 30)),
            ("network_partition", self.simulate_network_partition("master", 20)),
            ("fallback", self.test_fallback_behavior())
        ]
        
        for test_name, test_coro in tests:
            try:
                await test_coro
                logger.info(f"✅ {test_name} completed")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                
        return self.test_results
```

## 5. Cluster Orchestration

### Kubernetes Deployment
```yaml
# mlx-distributed-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlx-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlx-coordinator
  template:
    metadata:
      labels:
        app: mlx-coordinator
    spec:
      containers:
      - name: mlx-server
        image: mlx-distributed:latest
        ports:
        - containerPort: 8100
        - containerPort: 9100
        env:
        - name: DEVICE_ID
          value: "mini1"
        - name: DEVICE_ROLE
          value: "coordinator"
        resources:
          limits:
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          grpc:
            port: 9100
            service: tensor_service.TensorService
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8100
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mlx-coordinator-service
spec:
  selector:
    app: mlx-coordinator
  ports:
  - name: http
    port: 8100
    targetPort: 8100
  - name: grpc
    port: 9100
    targetPort: 9100
  type: LoadBalancer
```

### Service Mesh Configuration (Linkerd)
```yaml
# linkerd-service-profile.yaml
apiVersion: linkerd.io/v1alpha1
kind: ServiceProfile
metadata:
  name: mlx-tensor-service
spec:
  routes:
  - name: SendTensor
    condition:
      method: POST
      pathRegex: "/tensor_service.TensorService/SendTensor"
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
  - name: ForwardPass
    condition:
      method: POST
      pathRegex: "/tensor_service.TensorService/ForwardPass"
    timeout: 60s
    retries:
      attempts: 2
      perTryTimeout: 30s
```

## 6. Security Implementation

### TLS Configuration
```python
# tls_config.py
import ssl
import grpc

def create_tls_credentials():
    """Create TLS credentials for gRPC"""
    # Read certificates
    with open('certs/ca.crt', 'rb') as f:
        ca_cert = f.read()
    with open('certs/server.crt', 'rb') as f:
        server_cert = f.read()
    with open('certs/server.key', 'rb') as f:
        server_key = f.read()
        
    # Server credentials
    server_credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True
    )
    
    # Client credentials
    client_credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=server_key,
        certificate_chain=server_cert
    )
    
    return server_credentials, client_credentials
```

### API Key Authentication
```python
# auth_middleware.py
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = "your-secure-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to endpoints
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatRequest):
    # ... existing code ...
```

### Firewall Rules (macOS)
```bash
#!/bin/bash
# setup_firewall.sh

# Create pf rules
cat > /etc/pf.anchors/mlx_distributed << EOF
# Allow gRPC traffic only from cluster nodes
pass in proto tcp from {192.168.1.100, 192.168.1.101, 192.168.1.102} to any port {9100:9102}

# Block all other gRPC traffic
block in proto tcp to any port {9100:9102}

# Allow REST API from any source (protected by API key)
pass in proto tcp to any port {8100:8102}
EOF

# Load rules
sudo pfctl -f /etc/pf.conf
sudo pfctl -e
```

## Testing Checklist

### Pre-Deployment
- [ ] Verify identical Python/MLX versions across all nodes
- [ ] Test layer assignments with model_splitter.py
- [ ] Validate gRPC connectivity between all nodes
- [ ] Confirm firewall rules are configured

### Performance Testing
- [ ] Monitor GPU utilization on all devices during inference
- [ ] Verify memory usage stays within limits
- [ ] Check tensor serialization overhead
- [ ] Measure end-to-end latency vs single-device

### Fault Tolerance
- [ ] Test device failure scenarios
- [ ] Verify fallback to local processing
- [ ] Test recovery after network partition
- [ ] Validate error propagation

### Security
- [ ] Verify TLS encryption on gRPC channels
- [ ] Test API key authentication
- [ ] Confirm firewall rules block unauthorized access
- [ ] Audit logs for security events

## Monitoring Dashboard

Access monitoring at:
- Grafana: http://mini1:3000
- Prometheus: http://mini1:9090
- Individual device stats: http://{device}:{port}/distributed-stats

## Troubleshooting

### Common Issues

1. **gRPC Connection Failures**
   - Check firewall rules
   - Verify TLS certificates
   - Test with grpc_health_probe tool

2. **Memory Issues**
   - Monitor with asitop
   - Reduce batch size
   - Enable tensor compression

3. **Slow Inference**
   - Check GPU utilization balance
   - Verify layer assignment efficiency
   - Monitor network latency

4. **Fallback Triggering**
   - Check device health endpoints
   - Review gRPC timeout settings
   - Examine coordinator logs

## Performance Benchmarks

Expected performance metrics:
- Single device: ~50 tokens/second
- 3-device distributed: ~120 tokens/second
- Network overhead: <10ms per tensor transfer
- Fallback latency: <100ms detection + switch

## Next Steps

1. Run chaos testing suite
2. Deploy monitoring stack
3. Configure auto-scaling policies
4. Set up alerting rules
5. Document runbooks for operations team
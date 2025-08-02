# Distributed MLX Inference System (v2.0 - gRPC-based)

This is a gRPC-based distributed inference system for MLX models that supports heterogeneous Apple Silicon devices with different memory and compute capabilities.

## Features

- **Heterogeneous Device Support**: Automatically adapts to devices with different RAM and GPU capabilities
- **Multiple Sharding Strategies**: Uniform, memory-proportional, compute-proportional, and balanced strategies
- **Model Abstraction**: Supports multiple model architectures (Qwen, Llama, Mistral, Phi)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Resource-Aware Distribution**: Intelligent layer distribution based on device capabilities
- **gRPC Communication**: Efficient tensor serialization with Protocol Buffers
- **Health Monitoring**: Automatic health checks and failure detection
- **Easy Configuration**: JSON/YAML configuration files

## Architecture

The system uses a pipeline parallelism approach where model layers are distributed across devices:

```
Device 1 (Coordinator)     Device 2 (Worker)       Device 3 (Worker)
[Embedding + Layers 0-9]   [Layers 10-19]         [Layers 20-27 + LM Head]
        |                        |                        |
        ├───── gRPC ────────────>├───── gRPC ────────────>│
        │                        │                        │
   OpenAI API                Forward Pass            Final Output
```

## Quick Start

### 1. Install Dependencies

```bash
uv pip install grpcio grpcio-tools protobuf fastapi uvicorn pydantic pyyaml mlx-lm
```

### 2. Generate Protocol Buffers

```bash
./generate_proto.sh
```

### 3. Create Configuration

Create a `distributed_config.json` file:

```json
{
  "devices": [
    {
      "device_id": "mini1",
      "hostname": "mini1.local",
      "port": 50051,
      "role": "coordinator"
    },
    {
      "device_id": "mini2",
      "hostname": "mini2.local",
      "port": 50051,
      "role": "worker"
    }
  ],
  "model": {
    "name": "Qwen3-1.7B-8bit",
    "provider": "mlx-community"
  },
  "sharding": {
    "strategy": "balanced"
  }
}
```

### 4. Launch the Cluster

```bash
# Launch entire cluster (all devices + API server)
./launch_cluster.sh

# Or launch components individually:
# On worker device:
./launch_grpc_server.sh --device-id mini2 --port 50051

# On coordinator device:
./launch_grpc_server.sh --device-id mini1 --port 50051
./launch_distributed_api.sh --config distributed_config.json
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Cluster status
curl http://localhost:8000/cluster/status

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-1.7B-8bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Configuration Options

### Sharding Strategies

- **uniform**: Distribute layers equally across devices
- **memory_proportional**: Distribute based on available memory
- **compute_proportional**: Distribute based on GPU cores
- **balanced**: Consider both memory and compute (recommended)
- **custom**: Specify exact proportions for each device

### Device Roles

- **coordinator**: Runs the API server and coordinates inference
- **worker**: Participates in distributed inference

### Example: Heterogeneous Setup

For a Mac Studio (48GB) + Mac Mini (16GB):

```json
{
  "devices": [
    {
      "device_id": "studio",
      "hostname": "studio.local",
      "port": 50051,
      "role": "coordinator",
      "capabilities": {
        "memory_gb": 48,
        "gpu_cores": 30
      }
    },
    {
      "device_id": "mini",
      "hostname": "mini.local",
      "port": 50051,
      "role": "worker",
      "capabilities": {
        "memory_gb": 16,
        "gpu_cores": 10
      }
    }
  ],
  "sharding": {
    "strategy": "memory_proportional"
  }
}
```

## Monitoring

### Logs

All logs are stored in the `logs/` directory:
- `api_server.log`: API server logs
- `{device_id}_server.log`: gRPC server logs for each device

### Metrics

Access cluster metrics at:
- `/health`: Basic health check
- `/cluster/status`: Detailed cluster status including device assignments

## Stopping the Cluster

```bash
./stop_cluster.sh
```

## Advanced Usage

### Custom Sharding

Specify exact layer distribution:

```json
{
  "sharding": {
    "strategy": "custom",
    "custom_proportions": [0.7, 0.3]
  }
}
```

### Pre-sharding Models

For faster startup, you can pre-shard models (coming soon):

```bash
python preshard_model.py --model mlx-community/Qwen3-1.7B-8bit --output shards/
```

### Adding New Model Support

1. Create a new wrapper class in `model_abstraction.py`
2. Implement required methods: `load_model()`, `get_model_info()`, `create_shards()`
3. Register in `ModelFactory.MODEL_PATTERNS`

## Troubleshooting

### SSH Connection Issues

Ensure passwordless SSH is set up between devices:

```bash
ssh-copy-id mini2@mini2.local
```

### Port Already in Use

Check and kill existing processes:

```bash
lsof -i :50051
kill <PID>
```

### Memory Issues

Adjust the sharding strategy or reduce max recommended model size:

```python
# In device_capabilities.py
self.max_recommended_model_size_gb = self.memory_gb * 0.7  # Use 70% instead of 80%
```

## Performance Tips

1. Use Thunderbolt/10GbE connections between devices for best performance
2. Place embedding/LM head on the device with fastest storage
3. Use `memory_proportional` strategy for heterogeneous setups
4. Enable compression for slow networks (coming soon)

## Files Overview (New gRPC Implementation)

### Core Components
- `protos/distributed_inference.proto`: Protocol Buffer definitions
- `grpc_server.py`: gRPC server for each device
- `grpc_client.py`: Client for coordinating distributed inference
- `distributed_openai_api.py`: OpenAI-compatible API server

### Configuration & Management
- `distributed_config_v2.py`: Enhanced configuration system
- `device_capabilities.py`: Automatic device profiling
- `sharding_strategy.py`: Intelligent layer distribution algorithms
- `model_abstraction.py`: Support for multiple model architectures

### Launch Scripts
- `launch_cluster.sh`: Launch entire cluster
- `launch_grpc_server.sh`: Launch individual gRPC server
- `launch_distributed_api.sh`: Launch API server
- `stop_cluster.sh`: Stop all components

## Comparison with MPI-based System (v1.0)

| Feature | gRPC-based (v2.0) | MPI-based (v1.0) |
|---------|-------------------|------------------|
| Heterogeneous Support | ✅ Excellent | ❌ Limited |
| Setup Complexity | ✅ Simple | ❌ Complex |
| Debugging | ✅ Easy | ❌ Difficult |
| Performance | ✅ Good | ✅ Good |
| Fault Tolerance | ✅ Built-in | ❌ Limited |
| Dynamic Scaling | ✅ Possible | ❌ Not supported |

## Future Enhancements

- [ ] Model pre-sharding and caching
- [ ] Automatic device discovery
- [ ] Sub-layer sharding for very large models
- [ ] Quantization-aware sharding
- [ ] Pipeline parallelism optimizations
- [ ] Multi-model serving
- [ ] Request batching
- [ ] Prometheus metrics export

## Contributing

Contributions are welcome! Please focus on:
- Supporting more model architectures
- Improving sharding algorithms
- Adding monitoring/observability
- Performance optimizations
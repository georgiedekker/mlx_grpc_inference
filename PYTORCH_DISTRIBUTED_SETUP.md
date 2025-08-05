# PyTorch Distributed Inference Setup

## Overview

This implementation provides distributed inference across two Mac minis using PyTorch with the Gloo backend. It works around macOS-specific issues and provides an OpenAI-compatible API.

## Architecture

```
Mac mini 1 (192.168.5.1)          Mac mini 2 (192.168.5.2)
┌─────────────────────┐           ┌─────────────────────┐
│ Rank 0 (Master)     │           │ Rank 1 (Worker)     │
├─────────────────────┤           ├─────────────────────┤
│ - API Server (:8100)│           │ - Worker Process    │
│ - Model Layers 0-N/2│←─────────→│ - Model Layers N/2-N│
│ - Embeddings        │   Gloo    │ - Final Norm        │
│ - Orchestration     │   TCP     │ - LM Head           │
└─────────────────────┘           └─────────────────────┘
      Thunderbolt Bridge Network (192.168.5.0/24)
```

## Components

### 1. Core Files

- `pytorch_distributed_server.py` - Distributed model sharding and inference engine
- `pytorch_api_server.py` - FastAPI server with OpenAI-compatible endpoints
- `launch_pytorch_api.sh` - Production launch script

### 2. Test Files

- `test_torch_simple_model.py` - Tests distributed forward pass with tiny model
- `test_torch_distributed_fixed.py` - Comprehensive distributed communication tests
- `launch_torch_working.sh` - Test launcher for distributed components

### 3. Utilities

- `test_local_distributed.sh` - Test PyTorch distributed on single machine
- `test_pytorch.py` - Verify PyTorch and MPS installation

## Setup Instructions

### 1. Install Dependencies (Both Machines)

On mini1:
```bash
uv pip install torch torchvision transformers accelerate
```

On mini2:
```bash
pip3 install --user torch torchvision transformers accelerate
```

### 2. Network Configuration

Ensure Thunderbolt network is configured:
- mini1: 192.168.5.1
- mini2: 192.168.5.2
- MTU: 9000 (jumbo frames)

### 3. Launch Distributed Inference

```bash
# Start with small model for testing
MODEL_NAME="microsoft/phi-2" ./launch_pytorch_api.sh

# Or use larger model
MODEL_NAME="meta-llama/Llama-2-7b-hf" ./launch_pytorch_api.sh
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8100/health | jq

# Chat completion
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! Can you explain what you are?"}
    ],
    "max_tokens": 100
  }' | jq
```

## Key Features

1. **Model Sharding**: Automatically splits transformer layers across devices
2. **Pipeline Parallelism**: Forward passes flow through ranks sequentially
3. **OpenAI Compatibility**: Drop-in replacement for OpenAI API
4. **MPS Support**: Uses Apple Silicon GPU when available
5. **Robust Error Handling**: Graceful failures and detailed logging

## Troubleshooting

### Common Issues

1. **"Connection refused" errors**
   - Ensure no firewall blocking port 12355
   - Check that master starts before worker
   - Verify network connectivity with ping

2. **"Cannot find address for interface" errors**
   - Don't set GLOO_SOCKET_IFNAME on macOS
   - Let Gloo auto-detect interfaces

3. **Model loading failures**
   - Ensure enough disk space for model
   - Check Hugging Face token if using private models
   - Verify transformers library is installed

### Debug Commands

```bash
# Check if processes are running
ps aux | grep pytorch

# View logs
tail -f logs/api_master_*.log

# Test network
ping 192.168.5.2

# Check ports
lsof -i :12355
lsof -i :8100
```

## Performance Considerations

1. **Network Bandwidth**: Thunderbolt provides ~10-20 Gbps
2. **Latency**: Add ~1-2ms per forward pass for network communication
3. **Memory**: Each device needs RAM for its model shard + activations
4. **Optimization**: Consider batch processing for better throughput

## Limitations

1. Gloo backend doesn't support MPS directly (CPU for communication)
2. No dynamic batching in current implementation
3. Sequential pipeline (no pipeline parallelism overlap)
4. Requires manual model selection (no automatic routing)

## Future Improvements

1. Implement tensor parallelism for larger models
2. Add streaming support for real-time generation
3. Implement dynamic batching for better throughput
4. Add model quantization support
5. Create Kubernetes/Docker deployment configs

## Comparison with MLX

| Feature | MLX (Attempted) | PyTorch (Working) |
|---------|----------------|-------------------|
| Distributed Support | Limited/Experimental | Mature |
| macOS Integration | Native | Good (MPS support) |
| Model Ecosystem | Growing | Extensive |
| Performance | Potentially better | Good |
| Documentation | Limited | Comprehensive |

## Conclusion

While MLX showed promise for Apple Silicon optimization, its distributed capabilities are not yet mature enough for production use. PyTorch with Gloo backend provides a working solution for distributed inference across Mac minis, enabling you to run models that don't fit on a single device.

The implementation successfully demonstrates:
- ✅ Working distributed inference across 2 Mac minis
- ✅ Model sharding and pipeline parallelism
- ✅ OpenAI-compatible API
- ✅ Production-ready deployment scripts
- ✅ Comprehensive error handling and logging

This solution can be extended to support larger models, more devices, and additional optimizations as needed.
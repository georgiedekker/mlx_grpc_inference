# MLX Distributed Inference System

A clean, production-ready distributed inference system for MLX models across Apple Silicon devices using gRPC.

## Features

- Distributed inference across 3 Apple Silicon devices (M4)
- Model sharding with intelligent layer distribution
- gRPC-based tensor communication
- OpenAI-compatible REST API
- Real-time GPU monitoring
- Fault tolerance and graceful degradation

## Quick Start

1. Install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

2. Configure cluster:
```bash
cp config/cluster_config.yaml.example config/cluster_config.yaml
# Edit with your device settings
```

3. Start the cluster:
```bash
./scripts/start_cluster.sh
```

4. Test inference:
```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-1.7B-8bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Architecture

See [ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md) for detailed system design.

## Hardware Setup

- **Device 1 (Coordinator)**: mini1.local - Apple M4, 16GB RAM
- **Device 2 (Worker)**: mini2.local - Apple M4, 16GB RAM  
- **Device 3 (Worker)**: master.local - Apple M4, 16GB RAM
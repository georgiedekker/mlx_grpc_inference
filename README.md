# MLX Native Distributed Inference

Clean implementation of distributed inference using MLX's native pipeline parallelism across Thunderbolt-connected M4 Mac minis.

## Setup

1. Install dependencies:
```bash
pip install mlx mlx-lm mpi4py fastapi uvicorn
```

2. Ensure SSH access to mini2:
```bash
ssh-copy-id 192.168.5.2
```

3. Install MPI on both machines:
```bash
brew install open-mpi
```

## Usage

Start the distributed server:
```bash
./launch.sh
```

Test the API:
```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Files

- `server.py` - MLX native distributed inference server with accurate token metrics
- `launch.sh` - Launch script that sets up and runs distributed inference
- `pyproject.toml` - Python dependencies

## Configuration

- Model: `mlx-community/Qwen3-1.7B-8bit` (default)
- Devices: mini1 (192.168.5.1) + mini2 (192.168.5.2)
- Port: 8100

## Features

- OpenAI-compatible API
- Accurate token metrics (separate prompt vs generation timing)
- Pipeline parallelism across devices
- GPU acceleration on both M4 Mac minis
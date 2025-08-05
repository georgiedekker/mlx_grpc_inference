# Deploy Workers on mini2 and m4

## Prerequisites
1. Ensure the mlx_grpc_inference repository is cloned on each device
2. Install dependencies: `uv pip install -r requirements.txt`
3. Verify Thunderbolt connection (ping 192.168.5.1 from each device)

## On mini2 (192.168.5.2)

1. Navigate to the project directory:
   ```bash
   cd ~/Movies/mlx_grpc_inference
   ```

2. Start the worker (it will be worker 1 of 3):
   ```bash
   ./launch_worker.sh 1 3
   ```

## On m4 (192.168.5.3)

1. Navigate to the project directory:
   ```bash
   cd ~/Movies/mlx_grpc_inference
   ```

2. Start the worker (it will be worker 2 of 3):
   ```bash
   ./launch_worker.sh 2 3
   ```

## Verify Workers are Running

On mini1, test connectivity:
```bash
uv run python test_grpc_connectivity.py
```

All three connections should show as "healthy".

## Test Distributed Inference

Once all workers are running, test the complete system:
```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 20,
    "temperature": 0.7
  }' | jq
```
# Manual Distributed Launch Instructions

## Option 1: Using mpirun (Recommended)

In one terminal on mini1:
```bash
./launch_simple.sh
```

## Option 2: Manual Start on Each Machine

### Terminal 1 (on mini1):
```bash
# First, copy files to mini2
scp server_moe.py shard.py base.py qwen_moe_mini.py mini2@192.168.5.2:/Users/mini2/Movies/mlx_grpc_inference/

# Then start both ranks with mpirun
mpirun -n 2 -host mini1,mini2 python3 server_moe.py --mca btl_tcp_if_include 192.168.5.0/24
```

## Option 3: Separate Terminal Sessions

### On mini1 (Terminal 1):
```bash
mpirun -n 1 python3 server_moe.py
```

### On mini2 (Terminal 2 - SSH into mini2):
```bash
ssh mini2@192.168.5.2
cd /Users/mini2/Movies/mlx_grpc_inference
mpirun -n 1 python3 server_moe.py
```

## Testing the API

Once running, test with:
```bash
curl http://localhost:8100/
```

Or:
```bash
python3 test_moe_api.py
```

## What You Should See

When working correctly:
- mini1: "Rank 0/2" - API server on port 8100, handling layers 0-7
- mini2: "Rank 1/2" - Worker, handling layers 8-15

## Troubleshooting

If you see "Rank 0/1" on both, they're not connecting. Try:

1. Check network:
```bash
ping 192.168.5.2  # From mini1
ping 192.168.5.1  # From mini2
```

2. Test MPI connectivity:
```bash
mpirun -n 1 -host mini1 hostname : -n 1 -host mini2 hostname
```

3. Use IP addresses instead of hostnames:
```bash
mpirun -n 1 -host 192.168.5.1 python3 server_moe.py : \
       -n 1 -host 192.168.5.2 python3 server_moe.py
```
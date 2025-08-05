#!/bin/bash
# Test PyTorch distributed with different network interfaces

echo "Testing PyTorch distributed with different interfaces..."

# Clean up
pkill -f "python.*simple_torch_test" || true
sleep 1

# Test 1: Default (no interface specified)
echo -e "\n=== Test 1: Default (no GLOO_SOCKET_IFNAME) ==="
RANK=0 WORLD_SIZE=1 uv run python -c "
import torch.distributed as dist
import socket
print(f'Hostname: {socket.gethostname()}')
print(f'Testing single-node init...')
dist.init_process_group('gloo', init_method='tcp://127.0.0.1:12355', rank=0, world_size=1)
print('✅ Single-node init successful!')
dist.destroy_process_group()
"

# Test 2: With en0
echo -e "\n=== Test 2: With GLOO_SOCKET_IFNAME=en0 ==="
GLOO_SOCKET_IFNAME=en0 RANK=0 WORLD_SIZE=1 uv run python -c "
import torch.distributed as dist
import os
print(f'GLOO_SOCKET_IFNAME: {os.environ.get(\"GLOO_SOCKET_IFNAME\")}')
dist.init_process_group('gloo', init_method='tcp://127.0.0.1:12356', rank=0, world_size=1)
print('✅ Init with en0 successful!')
dist.destroy_process_group()
"

# Test 3: List available IPs
echo -e "\n=== Available IPs ==="
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
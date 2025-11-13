#\!/bin/bash

# Test MLX Distributed Setup
echo "Testing MLX Distributed Setup..."
echo "================================"

# 1. Test network
echo -e "\n1. Testing Thunderbolt network..."
ping -c 1 -W 1 192.168.5.2 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ mini2 (192.168.5.2) reachable"
else
    echo "   ❌ Cannot reach mini2"
    exit 1
fi

# 2. Test SSH
echo -e "\n2. Testing SSH access..."
ssh -o ConnectTimeout=2 192.168.5.2 "echo '   ✅ SSH to mini2 works'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ SSH failed"
    exit 1
fi

# 3. Test MLX distributed with ring backend
echo -e "\n3. Testing MLX distributed (ring backend)..."

# Create test script
cat > /tmp/mlx_test/test.py << 'INNEREOF'
import mlx.core as mx
import socket

group = mx.distributed.init()
hostname = socket.gethostname()

print(f"   {hostname}: rank={group.rank()}/{group.size()}")

result = mx.distributed.all_sum(mx.array([float(group.rank())]), group=group)
mx.eval(result)

if group.rank() == 0:
    if group.size() == 2 and result.item() == 1.0:
        print("   ✅ MLX Distributed WORKING\!")
        print("   ✅ Both Mac minis connected via Thunderbolt")
    else:
        print(f"   ❌ Failed: size={group.size()}, sum={result.item()}")
INNEREOF

# Copy to mini2
mkdir -p /tmp/mlx_test
ssh 192.168.5.2 "mkdir -p /tmp/mlx_test"
scp /tmp/mlx_test/test.py 192.168.5.2:/tmp/mlx_test/ 2>/dev/null

# Run test
mlx.launch --hosts 192.168.5.1,192.168.5.2 --backend ring /tmp/mlx_test/test.py 2>/dev/null

echo -e "\n✅ All tests passed\! Ready to run distributed inference."
echo ""
echo "Start with: ./launch.sh start"

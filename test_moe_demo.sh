#!/bin/bash

# Simple script to test MoE distributed inference

echo "🚀 Testing MoE Distributed Inference on your Mac minis"
echo "=================================================="

# Step 1: Copy the test to both machines
echo "1️⃣ Setting up test files..."
cp -r /tmp/mlx_test/test_moe_standalone.py /Users/mini1/Movies/mlx_grpc_inference/ 2>/dev/null || true
scp -q /tmp/mlx_test/test_moe_standalone.py mini2@192.168.5.2:/tmp/mlx_test/ 2>/dev/null || true

# Step 2: Check connection to mini2
echo "2️⃣ Checking connection to mini2..."
if ping -c 1 -W 1 192.168.5.2 > /dev/null 2>&1; then
    echo "   ✅ mini2 is reachable at 192.168.5.2"
else
    echo "   ❌ Cannot reach mini2. Make sure it's connected via Thunderbolt"
    exit 1
fi

# Step 3: Run the distributed test
echo "3️⃣ Launching distributed MoE test..."
echo "   This will use BOTH GPUs - one on each Mac mini"
echo ""

cd /tmp/mlx_test
mlx.launch --backend ring --hosts 192.168.5.1,192.168.5.2 python test_moe_standalone.py

echo ""
echo "=================================================="
echo "✨ Test complete! Check the output above to see:"
echo "   - Both devices connected (rank 0 and rank 1)"
echo "   - Each device processing different layers"
echo "   - GPU memory usage on both devices"
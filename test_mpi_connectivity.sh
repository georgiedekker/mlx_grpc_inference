#!/bin/bash

echo "Testing MPI Connectivity"
echo "========================"

# Test basic connectivity
echo "1. Testing ping to mini2..."
ping -c 2 192.168.5.2

echo ""
echo "2. Testing SSH to mini2..."
ssh 192.168.5.2 "echo 'SSH works: $(hostname)'"

echo ""
echo "3. Testing MPI with hostname..."
export OMPI_MCA_btl_tcp_if_include=bridge0
export OMPI_MCA_btl=self,tcp

mpirun \
    -np 1 --host 192.168.5.1 hostname : \
    -np 1 --host 192.168.5.2 hostname

echo ""
echo "4. Testing MPI with Python..."
mpirun \
    -np 1 --host 192.168.5.1 bash -c "source /Users/mini1/Movies/mlx_grpc_inference/.venv/bin/activate && python -c 'import mlx.core as mx; g=mx.distributed.init(); print(f\"mini1: rank {g.rank() if g else -1}\")'" : \
    -np 1 --host 192.168.5.2 bash -c "source /Users/mini2/.venv/bin/activate && python -c 'import mlx.core as mx; g=mx.distributed.init(); print(f\"mini2: rank {g.rank() if g else -1}\")'"

echo ""
echo "5. Testing distributed group size..."
cat > /tmp/test_dist.py << 'EOF'
import mlx.core as mx
import os
g = mx.distributed.init()
if g:
    print(f"{os.uname().nodename}: Rank {g.rank()}/{g.size()}")
else:
    print(f"{os.uname().nodename}: No group")
EOF

scp -q /tmp/test_dist.py 192.168.5.2:/tmp/

mpirun \
    -np 1 --host 192.168.5.1 bash -c "source /Users/mini1/Movies/mlx_grpc_inference/.venv/bin/activate && python /tmp/test_dist.py" : \
    -np 1 --host 192.168.5.2 bash -c "source /Users/mini2/.venv/bin/activate && python /tmp/test_dist.py"
#!/bin/bash
# Launch distributed MLX inference with MPI over Thunderbolt

# Set Thunderbolt-specific optimizations
export OMPI_MCA_btl_tcp_if_include=bridge0
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_endpoint_cache=65536

# Optional: Set socket buffer sizes for better throughput (requires sudo)
# Uncomment these lines if you want to optimize network buffers:
# sudo sysctl -w kern.ipc.maxsockbuf=16777216
# sudo sysctl -w net.inet.tcp.sendspace=4194304
# sudo sysctl -w net.inet.tcp.recvspace=4194304

# Kill any existing processes
pkill -f "python.*server.py"

# Set MLX distributed environment variables
export MLX_MASTER_ADDR=192.168.5.1
export MLX_MASTER_PORT=29500

# Launch with mpirun directly
echo "Launching distributed inference across 3 devices..."
mpirun -n 3 \
    --hostfile hosts.txt \
    --mca btl_tcp_if_include bridge0 \
    --mca btl self,tcp \
    uv run python server.py
#!/usr/bin/env python3
"""Test single node inference with the original distributed system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from distributed_config import DistributedConfig, DeviceConfig, DeviceRole
from distributed_comm import create_communicator, CommunicationBackend  
from distributed_mlx_inference import DistributedMLXInference

def test_single_node():
    print("Testing single node setup...")
    
    # Create minimal config
    device = DeviceConfig(
        device_id="mini1",
        hostname="mini1.local",
        port=50100,
        role=DeviceRole.MASTER,
        device_index=0
    )
    
    config = DistributedConfig(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        device_list=[device],
        model_parallel_size=1
    )
    
    # Initialize communicator
    comm = create_communicator(CommunicationBackend.GRPC)
    comm.init(rank=0, world_size=1, device_hostnames=["mini1.local"])
    
    # Create inference engine
    print("Creating inference engine...")
    inference = DistributedMLXInference(config, comm, local_rank=0)
    
    # Load model
    print("Loading model...")
    inference.load_model()
    
    # Test inference
    print("Testing inference...")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = inference.chat(messages, max_tokens=50)
    
    print(f"Response: {response}")
    
    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    test_single_node()
#!/usr/bin/env python3
"""
Test the enhanced distributed inference engine directly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from distributed_mlx_inference_dynamic import DistributedMLXInferenceDynamic
from distributed_config import DistributedConfig, DeviceConfig, DeviceRole
from distributed_comm import create_communicator, CommunicationBackend
from sharding_strategy import ShardInfo

def test_single_device():
    """Test inference on a single device."""
    print("🧪 Testing enhanced inference on single device...")
    
    # Create config for single device
    device_config = DeviceConfig(
        device_id="mini1",
        hostname="mini1.local",
        port=50100,
        role=DeviceRole.MASTER,
        device_index=0,
        capabilities={
            'memory_gb': 16,
            'available_memory_gb': 12,
            'gpu_cores': 10,
            'cpu_cores': 10
        }
    )
    
    config = DistributedConfig(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        device_list=[device_config],
        model_parallel_size=1
    )
    
    # Add extra attributes
    config.world_size = 1
    config.devices = [device_config]
    
    # Create communicator
    comm = create_communicator(CommunicationBackend.GRPC)
    comm.init(rank=0, world_size=1, device_hostnames=["mini1.local"])
    
    # Create inference engine
    inference = DistributedMLXInferenceDynamic(config, comm)
    
    # Load model shard
    shard_info = ShardInfo(
        start_layer=0,
        end_layer=27,  # All layers for single device
        total_layers=28,
        has_embeddings=True,
        has_head=True
    )
    
    print("📦 Loading model shard...")
    inference.load_model_shard(shard_info)
    
    # Test inference
    print("🤖 Testing inference...")
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response = inference.chat(messages, max_tokens=50)
    print(f"✅ Response: {response}")
    
    # Cleanup
    inference.shutdown()
    comm.finalize()
    
    print("✅ Single device test complete!")

if __name__ == "__main__":
    test_single_device()
#!/usr/bin/env python3
"""Test 2-device distributed inference."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from distributed_config import DistributedConfig
from distributed_comm import create_communicator, CommunicationBackend
from distributed_mlx_inference import DistributedMLXInference
import logging

logging.basicConfig(level=logging.INFO)

def test_inference():
    print("Testing 2-device distributed inference...")
    
    # Set environment
    os.environ["LOCAL_RANK"] = "0"
    
    # Load config
    config = DistributedConfig.load("distributed_config.json")
    print(f"Config loaded: {config.model_parallel_size} devices")
    
    # Initialize communicator
    comm = create_communicator(CommunicationBackend.GRPC)
    device_hostnames = [d.hostname for d in config.device_list]
    
    comm.init(
        rank=0,
        world_size=config.model_parallel_size,
        device_hostnames=device_hostnames
    )
    print("✓ Communicator initialized")
    
    # Initialize distributed inference
    print("\nInitializing distributed inference...")
    try:
        inference = DistributedMLXInference(config, comm, local_rank=0)
        print("✓ Distributed inference initialized")
        
        # Test a simple inference
        messages = [{"role": "user", "content": "Hello"}]
        print("\nTesting inference...")
        response = inference.chat(messages, max_tokens=10)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        comm.finalize()

if __name__ == "__main__":
    test_inference()
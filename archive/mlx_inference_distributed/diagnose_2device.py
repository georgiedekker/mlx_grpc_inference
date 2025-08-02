#!/usr/bin/env python3
"""Diagnose 2-device configuration issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from distributed_config import DistributedConfig
from distributed_comm import create_communicator, CommunicationBackend
import logging

logging.basicConfig(level=logging.INFO)

def test_2device_config():
    print("Testing 2-device configuration...")
    
    # Load config
    config = DistributedConfig.load("distributed_config.json")
    print(f"Loaded config: {config.model_parallel_size} devices")
    print(f"Devices: {[d.hostname for d in config.device_list]}")
    
    # Try to initialize communicator for rank 0
    print("\nInitializing communicator for rank 0...")
    comm = create_communicator(CommunicationBackend.GRPC)
    
    device_hostnames = [d.hostname for d in config.device_list]
    print(f"Device hostnames: {device_hostnames}")
    
    try:
        comm.init(
            rank=0,
            world_size=config.model_parallel_size,
            device_hostnames=device_hostnames
        )
        print("✓ Communicator initialized successfully")
        
        # Check connections
        print("\nChecking connections...")
        for i in range(config.model_parallel_size):
            if i != 0:
                try:
                    # Just check if we can reach the port
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((device_hostnames[i], 50100 + i))
                    sock.close()
                    
                    if result == 0:
                        print(f"✓ Can reach {device_hostnames[i]}:50100{i}")
                    else:
                        print(f"✗ Cannot reach {device_hostnames[i]}:50100{i}")
                except Exception as e:
                    print(f"✗ Error checking {device_hostnames[i]}: {e}")
                    
    except Exception as e:
        print(f"✗ Error initializing communicator: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if comm:
            comm.finalize()

if __name__ == "__main__":
    test_2device_config()
#!/usr/bin/env python3
"""Test gRPC communication between devices."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from distributed_comm import create_communicator, CommunicationBackend
from distributed_config import DistributedConfig
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_communication():
    """Test basic gRPC communication."""
    # Load config
    config = DistributedConfig.load("distributed_config.json")
    
    # Create communicator
    comm = create_communicator(CommunicationBackend.GRPC)
    
    # Initialize as rank 0
    device_hostnames = [d.hostname for d in config.devices]
    comm.init(rank=0, world_size=config.world_size, device_hostnames=device_hostnames)
    
    logger.info("Testing barrier synchronization...")
    
    try:
        comm.barrier()
        logger.info("Barrier test passed!")
    except Exception as e:
        logger.error(f"Barrier test failed: {e}")
        
    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    test_communication()
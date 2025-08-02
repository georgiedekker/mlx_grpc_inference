#!/usr/bin/env python3
"""Direct test of the distributed inference to debug the tensor passing."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from distributed_config import DistributedConfig
from distributed_mlx_inference import DistributedMLXInference
from distributed_comm import create_communicator, CommunicationBackend
import logging
import mlx.core as mx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_inference():
    """Test inference directly without the API layer."""
    # Load config
    config = DistributedConfig.load("distributed_config.json")
    
    # Create communicator
    comm = create_communicator(CommunicationBackend.GRPC)
    device_hostnames = [d.hostname for d in config.devices]
    comm.init(rank=0, world_size=config.world_size, device_hostnames=device_hostnames)
    
    # Create inference engine
    inference = DistributedMLXInference(config, comm)
    
    # Test with a simple prompt
    prompt = "Hello"
    
    logger.info("Testing direct inference...")
    try:
        # Test the chat interface
        response, token_count = inference.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.7,
            return_token_count=True
        )
        
        logger.info(f"Success! Response: {response}")
        logger.info(f"Tokens generated: {token_count}")
    except Exception as e:
        logger.error(f"Direct inference failed: {e}", exc_info=True)
    finally:
        comm.finalize()

if __name__ == "__main__":
    test_direct_inference()
#!/usr/bin/env python3
"""
Worker script for distributed MLX inference.
This runs on worker devices (like mini2) to participate in distributed inference.
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from distributed_config import DistributedConfig
from distributed_mlx_inference import DistributedMLXInference
from distributed_comm import create_communicator, CommunicationBackend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedWorker:
    """Worker process for distributed MLX inference."""
    
    def __init__(self, rank: int):
        self.rank = rank
        self.config = None
        self.communicator = None
        self.inference = None
        self.running = False
    
    def initialize(self):
        """Initialize the worker."""
        # Set environment variables
        os.environ["LOCAL_RANK"] = str(self.rank)
        os.environ["DISTRIBUTED_CONFIG"] = "distributed_config.json"
        
        # Load configuration
        config_path = os.environ.get("DISTRIBUTED_CONFIG", "distributed_config.json")
        if os.path.exists(config_path):
            self.config = DistributedConfig.load(config_path)
        else:
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        # Create communicator
        backend = CommunicationBackend(self.config.communication_backend)
        self.communicator = create_communicator(backend)
        
        # Collect device hostnames for gRPC communication
        world_size = self.config.model_parallel_size
        device_hostnames = []
        for i in range(world_size):
            device = self.config.get_device_by_index(i)
            if device:
                # Use hostname without the API port (e.g., "mini2.local" instead of "mini2.local:8001")
                hostname = device.hostname.split(':')[0] if ':' in device.hostname else device.hostname
                device_hostnames.append(hostname)
            else:
                device_hostnames.append("localhost")
        
        # Initialize gRPC communicator
        self.communicator.init(rank=self.rank, world_size=world_size, device_hostnames=device_hostnames)
        
        # Initialize distributed inference
        self.inference = DistributedMLXInference(
            config=self.config,
            communicator=self.communicator,
            local_rank=self.rank
        )
        
        device_info = self.config.get_device_by_index(self.rank)
        logger.info(f"Worker initialized:")
        logger.info(f"  - Rank: {self.rank}")
        logger.info(f"  - World size: {world_size}")
        logger.info(f"  - Device ID: {device_info.device_id if device_info else 'unknown'}")
        logger.info(f"  - Hostname: {device_info.hostname if device_info else 'unknown'}")
        logger.info(f"  - gRPC initialized: {self.communicator._initialized if hasattr(self.communicator, '_initialized') else 'unknown'}")
        
        return True
    
    def run(self):
        """Run the worker process."""
        self.running = True
        logger.info(f"Worker {self.rank} is ready and waiting for inference requests...")
        logger.info(f"Worker {self.rank}: gRPC server is running on port {50100 + self.rank}")
        logger.info(f"Worker {self.rank}: Ready to handle distributed tensor operations")
        
        try:
            # The gRPC server handles all communication
            # We just keep the process alive
            while self.running:
                time.sleep(5.0)
                logger.debug(f"Worker {self.rank}: Still running, gRPC server active")
                    
        except KeyboardInterrupt:
            logger.info("Worker received shutdown signal")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the worker."""
        self.running = False
        if self.communicator:
            self.communicator.finalize()
        logger.info(f"Worker {self.rank} shut down")


def main():
    """Main worker entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Distributed Worker")
    parser.add_argument("--rank", type=int, default=1, 
                       help="Worker rank (default: 1)")
    parser.add_argument("--config", type=str, default="distributed_config.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Set config path
    os.environ["DISTRIBUTED_CONFIG"] = args.config
    
    # Create and run worker
    worker = DistributedWorker(rank=args.rank)
    
    if worker.initialize():
        logger.info(f"Starting worker {args.rank}...")
        worker.run()
    else:
        logger.error("Failed to initialize worker")
        sys.exit(1)


if __name__ == "__main__":
    main()
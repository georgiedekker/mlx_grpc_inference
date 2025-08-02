#!/usr/bin/env python3
"""
Simple MLX worker that auto-registers and waits for inference requests.
No configuration needed - just run this on any device!
"""

import os
import sys
import logging
import random
import argparse
from pathlib import Path
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_discovery import MLXServiceDiscovery
from distributed_config import DistributedConfig, DeviceConfig, DeviceRole
from distributed_mlx_inference_dynamic import DistributedMLXInferenceDynamic
from distributed_comm import create_communicator, CommunicationBackend
from hardware_detector import HardwareDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMLXWorker:
    """Simple worker that auto-registers and handles inference requests."""
    
    def __init__(self, port: Optional[int] = None, model_id: str = "mlx-community/Qwen3-1.7B-8bit"):
        """
        Initialize simple worker.
        
        Args:
            port: gRPC port (auto-assigned if None)
            model_id: Model to use for inference
        """
        self.port = port or (50000 + random.randint(100, 999))
        self.model_id = model_id
        self.discovery = None
        self.communicator = None
        self.inference = None
        self.running = False
        
    def start(self):
        """Start the worker and register with discovery service."""
        logger.info(f"🚀 Starting MLX worker on port {self.port}")
        
        # Create discovery service and register
        self.discovery = MLXServiceDiscovery()
        worker_info = self.discovery.register_worker(self.port)
        
        logger.info(f"📡 Registered with discovery service:")
        logger.info(f"   - Hostname: {worker_info.hostname}")
        logger.info(f"   - IP: {worker_info.ip_address}")
        logger.info(f"   - Port: {worker_info.port}")
        logger.info(f"   - Memory: {worker_info.memory_gb}GB total, {worker_info.available_memory_gb:.1f}GB available")
        logger.info(f"   - GPU cores: {worker_info.gpu_cores}")
        logger.info(f"   - Thunderbolt: {'Yes' if worker_info.thunderbolt_available else 'No'}")
        
        # Create minimal config for this worker
        config = self._create_worker_config(worker_info)
        
        # Initialize gRPC communicator
        self.communicator = create_communicator(CommunicationBackend.GRPC)
        self.communicator.init(
            rank=1,  # Workers are always non-zero rank
            world_size=2,  # Will be updated dynamically
            port=self.port
        )
        
        # Initialize inference engine
        self.inference = DistributedMLXInferenceDynamic(config, self.communicator)
        
        logger.info(f"✅ Worker ready and listening on port {self.port}")
        logger.info(f"💡 To use this worker, it will be auto-discovered by the master node")
        
        # Keep running
        self.running = True
        try:
            while self.running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down worker...")
            self.cleanup()
            
    def _create_worker_config(self, worker_info) -> DistributedConfig:
        """Create minimal config for worker."""
        # Detect hardware capabilities
        detector = HardwareDetector()
        hw_info = detector.generate_device_config()
        
        # Create device config
        device_config = DeviceConfig(
            device_id=worker_info.device_id,
            hostname=worker_info.hostname,
            port=worker_info.port,
            role=DeviceRole.WORKER,
            device_index=1,  # Will be updated by master
            capabilities=hw_info
        )
        
        # Create minimal distributed config
        config = DistributedConfig(
            model_name=self.model_id,
            device_list=[device_config],
            model_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        # Add extra attributes
        config.world_size = 1
        config.devices = [device_config]
        
        return config
        
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.discovery:
            self.discovery.cleanup()
        if self.communicator:
            self.communicator.finalize()

def main():
    """Main entry point for simple worker."""
    parser = argparse.ArgumentParser(description="Simple MLX worker with auto-discovery")
    parser.add_argument("--port", type=int, help="gRPC port (auto-assigned if not specified)")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-1.7B-8bit",
                       help="Model ID to use")
    
    args = parser.parse_args()
    
    # Create and start worker
    worker = SimpleMLXWorker(port=args.port, model_id=args.model)
    worker.start()

if __name__ == "__main__":
    main()
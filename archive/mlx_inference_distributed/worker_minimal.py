#!/usr/bin/env python3
"""
Minimal MLX worker that just registers itself for discovery.
The master will handle all the distributed inference setup.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mlx_discovery import MLXServiceDiscovery

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Minimal MLX Worker")
    parser.add_argument("--port", type=int, help="Port to listen on (auto-selected if not provided)")
    args = parser.parse_args()
    
    # Use auto-selected port if not provided
    port = args.port or 0
    
    logger.info(f"🚀 Starting minimal MLX worker on port {port}")
    
    # Create discovery service
    discovery = MLXServiceDiscovery()
    
    # Register this worker
    worker_info = discovery.register_worker(port=port)
    
    logger.info(f"✅ Worker registered:")
    logger.info(f"   - Hostname: {worker_info.hostname}")
    logger.info(f"   - Port: {worker_info.port}")
    logger.info(f"   - Memory: {worker_info.memory_gb:.1f}GB total, {worker_info.available_memory_gb:.1f}GB available")
    logger.info(f"   - GPU cores: {worker_info.gpu_cores}")
    logger.info(f"   - Thunderbolt: {'Yes' if worker_info.thunderbolt_available else 'No'}")
    
    logger.info("📡 Worker is now discoverable by the master node")
    logger.info("⏳ Press Ctrl+C to stop")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down worker...")
        discovery.cleanup()

if __name__ == "__main__":
    main()
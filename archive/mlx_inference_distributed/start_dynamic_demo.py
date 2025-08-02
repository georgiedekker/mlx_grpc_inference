#!/usr/bin/env python3
"""
Demo script to start a dynamic distributed inference cluster.
This shows how the pieces fit together.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mlx_discovery import MLXServiceDiscovery
from dynamic_cluster_manager import DynamicClusterManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_master():
    """Start the master node with discovery."""
    logger.info("🚀 Starting MLX Dynamic Distributed Inference Demo")
    logger.info("=" * 50)
    
    # Create cluster manager
    manager = DynamicClusterManager(
        model_id="mlx-community/Qwen3-1.7B-8bit",
        master_port=50100
    )
    
    # Start discovery
    manager.start()
    
    logger.info("")
    logger.info("📡 Master node is running with auto-discovery enabled")
    logger.info("")
    logger.info("To add workers:")
    logger.info("  1. On other Macs: python worker_simple.py")
    logger.info("  2. Workers will auto-register via mDNS")
    logger.info("")
    logger.info("Features:")
    logger.info("  ✅ Automatic device discovery")
    logger.info("  ✅ Dynamic shard rebalancing")  
    logger.info("  ✅ Thunderbolt network detection")
    logger.info("  ✅ Maximum RAM utilization")
    logger.info("")
    
    try:
        while True:
            time.sleep(5)
            config = manager.get_cluster_config()
            
            # Show cluster status
            logger.info(f"📊 Cluster: {config.world_size} devices, "
                       f"{manager.cluster_state.total_memory_gb:.1f}GB RAM, "
                       f"{manager.cluster_state.total_gpu_cores} GPU cores")
                       
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        manager.stop()

def start_worker(port=None):
    """Start a worker node."""
    from worker_simple import SimpleMLXWorker
    
    worker = SimpleMLXWorker(port=port)
    worker.start()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Dynamic Distributed Inference Demo")
    parser.add_argument("--mode", choices=["master", "worker"], default="master",
                       help="Run as master or worker")
    parser.add_argument("--port", type=int, help="Port for worker mode")
    
    args = parser.parse_args()
    
    if args.mode == "master":
        start_master()
    else:
        start_worker(args.port)
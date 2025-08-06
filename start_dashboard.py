#!/usr/bin/env python3
"""
Start the performance monitoring dashboard for MLX distributed inference.
"""
import asyncio
import sys
import logging
import uvicorn

# Add project root to path
sys.path.append('/Users/mini1/Movies/mlx_grpc_inference')

from src.performance_monitor import get_monitor, DashboardServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Start the monitoring dashboard."""
    logger.info("Starting MLX Distributed Inference Performance Dashboard")
    
    # Get or create monitor instance
    monitor = get_monitor()
    
    # Create dashboard server
    dashboard = DashboardServer(monitor, port=8888)
    
    logger.info("Dashboard available at: http://localhost:8888")
    logger.info("API endpoints:")
    logger.info("  - http://localhost:8888/api/metrics (current metrics)")
    logger.info("  - http://localhost:8888/api/layers (layer performance)")
    logger.info("  - http://localhost:8888/api/history (metrics history)")
    
    # Start dashboard server
    await dashboard.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
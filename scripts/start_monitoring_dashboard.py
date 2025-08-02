#!/usr/bin/env python3
"""
Standalone monitoring dashboard for MLX distributed inference.

This script starts a monitoring dashboard that can connect to running
distributed inference systems to provide real-time monitoring.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.metrics import MetricsCollector
from monitoring.health import HealthMonitor
from monitoring.dashboard import start_monitoring_dashboard


async def main():
    """Main entry point for standalone monitoring dashboard."""
    parser = argparse.ArgumentParser(description="MLX Distributed Inference Monitoring Dashboard")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Dashboard host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)"
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="monitoring-dashboard",
        help="Device ID for the monitoring dashboard (default: monitoring-dashboard)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create monitoring components
    metrics_collector = MetricsCollector(args.device_id)
    health_monitor = HealthMonitor(args.device_id)
    
    try:
        logger.info(f"Starting monitoring dashboard at http://{args.host}:{args.port}")
        
        # Start the dashboard
        await start_monitoring_dashboard(
            metrics_collector,
            health_monitor,
            host=args.host,
            port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            await metrics_collector.stop()
            await health_monitor.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
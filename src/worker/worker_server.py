"""
Worker server implementation for distributed inference using device abstraction.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from ..core.config import ClusterConfig, DeviceRole
from ..devices import WorkerDevice, DeviceFactory

logger = logging.getLogger(__name__)


class WorkerServer:
    """Worker server that manages a WorkerDevice instance."""
    
    def __init__(self, config_path: str, enable_monitoring: bool = True):
        """
        Initialize worker server with device abstraction.
        
        Args:
            config_path: Path to cluster configuration file
            enable_monitoring: Whether to enable monitoring and health checks (deprecated, now handled by device)
        """
        self.config = ClusterConfig.from_yaml(config_path)
        self.device_config = self.config.get_local_device()
        
        if not self.device_config:
            raise ValueError("Could not determine local device configuration")
        
        if self.device_config.role != DeviceRole.WORKER:
            raise ValueError(f"Device {self.device_config.device_id} is not configured as a worker")
        
        # Create worker device using device factory
        device_factory = DeviceFactory(self.config)
        self.worker_device = device_factory.create_worker(self.device_config)
        
        # Keep enable_monitoring for backward compatibility
        self.enable_monitoring = enable_monitoring
        
        logger.info(f"Initializing worker server for {self.device_config.device_id} with device abstraction")
    
    async def start(self):
        """Start the worker server using device abstraction."""
        try:
            logger.info(f"Starting worker server with device abstraction...")
            
            # Initialize the worker device
            await self.worker_device.initialize()
            
            # Verify device is ready
            if not self.worker_device.is_ready:
                raise RuntimeError("Worker device failed to initialize properly")
            
            # Get device health for logging
            health = await self.worker_device.check_health()
            logger.info(f"Worker device initialized successfully - Health: {health.is_healthy}, "
                       f"Assigned layers: {self.worker_device.get_assigned_layers()}")
            
            # The device handles its own gRPC server startup
            logger.info("Worker server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start worker server: {e}")
            
            # Attempt to shutdown the device on failure
            try:
                await self.worker_device.shutdown()
            except Exception as shutdown_error:
                logger.error(f"Error during cleanup after start failure: {shutdown_error}")
            
            raise
    
    async def shutdown(self):
        """Shutdown the worker server using device abstraction."""
        logger.info("Shutting down worker server...")
        
        try:
            # Shutdown the worker device
            await self.worker_device.shutdown()
            logger.info("Worker device shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during worker server shutdown: {e}")
    
    def get_device_metrics(self) -> dict:
        """Get device metrics and health information."""
        if not self.worker_device:
            return {"error": "Worker device not available"}
        
        metrics = self.worker_device.get_metrics()
        return {
            "device_id": self.worker_device.device_id,
            "state": self.worker_device.state.value,
            "is_ready": self.worker_device.is_ready,
            "is_healthy": self.worker_device.is_healthy,
            "assigned_layers": self.worker_device.get_assigned_layers(),
            "processing_requests": self.worker_device.get_processing_requests(),
            "metrics": metrics.__dict__ if metrics else None
        }
    
    async def get_device_health(self) -> dict:
        """Get device health status."""
        if not self.worker_device:
            return {"error": "Worker device not available"}
        
        health = await self.worker_device.check_health()
        return {
            "is_healthy": health.is_healthy,
            "state": health.state.value,
            "uptime_seconds": health.uptime_seconds,
            "memory_usage_percent": health.memory_usage_percent,
            "gpu_utilization_percent": health.gpu_utilization_percent,
            "error_message": health.error_message,
            "request_count": health.request_count,
            "average_response_time_ms": health.average_response_time_ms
        }


async def main():
    """Main entry point for worker server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Worker Server")
    parser.add_argument(
        "--config",
        type=str,
        default="config/cluster_config.yaml",
        help="Path to cluster configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = WorkerServer(args.config)
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(server.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await server.shutdown()
    except Exception as e:
        logger.error(f"Worker server error: {e}")
        await server.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
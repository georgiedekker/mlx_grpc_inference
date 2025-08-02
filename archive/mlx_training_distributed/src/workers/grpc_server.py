#!/usr/bin/env python3
"""
gRPC server for distributed MLX training workers.
Handles worker-to-coordinator communication and training coordination.
"""

import argparse
import asyncio
import grpc
import logging
import os
import sys
import time
from concurrent import futures
from typing import Dict, Any, Optional
import threading
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("Warning: MLX not available. Using fallback mode.")
    mx = None
    nn = None

# Import distributed training components
from src.training.distributed_trainer import DistributedTrainer, DistributedConfig
from src.training.training_coordinator import TrainingCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkerService:
    """gRPC service for handling distributed training tasks."""
    
    def __init__(self, device_id: int, coordinator_host: str, coordinator_port: int):
        self.device_id = device_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.status = "initializing"
        self.current_job = None
        self.training_thread = None
        self.shutdown_event = threading.Event()
        
        # Device/rank mapping
        self.world_rank = device_id  # Use device_id as world rank
        
        logger.info(f"Worker service initialized on device {device_id}")
        
    async def RegisterWorker(self, request, context):
        """Register this worker with the coordinator."""
        try:
            self.status = "registering"
            
            # Register with coordinator
            registration_data = {
                "device_id": self.device_id,
                "rank": self.world_rank,
                "host": request.host if hasattr(request, 'host') else "localhost",
                "port": request.port if hasattr(request, 'port') else 50051,
                "status": "ready",
                "capabilities": {
                    "mlx_available": mx is not None,
                    "max_memory_gb": 8,  # Placeholder
                    "device_type": "mlx"
                }
            }
            
            # In a real implementation, this would make gRPC call to coordinator
            logger.info(f"Registering worker {self.device_id} with coordinator at {self.coordinator_host}:{self.coordinator_port}")
            
            self.status = "ready"
            return {"success": True, "message": f"Worker {self.device_id} registered successfully"}
            
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            self.status = "error"
            return {"success": False, "message": str(e)}
    
    async def StartTraining(self, request, context):
        """Start a training job on this worker."""
        try:
            if self.status != "ready":
                return {"success": False, "message": f"Worker not ready, status: {self.status}"}
            
            self.status = "training"
            job_config = request.config if hasattr(request, 'config') else {}
            
            logger.info(f"Starting training job on device {self.device_id}")
            
            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(job_config,)
            )
            self.training_thread.start()
            
            return {"success": True, "message": f"Training started on device {self.device_id}"}
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.status = "error"
            return {"success": False, "message": str(e)}
    
    def _run_training(self, job_config: Dict[str, Any]):
        """Run training job in background thread."""
        try:
            # Create distributed configuration
            dist_config = DistributedConfig(
                world_size=job_config.get("world_size", 1),
                rank=self.world_rank,
                backend=job_config.get("backend", "allreduce"),
                master_addr=job_config.get("master_addr", self.coordinator_host),
                master_port=job_config.get("master_port", 29500)  # Different from gRPC port
            )
            
            # Create training coordinator
            coordinator = TrainingCoordinator(
                job_id=f"worker-{self.device_id}-{int(time.time())}",
                base_config=job_config
            )
            
            # Run training
            results = asyncio.run(coordinator.start_training())
            
            logger.info(f"Training completed on device {self.device_id}: {results}")
            self.status = "ready"
            
        except Exception as e:
            logger.error(f"Training failed on device {self.device_id}: {e}")
            self.status = "error"
    
    async def GetStatus(self, request, context):
        """Get current worker status."""
        status_info = {
            "device_id": self.device_id,
            "rank": self.world_rank,
            "status": self.status,
            "current_job": self.current_job,
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
        
        return {"status": status_info}
    
    async def StopTraining(self, request, context):
        """Stop current training job."""
        try:
            if self.training_thread and self.training_thread.is_alive():
                self.shutdown_event.set()
                self.training_thread.join(timeout=10)
                logger.info(f"Training stopped on device {self.device_id}")
            
            self.status = "ready"
            return {"success": True, "message": "Training stopped"}
            
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return {"success": False, "message": str(e)}


async def serve_worker(device_id: int, port: int, coordinator_host: str, coordinator_port: int):
    """Start the gRPC worker server."""
    
    # Create worker service
    worker_service = WorkerService(device_id, coordinator_host, coordinator_port)
    worker_service.start_time = time.time()
    
    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add service to server (in real implementation, would use generated gRPC stubs)
    # server.add_insecure_port(f'[::]:{port}')
    
    # For now, create a simple TCP server simulation
    logger.info(f"Worker gRPC server starting on device {device_id}, port {port}")
    logger.info(f"Coordinator: {coordinator_host}:{coordinator_port}")
    
    # Register with coordinator
    try:
        # Simulate registration call
        registration_result = await worker_service.RegisterWorker(
            type('Request', (), {'host': 'localhost', 'port': port})(),
            None
        )
        logger.info(f"Registration result: {registration_result}")
        
    except Exception as e:
        logger.error(f"Failed to register with coordinator: {e}")
    
    # Start server
    await server.start()
    logger.info(f"Worker {device_id} gRPC server started on port {port}")
    
    # Keep server running
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        await server.stop(5)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down worker...")
    sys.exit(0)


def main():
    """Main entry point for gRPC worker server."""
    parser = argparse.ArgumentParser(description="MLX Training Worker gRPC Server")
    
    # Core arguments - note: using --device-id (not --rank)
    parser.add_argument("--device-id", type=int, required=True,
                       help="Device ID for this worker (also used as rank)")
    parser.add_argument("--port", type=int, default=50051,
                       help="Port for gRPC server")
    
    # Coordinator connection
    parser.add_argument("--coordinator-host", type=str, default="localhost",
                       help="Coordinator host address")
    parser.add_argument("--coordinator-port", type=int, default=50050,
                       help="Coordinator gRPC port")
    
    # Distributed training settings
    parser.add_argument("--world-size", type=int, default=1,
                       help="Total number of workers")
    parser.add_argument("--backend", type=str, default="allreduce",
                       choices=["allreduce", "ring_allreduce", "parameter_server"],
                       help="Distributed backend")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - Worker[{args.device_id}] - %(levelname)s - %(message)s'
    )
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start worker server
    logger.info(f"Starting MLX Training Worker {args.device_id}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Coordinator: {args.coordinator_host}:{args.coordinator_port}")
    logger.info(f"  World size: {args.world_size}")
    logger.info(f"  Backend: {args.backend}")
    
    try:
        # Run async server
        asyncio.run(serve_worker(
            args.device_id,
            args.port,
            args.coordinator_host,
            args.coordinator_port
        ))
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
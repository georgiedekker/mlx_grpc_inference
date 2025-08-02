#!/usr/bin/env python
"""
MPI-based distributed inference server.

This runs the distributed inference with MPI and only the master node
exposes the API endpoint.
"""
import os
import sys
from mpi4py import MPI
import uvicorn
import asyncio
import logging

# Initialize MPI first
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for the distributed API
os.environ["OMPI_COMM_WORLD_RANK"] = str(rank)
os.environ["OMPI_COMM_WORLD_SIZE"] = str(size)
os.environ["DISTRIBUTED_CONFIG"] = "distributed_config.json"

logger.info(f"Starting MPI inference server: rank={rank}, size={size}, host={MPI.Get_processor_name()}")

# Import the distributed API app
from distributed_api import app, startup_event, shutdown_event

async def main():
    """Main function to run the appropriate role."""
    if rank == 0:
        # Master node runs the API server
        logger.info("Master node starting API server on port 8100")
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    else:
        # Worker nodes need to run the startup event to initialize inference
        await startup_event()
        logger.info(f"Worker node {rank} ready for distributed inference")
        
        # Get the distributed inference instance
        from distributed_api import distributed_inference
        
        # Worker loop - actively participate in inference requests
        # Optimized non-blocking loop with better error handling
        try:
            idle_count = 0
            max_idle_cycles = 100  # Adaptive sleep after idle cycles
            
            while True:
                # Check if master is requesting inference (non-blocking)
                request = comm.iprobe(source=0, tag=100)
                if request:
                    idle_count = 0  # Reset idle counter
                    try:
                        # Receive the signal with timeout
                        msg = comm.recv(source=0, tag=100)
                        logger.info(f"Worker {rank} received inference request: {msg['type']}")
                        
                        if msg['type'] == 'chat':
                            # Participate in the distributed chat
                            try:
                                # Worker calls chat with None messages - it will synchronize with master
                                _, _ = distributed_inference.chat(
                                    messages=None,  # Workers don't need messages
                                    max_tokens=msg['max_tokens'],
                                    temperature=msg['temperature'],
                                    top_p=msg['top_p'],
                                    repetition_penalty=msg['repetition_penalty'],
                                    return_token_count=True
                                )
                                logger.info(f"Worker {rank} completed inference participation")
                            except Exception as e:
                                logger.error(f"Worker {rank} inference error: {e}")
                        elif msg['type'] == 'shutdown':
                            logger.info(f"Worker {rank} received shutdown signal")
                            break
                    except Exception as e:
                        logger.error(f"Worker {rank} message handling error: {e}")
                else:
                    idle_count += 1
                    # Adaptive sleep - longer when idle
                    if idle_count < max_idle_cycles:
                        await asyncio.sleep(0.01)  # Active polling
                    else:
                        await asyncio.sleep(0.05)  # Longer idle sleep
        except KeyboardInterrupt:
            logger.info(f"Worker node {rank} shutting down")
    
    # Cleanup
    await shutdown_event()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(f"Rank {rank} shutting down")
    finally:
        MPI.Finalize()
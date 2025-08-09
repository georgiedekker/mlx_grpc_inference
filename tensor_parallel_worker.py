#!/usr/bin/env python3
import asyncio
import logging
import mlx.core as mx
from mlx_lm import load
import grpc
from concurrent import futures
import sys

sys.path.append('/Users/mini2/Movies/mlx_grpc_inference')

from src.communication import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mx.set_default_device(mx.gpu)

class TensorParallelWorker(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, device_id=1, world_size=2):
        self.device_id = device_id
        self.world_size = world_size
        logger.info(f'Tensor parallel worker {device_id} initialized')
    
    def AllReduce(self, request, context):
        # Handle AllReduce operations
        logger.info(f'AllReduce operation: {request.operation}')
        response = inference_pb2.AllReduceResponse()
        response.status = 'completed'
        return response
    
    def HealthCheck(self, request, context):
        return inference_pb2.HealthResponse(
            status='healthy',
            message=f'Worker {self.device_id} ready for tensor parallelism'
        )

async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    worker = TensorParallelWorker(device_id=1, world_size=2)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(worker, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info('Tensor parallel worker started on port 50051')
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())

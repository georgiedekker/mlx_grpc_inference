#!/usr/bin/env python3
"""
Debug worker to see what's happening with gRPC.
"""
import grpc
import asyncio
import logging
from concurrent import futures
from src.communication import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugService(inference_pb2_grpc.InferenceServiceServicer):
    def ProcessLayers(self, request, context):
        logger.info("ProcessLayers called!")
        try:
            logger.info(f"Request type: {type(request)}")
            logger.info(f"Request has start_layer: {hasattr(request, 'start_layer')}")
            logger.info(f"Request has input_tensor: {hasattr(request, 'input_tensor')}")
            if hasattr(request, 'start_layer'):
                logger.info(f"start_layer = {request.start_layer}")
            if hasattr(request, 'end_layer'):
                logger.info(f"end_layer = {request.end_layer}")
            
            # Return minimal response
            response = inference_pb2.LayerResponseV2()
            tensor = inference_pb2.Tensor()
            tensor.data = b"test"
            tensor.shape.extend([1, 2, 3])
            tensor.dtype = "float32"
            response.output_tensor.CopyFrom(tensor)
            return response
            
        except Exception as e:
            logger.error(f"Error in ProcessLayers: {e}", exc_info=True)
            raise

async def serve():
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
        ]
    )
    
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(DebugService(), server)
    server.add_insecure_port('[::]:50052')
    
    await server.start()
    logger.info("Debug worker started on port 50052")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
#!/usr/bin/env python3
"""
Minimal worker to test gRPC without complex dependencies.
"""
import sys
import grpc
import asyncio
from concurrent import futures

# Import proto files
from src.communication import inference_pb2, inference_pb2_grpc

print("MINIMAL WORKER STARTED", flush=True)

class MinimalWorkerService(inference_pb2_grpc.InferenceServiceServicer):
    def ProcessLayers(self, request, context):
        print(f"✓ ProcessLayers called!", flush=True)
        print(f"✓ Request type: {type(request)}", flush=True)
        print(f"✓ start_layer: {request.start_layer}", flush=True)
        print(f"✓ end_layer: {request.end_layer}", flush=True)
        
        # Return dummy response
        response = inference_pb2.LayerResponseV2()
        response.output_tensor.data = b"dummy_data"
        response.output_tensor.shape.extend([1, 5, 768])
        response.output_tensor.dtype = "float32"
        
        print("✓ Returning response", flush=True)
        return response
    
    def HealthCheck(self, request, context):
        return inference_pb2.HealthResponse(status="healthy", message="Minimal worker")

async def serve():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(MinimalWorkerService(), server)
    server.add_insecure_port(f'[::]:{port}')
    
    await server.start()
    print(f"Minimal worker listening on port {port}", flush=True)
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
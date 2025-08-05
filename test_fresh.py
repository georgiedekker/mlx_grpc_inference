#!/usr/bin/env python3
"""
Fresh test with no cached imports.
"""
import sys
import importlib

# Force reload of proto modules
if 'src.communication.inference_pb2' in sys.modules:
    del sys.modules['src.communication.inference_pb2']
if 'src.communication.inference_pb2_grpc' in sys.modules:
    del sys.modules['src.communication.inference_pb2_grpc']

import grpc
from src.communication import inference_pb2, inference_pb2_grpc

print("Proto module location:", inference_pb2.__file__)
print("Available request types:", [x for x in dir(inference_pb2) if 'Request' in x])

# Create minimal request
request = inference_pb2.LayerRequestV2()
request.start_layer = 10
request.end_layer = 11

# Create minimal tensor
tensor = inference_pb2.Tensor()
tensor.data = b"test"
tensor.shape.extend([1])
tensor.dtype = "float32"
request.input_tensor.CopyFrom(tensor)

print(f"\nRequest type: {type(request).__name__}")
print(f"Request proto: {request.DESCRIPTOR.full_name}")

# Create fresh channel and stub
channel = grpc.insecure_channel("192.168.5.2:50051")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

print(f"\nStub type: {type(stub)}")
print(f"ProcessLayers method: {stub.ProcessLayers}")

try:
    print("\nCalling ProcessLayers...")
    response = stub.ProcessLayers(request, timeout=5)
    print("✓ Success!")
except grpc.RpcError as e:
    print(f"✗ Error: {e.code()}")
    print(f"  Details: {e.details()}")
    
channel.close()
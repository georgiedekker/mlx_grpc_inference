#!/usr/bin/env python3
"""
Test connecting to port 50052.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

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

print(f"Connecting to 192.168.5.2:50052")

# Connect to port 50052
channel = grpc.insecure_channel("192.168.5.2:50052")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

try:
    print("Calling ProcessLayers...")
    response = stub.ProcessLayers(request, timeout=5)
    print("✓ Success!")
    print(f"Response: {response.output_tensor.shape}")
except grpc.RpcError as e:
    print(f"✗ Error: {e.code()}")
    print(f"  Details: {e.details()}")
    
channel.close()
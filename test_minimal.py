#!/usr/bin/env python3
"""
Minimal test to isolate the proto issue.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

# Create empty request
request = inference_pb2.LayerRequestV2()
request.start_layer = 10
request.end_layer = 11

# Create empty tensor
tensor = inference_pb2.Tensor()
tensor.data = b""  # Empty data
tensor.shape.extend([1])
tensor.dtype = "float32"
request.input_tensor.CopyFrom(tensor)

print(f"Request class: {type(request)}")
print(f"Request proto: {request.DESCRIPTOR.full_name}")

# Send to worker
channel = grpc.insecure_channel("192.168.5.2:50051")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

try:
    print("\nSending minimal request...")
    response = stub.ProcessLayers(request, timeout=5)
    print("✓ Request accepted!")
except grpc.RpcError as e:
    print(f"✗ Error: {e.code()}")
    print(f"  Details: {e.details()}")
    
channel.close()
#!/usr/bin/env python3
"""
Test minimal RPC to debug deserialization.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

print("Testing minimal LayerRequestV2...")

# Create minimal request with just required fields
request = inference_pb2.LayerRequestV2()
request.start_layer = 10
request.end_layer = 19

# Create a minimal Tensor
tensor = inference_pb2.Tensor()
tensor.data = b"test"  # Just some bytes
tensor.shape.extend([1, 2, 3])
tensor.dtype = "float32"

request.input_tensor.CopyFrom(tensor)

print(f"Request created: start={request.start_layer}, end={request.end_layer}")
print(f"Tensor: data_size={len(request.input_tensor.data)}, shape={list(request.input_tensor.shape)}")

try:
    channel = grpc.insecure_channel("192.168.5.2:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    print("Calling ProcessLayers...")
    response = stub.ProcessLayers(request, timeout=5)
    print("✓ RPC succeeded!")
    print(f"Response tensor shape: {list(response.output_tensor.shape)}")
    
except grpc.RpcError as e:
    print(f"✗ RPC failed:")
    print(f"  Code: {e.code()}")
    print(f"  Details: {e.details()}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
finally:
    if 'channel' in locals():
        channel.close()
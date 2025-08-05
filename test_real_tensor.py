#!/usr/bin/env python3
"""
Test with a real serialized tensor.
"""
import grpc
import mlx.core as mx
from src.communication import inference_pb2, inference_pb2_grpc
from src.communication.tensor_utils import serialize_mlx_array

print("Creating real tensor for testing...")

# Create a real MLX tensor
test_array = mx.random.normal((1, 5, 768))  # batch=1, seq_len=5, hidden_dim=768
print(f"Test tensor shape: {test_array.shape}")

# Serialize it properly
data, metadata = serialize_mlx_array(test_array, compress=True, compression_algorithm="lz4")

# Create the proto Tensor
tensor = inference_pb2.Tensor()
tensor.data = data
tensor.shape.extend(metadata['shape'])
tensor.dtype = metadata['dtype']

# Create the request
request = inference_pb2.LayerRequestV2()
request.input_tensor.CopyFrom(tensor)
request.start_layer = 10
request.end_layer = 19

print(f"Request: layers {request.start_layer}-{request.end_layer-1}")
print(f"Tensor: size={len(tensor.data)} bytes, shape={list(tensor.shape)}, dtype={tensor.dtype}")

try:
    channel = grpc.insecure_channel("192.168.5.2:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    print("\nCalling ProcessLayers...")
    response = stub.ProcessLayers(request, timeout=10)
    
    print("✓ SUCCESS!")
    print(f"Response tensor: shape={list(response.output_tensor.shape)}, size={len(response.output_tensor.data)} bytes")
    
except grpc.RpcError as e:
    print(f"\n✗ RPC Error:")
    print(f"  Code: {e.code()}")
    print(f"  Details: {e.details()}")
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
finally:
    if 'channel' in locals():
        channel.close()
#!/usr/bin/env python3
"""
Test ProcessLayers RPC directly.
"""
import grpc
import mlx.core as mx
import numpy as np
from src.communication import inference_pb2, inference_pb2_grpc
from src.communication.tensor_utils import serialize_mlx_array

# Create a test tensor
test_array = mx.random.normal((1, 10, 768))  # batch=1, seq_len=10, hidden_dim=768
print(f"Test tensor shape: {test_array.shape}")

# Serialize it
data, metadata = serialize_mlx_array(test_array, compress=True, compression_algorithm="lz4")
tensor = inference_pb2.Tensor()
tensor.data = data
tensor.shape.extend(metadata['shape'])
tensor.dtype = metadata['dtype']

print(f"Serialized tensor: data size={len(data)}, shape={list(tensor.shape)}, dtype={tensor.dtype}")

# Create request
request = inference_pb2.LayerRequestV2()
request.input_tensor.CopyFrom(tensor)
request.start_layer = 10
request.end_layer = 19

print(f"\nTesting ProcessLayers on mini2...")

try:
    channel = grpc.insecure_channel("192.168.5.2:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    response = stub.ProcessLayers(request, timeout=5)
    print(f"✓ Success! Response tensor shape: {list(response.output_tensor.shape)}")
    
except grpc.RpcError as e:
    print(f"✗ RPC Error:")
    print(f"  Code: {e.code()}")
    print(f"  Details: {e.details()}")
    print(f"  Debug: {e.debug_error_string()}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
finally:
    if 'channel' in locals():
        channel.close()
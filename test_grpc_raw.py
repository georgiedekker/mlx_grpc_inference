#!/usr/bin/env python3
"""
Test raw gRPC communication to isolate the issue.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

# Create minimal request
request = inference_pb2.LayerRequestV2()
request.start_layer = 10
request.end_layer = 19

# Create proper tensor with MLX
import mlx.core as mx
from src.communication.tensor_utils import serialize_mlx_array

# Create MLX array
test_array = mx.random.normal((1, 5, 768))
print(f"Test array shape: {test_array.shape}")

# Serialize it
data, metadata = serialize_mlx_array(test_array, compress=True, compression_algorithm="lz4")

# Create tensor proto
tensor = inference_pb2.Tensor()
tensor.data = data
tensor.shape.extend(metadata['shape'])
tensor.dtype = metadata['dtype']
request.input_tensor.CopyFrom(tensor)

print(f"Request type: {type(request).__name__}")
print(f"Request proto name: {request.DESCRIPTOR.full_name}")

# Try to serialize/deserialize
serialized = request.SerializeToString()
print(f"Serialized size: {len(serialized)} bytes")

# Try to deserialize
deserialized = inference_pb2.LayerRequestV2()
deserialized.ParseFromString(serialized)
print(f"Deserialized successfully: start_layer={deserialized.start_layer}")

# Now test with gRPC
try:
    channel = grpc.insecure_channel("192.168.5.2:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    print("\nSending to worker...")
    response = stub.ProcessLayers(request, timeout=5)
    print("Success!")
    
except grpc.RpcError as e:
    print(f"\nRPC Error: {e.code()}")
    print(f"Details: {e.details()}")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
finally:
    if 'channel' in locals():
        channel.close()
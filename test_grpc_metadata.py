#!/usr/bin/env python3
"""
Check gRPC server is responsive.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("192.168.5.2:50051")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

# Test health check
try:
    response = stub.HealthCheck(inference_pb2.HealthRequest())
    print(f"✓ Health check successful: {response.status}")
    print(f"  Message: {response.message}")
except Exception as e:
    print(f"✗ Health check failed: {e}")

# Print what the stub expects for ProcessLayers
print(f"\nProcessLayers expects:")
print(f"  Request type: {stub.ProcessLayers._request_serializer}")
print(f"  Response type: {stub.ProcessLayers._response_deserializer}")

channel.close()
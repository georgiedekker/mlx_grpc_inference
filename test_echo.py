#!/usr/bin/env python3
"""
Test with a simple echo to verify gRPC is working.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

# Test GetDeviceInfo which should be simpler
try:
    channel = grpc.insecure_channel("192.168.5.2:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    print("Testing GetDeviceInfo RPC...")
    request = inference_pb2.Empty()
    response = stub.GetDeviceInfo(request, timeout=5)
    print(f"✓ Success! Device: {response.device_id}")
    
except grpc.RpcError as e:
    print(f"✗ RPC Error: {e.code()} - {e.details()}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
finally:
    if 'channel' in locals():
        channel.close()
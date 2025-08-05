#!/usr/bin/env python3
"""
Debug the distributed inference by checking layer assignments.
"""
import grpc
from src.communication import inference_pb2, inference_pb2_grpc

# Check what layers each worker has
workers = [
    ("192.168.5.1:50051", "mini1 (local)"),
    ("192.168.5.2:50051", "mini2"),
    ("192.168.5.3:50051", "m4"),
]

print("=== Layer Assignment Debug ===\n")

# Model has 28 layers
total_layers = 28
layers_per_worker = total_layers // 3
remainder = total_layers % 3

print(f"Total layers: {total_layers}")
print(f"Layers per worker: {layers_per_worker}")
print(f"Remainder: {remainder}\n")

# Calculate expected assignments
start_layer = 0
for i, (addr, name) in enumerate(workers):
    num_layers = layers_per_worker + (1 if i < remainder else 0)
    end_layer = start_layer + num_layers
    print(f"{name} ({addr}): layers {start_layer}-{end_layer-1}")
    start_layer = end_layer

print("\n=== Testing Worker Connectivity ===\n")

# Test each worker
for addr, name in workers:
    if addr == "192.168.5.1:50051":
        print(f"{name}: Local processing (no gRPC)")
        continue
        
    try:
        channel = grpc.insecure_channel(addr)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        # Try health check
        response = stub.HealthCheck(inference_pb2.HealthRequest(), timeout=2)
        print(f"{name}: ✓ Connected - {response.status}")
    except Exception as e:
        print(f"{name}: ✗ Error - {e}")
    finally:
        if 'channel' in locals():
            channel.close()
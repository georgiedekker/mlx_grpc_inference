#!/usr/bin/env python3
"""
Test basic gRPC connectivity between devices.
"""
import grpc
import sys
from src.communication import inference_pb2, inference_pb2_grpc

def test_connection(address):
    """Test gRPC connection to a worker."""
    print(f"Testing connection to {address}...")
    
    try:
        # Create channel with short timeout
        channel = grpc.insecure_channel(address, options=[
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
        ])
        
        # Create stub
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        # Test health check
        request = inference_pb2.HealthRequest()
        response = stub.HealthCheck(request, timeout=5)
        
        print(f"✓ Connected to {address}")
        print(f"  Status: {response.status}")
        print(f"  Message: {response.message}")
        return True
        
    except grpc.RpcError as e:
        print(f"✗ Failed to connect to {address}")
        print(f"  Error: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        print(f"✗ Failed to connect to {address}")
        print(f"  Error: {type(e).__name__}: {e}")
        return False
    finally:
        if 'channel' in locals():
            channel.close()

if __name__ == "__main__":
    # Test addresses
    addresses = [
        "192.168.5.1:50051",  # mini1 (local)
        "192.168.5.2:50051",  # mini2
        "192.168.5.3:50051",  # m4
    ]
    
    if len(sys.argv) > 1:
        # Test specific address
        addresses = [sys.argv[1]]
    
    print("=== gRPC Connectivity Test ===\n")
    
    success_count = 0
    for addr in addresses:
        if test_connection(addr):
            success_count += 1
        print()
    
    print(f"Summary: {success_count}/{len(addresses)} connections successful")
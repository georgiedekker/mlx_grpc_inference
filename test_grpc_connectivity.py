#!/usr/bin/env python3
"""
Test gRPC connectivity to worker nodes using .local hostnames.
"""

import grpc
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_grpc_connection(hostname, port=50051):
    """Test basic gRPC connectivity to a worker."""
    print(f"🔍 Testing gRPC connection to {hostname}:{port}")
    
    try:
        # Test different channel options to resolve DNS issues
        channel_options = [
            ('grpc.dns_enable_srv_queries', False),
            ('grpc.enable_retries', False),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
        ]
        
        # Try IPv4-only connection first
        channel = grpc.insecure_channel(
            f"{hostname}:{port}",
            options=channel_options
        )
        
        # Test connection with timeout
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
            print(f"✅ Channel ready to {hostname}")
            
            # Try to import and test the actual gRPC service
            from communication.inference_pb2_grpc import InferenceServiceStub
            from communication.inference_pb2 import Empty
            
            stub = InferenceServiceStub(channel)
            
            # Test health check
            response = stub.HealthCheck(Empty(), timeout=5.0)
            print(f"✅ Health check successful: {response.device_id}")
            
            return True
            
        except grpc.RpcError as e:
            print(f"❌ gRPC error: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
        finally:
            channel.close()
            
    except Exception as e:
        print(f"❌ Failed to create channel: {e}")
        return False

def test_dns_resolution():
    """Test DNS resolution for .local hostnames."""
    import socket
    
    hostnames = ["mini2.local", "master.local"]
    
    for hostname in hostnames:
        print(f"\n🔍 Testing DNS resolution for {hostname}")
        
        try:
            # Test IPv4 resolution
            ipv4_info = socket.getaddrinfo(hostname, 50051, socket.AF_INET)
            if ipv4_info:
                ip = ipv4_info[0][4][0]
                print(f"✅ IPv4: {hostname} -> {ip}")
            
            # Test IPv6 resolution
            try:
                ipv6_info = socket.getaddrinfo(hostname, 50051, socket.AF_INET6)
                if ipv6_info:
                    ip = ipv6_info[0][4][0]
                    print(f"✅ IPv6: {hostname} -> {ip}")
                else:
                    print(f"⚠️  No IPv6 address for {hostname}")
            except socket.gaierror:
                print(f"⚠️  IPv6 resolution failed for {hostname}")
                
        except socket.gaierror as e:
            print(f"❌ DNS resolution failed for {hostname}: {e}")

def test_with_ipv4_only():
    """Test gRPC connection forcing IPv4 only."""
    print("\n🔍 Testing with IPv4-only gRPC connections")
    
    # Set environment variable to prefer IPv4
    import os
    os.environ['GRPC_DNS_RESOLVER'] = 'native'
    
    hostnames = ["mini2.local", "master.local"]
    
    for hostname in hostnames:
        success = test_grpc_connection(hostname)
        if not success:
            print(f"⚠️  Retrying {hostname} with explicit IPv4...")
            
            # Try to get IPv4 address and connect directly
            try:
                import socket
                ipv4_info = socket.getaddrinfo(hostname, 50051, socket.AF_INET)
                if ipv4_info:
                    ip = ipv4_info[0][4][0]
                    print(f"📍 Trying direct IPv4 connection to {ip}")
                    success = test_grpc_connection(ip)
                    
            except Exception as e:
                print(f"❌ IPv4 fallback failed: {e}")

def main():
    """Run all connectivity tests."""
    print("🧪 gRPC Connectivity Test for .local Hostnames")
    print("=" * 60)
    
    # Test 1: DNS Resolution
    test_dns_resolution()
    
    print("\n" + "=" * 60)
    
    # Test 2: Direct gRPC connections
    hostnames = ["mini2.local", "master.local"]
    results = {}
    
    for hostname in hostnames:
        print(f"\n🔗 Testing gRPC connection to {hostname}")
        results[hostname] = test_grpc_connection(hostname)
    
    print("\n" + "=" * 60)
    
    # Test 3: IPv4-only approach if needed
    if not all(results.values()):
        test_with_ipv4_only()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    for hostname, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {hostname}: {status}")
    
    if all(results.values()):
        print("\n🎉 All gRPC connections successful!")
        print("The DNS issue might be specific to the coordinator startup.")
    else:
        print("\n⚠️  Some connections failed. Need to fix gRPC DNS resolution.")

if __name__ == "__main__":
    main()
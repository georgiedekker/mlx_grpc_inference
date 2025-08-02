#!/usr/bin/env python3
"""
Test the fixed gRPC client with DNS resolver.
"""

import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fixed_grpc_client():
    """Test the updated gRPC client with DNS resolution."""
    print("🧪 Testing Fixed gRPC Client with DNS Resolution")
    print("=" * 60)
    
    try:
        from core.config import ClusterConfig
        from communication.grpc_client import GRPCInferenceClient
        
        # Load cluster config
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        
        # Test workers
        workers = config.get_workers()
        
        for worker in workers:
            print(f"\n🔗 Testing connection to {worker.device_id} ({worker.hostname})")
            
            try:
                client = GRPCInferenceClient(worker, timeout=10.0)
                
                # Test health check
                health = client.health_check()
                if health.get('healthy'):
                    print(f"✅ Health check successful")
                    print(f"   Device ID: {health.get('device_id')}")
                    print(f"   Timestamp: {health.get('timestamp')}")
                else:
                    print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
                
                # Test device info
                try:
                    info = client.get_device_info()
                    print(f"✅ Device info retrieved")
                    print(f"   Role: {info.get('role')}")
                    print(f"   Assigned layers: {info.get('assigned_layers')}")
                    print(f"   GPU utilization: {info.get('gpu_utilization', 0):.1f}%")
                except Exception as e:
                    print(f"⚠️  Device info failed: {e}")
                
                client.close()
                
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

def test_connection_pool():
    """Test the connection pool functionality."""
    print("\n🏊 Testing Connection Pool")
    print("=" * 40)
    
    try:
        from core.config import ClusterConfig
        from communication.grpc_client import ConnectionPool
        
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        local_device_id = config.get_local_device_id()
        
        print(f"Local device: {local_device_id}")
        
        # Create connection pool
        pool = ConnectionPool(config, local_device_id)
        
        print(f"Connection pool created with {len(pool.clients)} connections")
        
        # Test each connection
        for device_id, client in pool.clients.items():
            print(f"\n📡 Testing pooled connection to {device_id}")
            
            try:
                health = client.health_check()
                if health.get('healthy'):
                    print(f"✅ Pooled connection healthy")
                else:
                    print(f"❌ Pooled connection unhealthy: {health.get('error')}")
            except Exception as e:
                print(f"❌ Pooled connection error: {e}")
        
        pool.close_all()
        print("\n🔐 All connections closed")
        
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")

def main():
    """Run all tests."""
    test_fixed_grpc_client()
    test_connection_pool()
    
    print("\n" + "=" * 60)
    print("🎯 If all tests pass, the coordinator should now start successfully!")

if __name__ == "__main__":
    main()
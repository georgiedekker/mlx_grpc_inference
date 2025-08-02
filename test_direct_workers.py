#!/usr/bin/env python3
"""
Direct test of worker connections to see if the GPUs actually light up.
"""

import asyncio
import sys
import grpc
import numpy as np
import mlx.core as mx
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array

async def test_worker_connection(hostname, port, device_name):
    """Test connection to a single worker."""
    print(f"\n🔍 Testing {device_name} ({hostname}:{port})...")
    
    # Resolve DNS to IP
    target = resolve_grpc_target(hostname, port)
    print(f"   DNS resolved: {hostname}:{port} -> {target}")
    
    try:
        # Create gRPC channel
        channel = grpc.insecure_channel(target)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        # Health check
        health_request = inference_pb2.Empty()
        health_response = stub.HealthCheck(health_request, timeout=5.0)
        print(f"   ✅ Health check: Device {health_response.device_id}, Healthy: {health_response.healthy}")
        
        # Get device info
        device_info = stub.GetDeviceInfo(health_request, timeout=5.0)
        print(f"   📊 Device info:")
        print(f"      - Hostname: {device_info.hostname}")
        print(f"      - Rank: {device_info.rank}")
        print(f"      - Role: {device_info.role}")
        print(f"      - Assigned layers: {list(device_info.assigned_layers)}")
        print(f"      - GPU utilization: {device_info.gpu_utilization:.1f}%")
        print(f"      - Memory usage: {device_info.memory_usage_gb:.1f} GB")
        
        # Test layer processing with a small tensor
        print(f"   🧠 Testing layer processing...")
        
        # Create a test tensor (batch_size=1, seq_len=10, hidden_size=2048)
        test_tensor = mx.random.normal(shape=(1, 10, 2048))
        data, metadata = serialize_mlx_array(test_tensor)
        
        # Create layer request
        layer_request = inference_pb2.LayerRequest(
            request_id=f"test-{device_name}",
            input_tensor=data,
            layer_indices=list(device_info.assigned_layers)[:2],  # Test first 2 assigned layers
            metadata=inference_pb2.TensorMetadata(
                shape=metadata['shape'],
                dtype=metadata['dtype'],
                compressed=metadata.get('compressed', False)
            )
        )
        
        # Send request and measure time
        import time
        start_time = time.time()
        layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
        processing_time = time.time() - start_time
        
        print(f"   ⚡ Layer processing completed in {processing_time:.2f}s")
        print(f"      - Request ID: {layer_response.request_id}")
        print(f"      - Output tensor shape: {list(layer_response.metadata.shape)}")
        print(f"      - Server processing time: {layer_response.processing_time_ms:.1f}ms")
        print(f"      - Device ID: {layer_response.device_id}")
        
        # Close channel
        channel.close()
        
        return True
        
    except grpc.RpcError as e:
        print(f"   ❌ gRPC error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_all_workers():
    """Test all workers to see if GPUs light up."""
    print("🚀 Testing Direct Worker Connections")
    print("="*50)
    
    workers = [
        ("mini2.local", 50051, "mini2 (Worker 1)"),
        ("master.local", 50051, "master (Worker 2)")
    ]
    
    results = []
    for hostname, port, device_name in workers:
        success = await test_worker_connection(hostname, port, device_name)
        results.append((device_name, success))
    
    print(f"\n📊 Summary:")
    print("="*50)
    all_working = True
    for device_name, success in results:
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {device_name}: {status}")
        if not success:
            all_working = False
    
    if all_working:
        print(f"\n🎉 All workers are responding and processing layers!")
        print(f"   Your M4 Mac GPUs should be lighting up during layer processing.")
        print(f"   You can monitor GPU usage with: python scripts/monitor_gpus.py")
    else:
        print(f"\n⚠️  Some workers are not responding. Check worker logs.")
    
    return all_working

if __name__ == "__main__":
    asyncio.run(test_all_workers())
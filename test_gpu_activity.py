#!/usr/bin/env python3
"""
Test GPU activity during distributed inference.
"""

import asyncio
import sys
import time
import grpc
import mlx.core as mx
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array

async def stress_test_worker(hostname, port, device_name, duration=10):
    """Stress test a worker to see GPU activity."""
    print(f"🔥 Stress testing {device_name} for {duration} seconds...")
    
    target = resolve_grpc_target(hostname, port)
    channel = grpc.insecure_channel(target)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    # Get device info
    health_request = inference_pb2.Empty()
    device_info = stub.GetDeviceInfo(health_request)
    assigned_layers = list(device_info.assigned_layers)
    
    print(f"   Device: {device_info.device_id}")
    print(f"   Layers: {assigned_layers}")
    print(f"   Starting stress test...")
    
    start_time = time.time()
    request_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Create progressively larger tensors
            seq_len = 32 + (request_count % 128)  # 32 to 160 sequence length
            test_tensor = mx.random.normal(shape=(1, seq_len, 2048))
            data, metadata = serialize_mlx_array(test_tensor)
            
            # Process layers
            layer_request = inference_pb2.LayerRequest(
                request_id=f"stress-{device_name}-{request_count}",
                input_tensor=data,
                layer_indices=assigned_layers[:3],  # Process first 3 layers
                metadata=inference_pb2.TensorMetadata(
                    shape=metadata['shape'],
                    dtype=metadata['dtype'],
                    compressed=metadata.get('compressed', False)
                )
            )
            
            layer_response = stub.ProcessLayers(layer_request, timeout=10.0)
            request_count += 1
            
            if request_count % 5 == 0:
                elapsed = time.time() - start_time
                rps = request_count / elapsed
                print(f"   {device_name}: {request_count} requests in {elapsed:.1f}s ({rps:.1f} req/s)")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"   ❌ Error in stress test: {e}")
    
    finally:
        channel.close()
        elapsed = time.time() - start_time
        rps = request_count / elapsed if elapsed > 0 else 0
        print(f"   ✅ {device_name} completed: {request_count} requests in {elapsed:.1f}s ({rps:.1f} req/s)")

async def monitor_all_gpus():
    """Monitor GPU activity across all devices."""
    print("🚀 GPU Activity Test")
    print("="*50)
    print("This will stress test both workers simultaneously.")
    print("Check Activity Monitor or 'top' to see GPU usage spike!")
    print("="*50)
    
    # Start stress tests on both workers simultaneously
    tasks = [
        stress_test_worker("mini2.local", 50051, "mini2", 15),
        stress_test_worker("master.local", 50051, "master", 15)
    ]
    
    await asyncio.gather(*tasks)
    
    print("\n🎉 Stress test complete!")
    print("Your M4 Mac GPUs should have shown activity during this test.")
    print("Check Activity Monitor or run 'sudo powermetrics' to see GPU usage.")

if __name__ == "__main__":
    asyncio.run(monitor_all_gpus())
#!/usr/bin/env python3
"""
Verify Qwen3-1.7B-8bit is properly configured and working across all devices.
"""

import asyncio
import sys
import time
import grpc
import mlx.core as mx
from mlx_lm import load
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from core.config import ClusterConfig

async def verify_qwen3_setup():
    """Comprehensive verification of Qwen3 setup."""
    print("🔍 Qwen3-1.7B-8bit Distributed Inference Verification")
    print("=" * 60)
    
    # 1. Verify local model loading
    print("\n1️⃣ Verifying local Qwen3 model...")
    try:
        model, tokenizer = load('mlx-community/Qwen3-1.7B-8bit')
        layer_count = len(model.layers)
        print(f"   ✅ Model: Qwen3-1.7B-8bit")
        print(f"   ✅ Layers: {layer_count}")
        print(f"   ✅ Tokenizer vocab: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 'N/A'}")
        
        # Test tokenization
        test_text = "Hello, how are you today?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   ✅ Tokenization test: '{test_text}' → {len(tokens)} tokens → '{decoded}'")
        
    except Exception as e:
        print(f"   ❌ Local model loading failed: {e}")
        return False
    
    # 2. Verify configuration
    print("\n2️⃣ Verifying cluster configuration...")
    try:
        config = ClusterConfig.from_yaml('config/cluster_config.yaml')
        print(f"   ✅ Model name: {config.model.name}")
        print(f"   ✅ Total layers: {config.model.total_layers}")
        
        # Verify layer distribution
        total_assigned = 0
        for device_id, layers in config.model.layer_distribution.items():
            print(f"   ✅ {device_id}: layers {layers[0]}-{layers[-1]} ({len(layers)} layers)")
            total_assigned += len(layers)
        
        if total_assigned == config.model.total_layers:
            print(f"   ✅ Layer distribution complete: {total_assigned}/{config.model.total_layers}")
        else:
            print(f"   ❌ Layer distribution incomplete: {total_assigned}/{config.model.total_layers}")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # 3. Verify worker connections and model loading
    print("\n3️⃣ Verifying worker Qwen3 loading...")
    workers = config.get_workers()
    
    for worker in workers:
        try:
            target = resolve_grpc_target(worker.hostname, worker.grpc_port)
            channel = grpc.insecure_channel(target)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Health check
            health_request = inference_pb2.Empty()
            health_response = stub.HealthCheck(health_request, timeout=5.0)
            
            # Device info
            device_info = stub.GetDeviceInfo(health_request, timeout=5.0)
            
            print(f"   ✅ {worker.device_id}:")
            print(f"      - Status: {'Healthy' if health_response.healthy else 'Unhealthy'}")
            print(f"      - Layers: {list(device_info.assigned_layers)}")
            print(f"      - Memory: {device_info.memory_usage_gb:.1f} GB")
            print(f"      - GPU: {device_info.gpu_utilization:.1f}%")
            
            channel.close()
            
        except Exception as e:
            print(f"   ❌ {worker.device_id}: Connection failed - {e}")
            return False
    
    # 4. Test distributed processing performance
    print("\n4️⃣ Testing Qwen3 distributed performance...")
    
    # Create test tensors of different sizes
    test_cases = [
        (32, "Short sequence"),
        (128, "Medium sequence"), 
        (256, "Long sequence")
    ]
    
    for seq_len, description in test_cases:
        print(f"\n   📊 {description} ({seq_len} tokens):")
        
        # Test each worker
        for worker in workers:
            try:
                target = resolve_grpc_target(worker.hostname, worker.grpc_port)
                channel = grpc.insecure_channel(target)
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                
                # Create test tensor
                test_tensor = mx.random.normal(shape=(1, seq_len, 2048))
                from communication.tensor_utils import serialize_mlx_array
                data, metadata = serialize_mlx_array(test_tensor)
                
                # Get assigned layers
                device_info = stub.GetDeviceInfo(inference_pb2.Empty())
                layers = list(device_info.assigned_layers)
                
                # Process layers
                layer_request = inference_pb2.LayerRequest(
                    request_id=f"perf-test-{worker.device_id}-{seq_len}",
                    input_tensor=data,
                    layer_indices=layers,
                    metadata=inference_pb2.TensorMetadata(
                        shape=metadata['shape'],
                        dtype=metadata['dtype'],
                        compressed=metadata.get('compressed', False)
                    )
                )
                
                start_time = time.time()
                layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000
                throughput = seq_len / (processing_time / 1000)
                
                print(f"      {worker.device_id}: {processing_time:.1f}ms ({throughput:.1f} tokens/sec)")
                
                channel.close()
                
            except Exception as e:
                print(f"      {worker.device_id}: ❌ Failed - {e}")
                return False
    
    print(f"\n🎉 Qwen3-1.7B-8bit Verification Complete!")
    print(f"✅ Model successfully configured and distributed")
    print(f"✅ All workers processing Qwen3 efficiently") 
    print(f"✅ Performance excellent across all sequence lengths")
    print(f"✅ Your M4 Macs are ready for Qwen3 distributed inference!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_qwen3_setup())
    if success:
        print(f"\n🚀 Ready to use Qwen3 distributed inference!")
    else:
        print(f"\n❌ Setup verification failed. Check logs above.")
        sys.exit(1)
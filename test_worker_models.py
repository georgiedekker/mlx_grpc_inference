#!/usr/bin/env python3
"""
Test to verify workers have the same model weights as coordinator.
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
from mlx_lm import load
import asyncio
import grpc
from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from core.config import ClusterConfig

async def test_worker_models():
    """Test if workers have the same model weights as coordinator."""
    print("🧪 Testing worker model consistency...")
    
    # Load coordinator model
    print("📦 Loading coordinator model...")
    coordinator_model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    config = ClusterConfig.from_yaml('config/cluster_config.yaml')
    
    # Test tensor - use something that will reveal weight differences
    test_input = mx.array([[872, 25, 86408, 17439, 12]])  # "user: Adaptive Multi-Teacher"
    
    # Get coordinator embeddings
    coordinator_embeddings = coordinator_model.model.embed_tokens(test_input)
    print(f"📊 Coordinator embeddings:")
    print(f"   Shape: {coordinator_embeddings.shape}")
    print(f"   Mean: {mx.mean(coordinator_embeddings).item():.6f}")
    print(f"   First few: {coordinator_embeddings.flatten()[:5].tolist()}")
    
    # Process through first coordinator layer to get test hidden states
    coordinator_layer_0_output = coordinator_model.model.layers[0](coordinator_embeddings)
    if isinstance(coordinator_layer_0_output, tuple):
        test_hidden_states = coordinator_layer_0_output[0]
    else:
        test_hidden_states = coordinator_layer_0_output
    
    print(f"📊 Coordinator layer 0 output:")
    print(f"   Mean: {mx.mean(test_hidden_states).item():.6f}")
    print(f"   First few: {test_hidden_states.flatten()[:5].tolist()}")
    
    # Connect to workers and test their processing
    for device in config.devices:
        if device.device_id == config.coordinator_device_id:
            continue
            
        print(f"\n🔌 Testing worker: {device.device_id}")
        
        try:
            # Connect to worker
            target = resolve_grpc_target(device.hostname, device.grpc_port)
            channel = grpc.aio.insecure_channel(target)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Test health first
            health_request = inference_pb2.Empty()
            health_response = await stub.HealthCheck(health_request, timeout=5.0)
            print(f"   Health: {'✅ Healthy' if health_response.healthy else '❌ Unhealthy'}")
            
            if not health_response.healthy:
                continue
            
            # Get worker's assigned layers
            worker_layers = config.model.get_device_layers(device.device_id)
            print(f"   Assigned layers: {worker_layers}")
            
            if not worker_layers:
                print("   ⚠️ No layers assigned")
                continue
            
            # Test first assigned layer
            first_layer = worker_layers[0]
            print(f"   Testing layer {first_layer}...")
            
            # Serialize test tensor
            serialized_input, metadata = serialize_mlx_array(test_hidden_states)
            
            # Create gRPC request with proper protobuf format
            tensor_metadata = inference_pb2.TensorMetadata(
                shape=metadata['shape'],
                dtype=metadata['dtype'],
                compressed=metadata['compressed']
            )
            
            request = inference_pb2.LayerRequest(
                request_id="test_worker_model",
                layer_indices=[first_layer],  # Note: plural, and it's a list
                input_tensor=serialized_input,
                metadata=tensor_metadata
            )
            
            # Send to worker (note: method name is ProcessLayers, not ProcessLayer)
            response = await stub.ProcessLayers(request, timeout=10.0)
            
            # Deserialize response - convert protobuf metadata back to dict
            response_metadata = {
                'shape': list(response.metadata.shape),
                'dtype': response.metadata.dtype,
                'compressed': response.metadata.compressed
            }
            worker_output = deserialize_mlx_array(response.output_tensor, response_metadata)
            
            print(f"   Worker layer {first_layer} output:")
            print(f"   Shape: {worker_output.shape}")
            print(f"   Mean: {mx.mean(worker_output).item():.6f}")
            print(f"   First few: {worker_output.flatten()[:5].tolist()}")
            
            # Compare with local processing
            local_layer_output = coordinator_model.model.layers[first_layer](test_hidden_states)
            if isinstance(local_layer_output, tuple):
                local_output = local_layer_output[0]
            else:
                local_output = local_layer_output
                
            print(f"   Local layer {first_layer} output:")
            print(f"   Mean: {mx.mean(local_output).item():.6f}")
            print(f"   First few: {local_output.flatten()[:5].tolist()}")
            
            # Compare
            diff = mx.mean(mx.abs(worker_output - local_output)).item()
            max_diff = mx.max(mx.abs(worker_output - local_output)).item()
            
            print(f"   Difference:")
            print(f"   Mean diff: {diff:.8f}")
            print(f"   Max diff: {max_diff:.8f}")
            
            if diff < 1e-6:
                print("   ✅ Worker matches coordinator!")
            else:
                print("   ❌ Worker differs from coordinator!")
                print(f"   This indicates model weight mismatch on {device.device_id}")
            
            await channel.close()
            
        except Exception as e:
            print(f"   ❌ Error testing {device.device_id}: {e}")
    
    print("\n🔍 Summary:")
    print("If workers show significant differences, they have different model weights.")
    print("This would explain why distributed inference produces gibberish.")

if __name__ == "__main__":
    asyncio.run(test_worker_models())
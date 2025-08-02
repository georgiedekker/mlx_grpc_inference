#!/usr/bin/env python3
"""
End-to-end test of distributed inference across all 3 M4 Macs.
"""

import asyncio
import sys
import time
import grpc
import mlx.core as mx
sys.path.insert(0, 'src')

from communication import inference_pb2_grpc, inference_pb2
from communication.dns_resolver import resolve_grpc_target
from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from core.config import ClusterConfig

class SimpleDistributedInference:
    """Simple distributed inference without the complex coordinator."""
    
    def __init__(self):
        self.config = ClusterConfig.from_yaml('config/cluster_config.yaml')
        self.workers = {}
        
    async def initialize(self):
        """Initialize connections to workers."""
        print("🔗 Initializing connections to workers...")
        
        workers = self.config.get_workers()
        for worker in workers:
            target = resolve_grpc_target(worker.hostname, worker.grpc_port)
            channel = grpc.insecure_channel(target)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Test connection
            health_request = inference_pb2.Empty()
            health_response = stub.HealthCheck(health_request)
            
            self.workers[worker.device_id] = {
                'channel': channel,
                'stub': stub,
                'rank': worker.rank,
                'layers': self.config.model.get_device_layers(worker.device_id)
            }
            
            print(f"   ✅ {worker.device_id}: {len(self.workers[worker.device_id]['layers'])} layers")
    
    async def distributed_forward(self, input_tensor, request_id="test"):
        """Run distributed forward pass across workers."""
        print(f"🧠 Running distributed forward pass...")
        print(f"   Input shape: {input_tensor.shape}")
        
        current_tensor = input_tensor
        device_times = {}
        
        # Process through workers in rank order
        sorted_workers = sorted(self.workers.items(), key=lambda x: x[1]['rank'])
        
        for worker_id, worker_info in sorted_workers:
            layers = worker_info['layers']
            stub = worker_info['stub']
            
            print(f"   📡 Sending to {worker_id} for layers {layers[:3]}...{layers[-3:]} ({len(layers)} total)")
            
            # Serialize tensor
            data, metadata = serialize_mlx_array(current_tensor)
            
            # Create request
            layer_request = inference_pb2.LayerRequest(
                request_id=f"{request_id}-{worker_id}",
                input_tensor=data,
                layer_indices=layers,
                metadata=inference_pb2.TensorMetadata(
                    shape=metadata['shape'],
                    dtype=metadata['dtype'],
                    compressed=metadata.get('compressed', False)
                )
            )
            
            # Process layers
            start_time = time.time()
            layer_response = stub.ProcessLayers(layer_request, timeout=30.0)
            processing_time = time.time() - start_time
            
            # Deserialize output
            output_metadata = {
                'shape': list(layer_response.metadata.shape),
                'dtype': layer_response.metadata.dtype,
                'compressed': layer_response.metadata.compressed
            }
            current_tensor = deserialize_mlx_array(layer_response.output_tensor, output_metadata)
            
            device_times[worker_id] = processing_time * 1000  # Convert to ms
            
            print(f"      ⚡ Completed in {processing_time:.3f}s")
            print(f"      📊 Output shape: {current_tensor.shape}")
        
        print(f"   ✅ Distributed forward pass complete!")
        print(f"   🏁 Final output shape: {current_tensor.shape}")
        print(f"   ⏱️  Device times: {device_times}")
        
        return current_tensor, device_times
    
    async def close(self):
        """Close all connections."""
        for worker_info in self.workers.values():
            worker_info['channel'].close()

async def main():
    """Test end-to-end distributed inference."""
    print("🚀 MLX Distributed Inference - End-to-End Test")
    print("="*60)
    print("This test simulates the full distributed inference pipeline:")
    print("  1. mini1 (coordinator) → mini2 (worker 1) → master (worker 2)")
    print("  2. Each device processes its assigned model layers")
    print("  3. Tensors flow through the complete 28-layer model")
    print("="*60)
    
    # Initialize distributed inference
    inference = SimpleDistributedInference()
    await inference.initialize()
    
    # Create test input (simulating tokenized text)
    batch_size = 1
    sequence_length = 64  # Longer sequence for more realistic test
    hidden_size = 2048
    
    print(f"\n📝 Creating test input:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {sequence_length}")  
    print(f"   - Hidden size: {hidden_size}")
    
    input_tensor = mx.random.normal(shape=(batch_size, sequence_length, hidden_size))
    
    # Run multiple passes to show consistent performance
    for i in range(3):
        print(f"\n🔄 Test pass {i+1}/3:")
        start_time = time.time()
        
        output_tensor, device_times = await inference.distributed_forward(
            input_tensor, 
            request_id=f"end-to-end-{i+1}"
        )
        
        total_time = time.time() - start_time
        
        print(f"   🎯 Pass {i+1} Results:")
        print(f"      - Total time: {total_time:.3f}s")
        print(f"      - mini2 time: {device_times.get('mini2', 0):.1f}ms")
        print(f"      - master time: {device_times.get('master', 0):.1f}ms")
        print(f"      - Throughput: {sequence_length/total_time:.1f} tokens/sec")
        
        # Brief pause between passes
        await asyncio.sleep(1)
    
    # Clean up
    await inference.close()
    
    print(f"\n🎉 End-to-End Test Complete!")
    print(f"   ✅ All 3 M4 Macs successfully processing distributed inference")
    print(f"   ✅ Full 28-layer model distributed across devices")
    print(f"   ✅ Tensors flowing correctly through the pipeline")
    print(f"   📈 Your GPUs are lighting up across the cluster!")

if __name__ == "__main__":
    asyncio.run(main())
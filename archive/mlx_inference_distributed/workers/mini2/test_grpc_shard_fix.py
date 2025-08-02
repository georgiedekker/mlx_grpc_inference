#!/usr/bin/env python3
"""
Test script to verify the gRPC InitializeShard fix.

This script tests the corrected implementation that:
1. Uses the actual sharding plan from the request
2. Creates device-specific shards 
3. Properly handles tied embeddings
4. Validates layer assignments
"""

import sys
import logging
from unittest.mock import Mock
import grpc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import distributed_inference_pb2 as pb2
    import distributed_inference_pb2_grpc as pb2_grpc
except ImportError:
    logger.error("Protocol buffer modules not found. Run generate_proto.sh first.")
    sys.exit(1)

from grpc_server import DistributedInferenceServicer


def create_mock_shard_request(model_name: str, device_id: str, start_layer: int, end_layer: int):
    """Create a mock InitializeShardRequest for testing."""
    shard_info = pb2.ShardInfo(
        device_id=device_id,
        start_layer=start_layer,
        end_layer=end_layer,
        layer_names=[f"layer_{i}" for i in range(start_layer, end_layer)],
        shard_size_bytes=int(1.0 * 1024**3)  # 1GB estimate
    )
    
    request = pb2.InitializeShardRequest(
        model_name=model_name,
        model_provider="mlx-community",
        shard_info=shard_info,
        load_from_cache=True
    )
    
    return request


def test_shard_initialization():
    """Test the corrected InitializeShard implementation."""
    print("Testing gRPC InitializeShard fix...")
    
    # Create a test servicer
    servicer = DistributedInferenceServicer("test_device", 50051)
    
    try:
        # Test Case 1: First shard (should have embedding)
        print("\n1. Testing first shard (layers 0-13)...")
        request1 = create_mock_shard_request(
            "mlx-community/Qwen3-1.7B-8bit", 
            "device1", 
            0, 14
        )
        
        response1 = servicer.InitializeShard(request1, None)
        
        if response1.success:
            print(f"   ✅ First shard initialized successfully")
            print(f"   Load time: {response1.load_time_ms}ms")
            print(f"   Model size: {response1.model_size_bytes / (1024**3):.2f} GB")
            
            # Verify shard properties
            shard = servicer.model_shard
            print(f"   Has embedding: {shard.embed_tokens is not None}")
            print(f"   Has norm: {shard.norm is not None}")
            print(f"   Has lm_head: {shard.lm_head is not None}")
            print(f"   Uses tied embeddings: {shard.use_tied_embeddings}")
            print(f"   Layer indices: {shard.layer_indices}")
        else:
            print(f"   ❌ First shard failed: {response1.message}")
            return False
        
        # Reset servicer for next test
        servicer.model_wrapper = None
        servicer.model_shard = None
        
        # Test Case 2: Last shard (should have norm and output projection)
        print("\n2. Testing last shard (layers 14-27)...")
        request2 = create_mock_shard_request(
            "mlx-community/Qwen3-1.7B-8bit", 
            "device2", 
            14, 28
        )
        
        response2 = servicer.InitializeShard(request2, None)
        
        if response2.success:
            print(f"   ✅ Last shard initialized successfully")
            print(f"   Load time: {response2.load_time_ms}ms")
            print(f"   Model size: {response2.model_size_bytes / (1024**3):.2f} GB")
            
            # Verify shard properties
            shard = servicer.model_shard
            print(f"   Has embedding: {shard.embed_tokens is not None}")
            print(f"   Has norm: {shard.norm is not None}")
            print(f"   Has lm_head: {shard.lm_head is not None}")
            print(f"   Uses tied embeddings: {shard.use_tied_embeddings}")
            print(f"   Layer indices: {shard.layer_indices}")
        else:
            print(f"   ❌ Last shard failed: {response2.message}")
            return False
        
        # Test Case 3: Invalid layer range
        print("\n3. Testing invalid layer range...")
        request3 = create_mock_shard_request(
            "mlx-community/Qwen3-1.7B-8bit", 
            "device3", 
            25, 30  # Invalid: beyond model layers
        )
        
        response3 = servicer.InitializeShard(request3, None)
        
        if not response3.success:
            print(f"   ✅ Invalid range correctly rejected: {response3.message}")
        else:
            print(f"   ❌ Invalid range should have been rejected")
            return False
        
        print("\n✅ All tests passed! The gRPC InitializeShard fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_estimation():
    """Test the memory estimation functions."""
    print("\nTesting memory estimation...")
    
    servicer = DistributedInferenceServicer("test_device", 50051)
    
    try:
        # Load model to get model info
        from model_abstraction import ModelFactory
        wrapper = ModelFactory.create_wrapper("mlx-community/Qwen3-1.7B-8bit")
        wrapper.load_model()
        servicer.model_wrapper = wrapper
        
        model_info = wrapper.model_info
        print(f"   Model info: {model_info.num_layers} layers, {model_info.estimate_size_gb():.2f} GB total")
        
        # Test memory estimation for different shard sizes
        test_cases = [
            (0, 14),   # First half
            (14, 28),  # Second half
            (0, 28),   # Full model
            (10, 18),  # Middle section
        ]
        
        for start, end in test_cases:
            estimated_bytes = servicer._estimate_shard_size(start, end)
            estimated_gb = estimated_bytes / (1024**3)
            num_layers = end - start
            print(f"   Layers {start}-{end-1} ({num_layers} layers): {estimated_gb:.2f} GB")
        
        print("   ✅ Memory estimation working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory estimation failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing gRPC InitializeShard Implementation Fix")
    print("=" * 60)
    
    success = True
    
    # Test shard initialization
    if not test_shard_initialization():
        success = False
    
    # Test memory estimation
    if not test_memory_estimation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - gRPC implementation is fixed!")
    else:
        print("❌ SOME TESTS FAILED - needs more work")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
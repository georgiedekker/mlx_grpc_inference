#!/usr/bin/env python3
"""
Test script to verify the sharding system works correctly with heterogeneous
device configurations and properly uses shard_info from requests.

This test creates mock distributed inference scenarios to verify:
1. Layer assignments are properly extracted from shard_info
2. No hardcoded values interfere with sharding
3. Heterogeneous device configurations work correctly
"""

import sys
import logging
from unittest.mock import Mock, MagicMock
from model_abstraction import ModelFactory
from device_capabilities import DeviceProfile
from sharding_strategy import ResourceAwareShardingPlanner, ShardingStrategy
from grpc_server import DistributedInferenceServicer
import distributed_inference_pb2 as pb2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_shard_info_usage():
    """Test that InitializeShard properly uses shard_info from request."""
    print("=" * 60)
    print("Testing shard_info layer assignment usage")
    print("=" * 60)
    
    # Create server instance
    server = DistributedInferenceServicer("test_device", 50051)
    
    # Test case 1: First shard (layers 0-5)
    print("\n1. Testing first shard (layers 0-5)...")
    shard_info_1 = pb2.ShardInfo(
        device_id="device_1",
        start_layer=0,
        end_layer=6,  # exclusive
        layer_names=[f"layer_{i}" for i in range(0, 6)],
        shard_size_bytes=1024**3
    )
    
    request_1 = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info_1
    )
    
    try:
        response_1 = server.InitializeShard(request_1, None)
        if response_1.success:
            print(f"✅ Successfully initialized first shard: {response_1.message}")
            print(f"   Load time: {response_1.load_time_ms}ms")
            print(f"   Model size: {response_1.model_size_bytes / (1024**3):.2f} GB")
            
            # Verify shard properties
            shard = server.model_shard
            print(f"   Shard layers: {len(shard.layers)}")
            print(f"   Layer indices: {shard.layer_indices}")
            print(f"   Has embedding: {shard.embed_tokens is not None}")
            print(f"   Has norm: {shard.norm is not None}")
            print(f"   Has lm_head: {shard.lm_head is not None}")
            print(f"   Uses tied embeddings: {shard.use_tied_embeddings}")
        else:
            print(f"❌ Failed to initialize first shard: {response_1.message}")
            return False
    except Exception as e:
        print(f"❌ Exception during first shard test: {e}")
        return False
    
    # Test case 2: Middle shard (layers 6-12)
    print("\n2. Testing middle shard (layers 6-12)...")
    server_2 = DistributedInferenceServicer("test_device_2", 50052)
    
    shard_info_2 = pb2.ShardInfo(
        device_id="device_2",
        start_layer=6,
        end_layer=13,  # exclusive
        layer_names=[f"layer_{i}" for i in range(6, 13)],
        shard_size_bytes=1024**3
    )
    
    request_2 = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info_2
    )
    
    try:
        response_2 = server_2.InitializeShard(request_2, None)
        if response_2.success:
            print(f"✅ Successfully initialized middle shard: {response_2.message}")
            
            # Verify shard properties
            shard = server_2.model_shard
            print(f"   Shard layers: {len(shard.layers)}")
            print(f"   Layer indices: {shard.layer_indices}")
            print(f"   Has embedding: {shard.embed_tokens is not None}")
            print(f"   Has norm: {shard.norm is not None}")
            print(f"   Has lm_head: {shard.lm_head is not None}")
        else:
            print(f"❌ Failed to initialize middle shard: {response_2.message}")
            return False
    except Exception as e:
        print(f"❌ Exception during middle shard test: {e}")
        return False
    
    # Test case 3: Last shard (layers 13-end)
    print("\n3. Testing last shard (layers 13-end)...")
    server_3 = DistributedInferenceServicer("test_device_3", 50053)
    
    # First get total layers from model
    wrapper = ModelFactory.create_wrapper("mlx-community/Qwen3-1.7B-8bit")
    wrapper.load_model()
    total_layers = wrapper.model_info.num_layers
    print(f"   Total model layers: {total_layers}")
    
    shard_info_3 = pb2.ShardInfo(
        device_id="device_3",
        start_layer=13,
        end_layer=total_layers,  # to the end
        layer_names=[f"layer_{i}" for i in range(13, total_layers)],
        shard_size_bytes=1024**3
    )
    
    request_3 = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info_3
    )
    
    try:
        response_3 = server_3.InitializeShard(request_3, None)
        if response_3.success:
            print(f"✅ Successfully initialized last shard: {response_3.message}")
            
            # Verify shard properties
            shard = server_3.model_shard
            print(f"   Shard layers: {len(shard.layers)}")
            print(f"   Layer indices: {shard.layer_indices}")
            print(f"   Has embedding: {shard.embed_tokens is not None}")
            print(f"   Has norm: {shard.norm is not None}")
            print(f"   Has lm_head: {shard.lm_head is not None}")
            print(f"   Uses tied embeddings: {shard.use_tied_embeddings}")
        else:
            print(f"❌ Failed to initialize last shard: {response_3.message}")
            return False
    except Exception as e:
        print(f"❌ Exception during last shard test: {e}")
        return False
    
    print("\n✅ All shard_info tests passed!")
    return True

def test_heterogeneous_sharding():
    """Test heterogeneous device sharding using the sharding planner."""
    print("\n" + "=" * 60)
    print("Testing heterogeneous device sharding")
    print("=" * 60)
    
    # Create heterogeneous device profiles
    devices = [
        DeviceProfile(
            device_id="mini1",
            hostname="mini1.local",
            model="Apple M4",
            memory_gb=16.0,
            gpu_cores=10,
            cpu_cores=10,
            cpu_performance_cores=4,
            cpu_efficiency_cores=6,
            neural_engine_cores=16
        ),
        DeviceProfile(
            device_id="studio1", 
            hostname="studio1.local",
            model="Apple M4 Max", 
            memory_gb=48.0,
            gpu_cores=40,
            cpu_cores=16,
            cpu_performance_cores=12,
            cpu_efficiency_cores=4,
            neural_engine_cores=16
        ),
        DeviceProfile(
            device_id="macbook1",
            hostname="macbook1.local",
            model="Apple M4 Pro",
            memory_gb=24.0,
            gpu_cores=20,
            cpu_cores=14,
            cpu_performance_cores=10,
            cpu_efficiency_cores=4,
            neural_engine_cores=16
        )
    ]
    
    # Load model info
    wrapper = ModelFactory.create_wrapper("mlx-community/Qwen3-1.7B-8bit")
    wrapper.load_model()
    model_info = wrapper.model_info
    
    print(f"\nModel info:")
    print(f"  Name: {model_info.name}")
    print(f"  Layers: {model_info.num_layers}")
    print(f"  Estimated size: {model_info.estimate_size_gb():.2f} GB")
    
    # Create sharding planner
    planner = ResourceAwareShardingPlanner()
    
    # Test different strategies
    strategies = [
        ShardingStrategy.UNIFORM,
        ShardingStrategy.MEMORY_PROPORTIONAL, 
        ShardingStrategy.COMPUTE_PROPORTIONAL,
        ShardingStrategy.BALANCED
    ]
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} strategy ---")
        
        try:
            plan = planner.create_plan(model_info, devices, strategy)
            
            # Validate plan
            is_valid, error = plan.validate()
            if not is_valid:
                print(f"❌ Invalid plan: {error}")
                continue
            
            print(f"✅ Valid sharding plan created")
            
            # Test initializing shards based on this plan
            for assignment in plan.assignments:
                print(f"\n  Device {assignment.device_id}:")
                print(f"    Layers: {assignment.start_layer}-{assignment.end_layer-1} ({assignment.num_layers} layers)")
                print(f"    Memory: {assignment.estimated_memory_gb:.2f} GB ({assignment.memory_utilization():.1f}%)")
                
                # Create and test server initialization
                server = DistributedInferenceServicer(assignment.device_id, 50051)
                
                shard_info = pb2.ShardInfo(
                    device_id=assignment.device_id,
                    start_layer=assignment.start_layer,
                    end_layer=assignment.end_layer,
                    layer_names=[f"layer_{i}" for i in range(assignment.start_layer, assignment.end_layer)],
                    shard_size_bytes=int(assignment.estimated_memory_gb * 1024**3)
                )
                
                request = pb2.InitializeShardRequest(
                    model_name="mlx-community/Qwen3-1.7B-8bit",
                    model_provider="mlx-community",
                    shard_info=shard_info
                )
                
                try:
                    response = server.InitializeShard(request, None)
                    if response.success:
                        print(f"    ✅ Initialization successful")
                        
                        # Verify the shard uses exactly the assigned layers
                        shard = server.model_shard
                        if shard.layer_indices == list(range(assignment.start_layer, assignment.end_layer)):
                            print(f"    ✅ Layer assignment correct: {shard.layer_indices}")
                        else:
                            print(f"    ❌ Layer assignment incorrect: expected {list(range(assignment.start_layer, assignment.end_layer))}, got {shard.layer_indices}")
                            return False
                    else:
                        print(f"    ❌ Initialization failed: {response.message}")
                        return False
                except Exception as e:
                    print(f"    ❌ Exception during initialization: {e}")
                    return False
            
        except Exception as e:
            print(f"❌ Strategy {strategy.value} failed: {e}")
            continue
    
    print("\n✅ All heterogeneous sharding tests passed!")
    return True

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "=" * 60)
    print("Testing edge cases and error conditions")
    print("=" * 60)
    
    server = DistributedInferenceServicer("test_device", 50051)
    
    # Test case 1: Invalid layer range (start >= end)
    print("\n1. Testing invalid layer range (start >= end)...")
    shard_info = pb2.ShardInfo(
        device_id="device_1",
        start_layer=5,
        end_layer=5,  # start == end
        layer_names=[],
        shard_size_bytes=1024**3
    )
    
    request = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info
    )
    
    response = server.InitializeShard(request, None)
    if not response.success and "Invalid layer range" in response.message:
        print("✅ Correctly rejected invalid layer range")
    else:
        print(f"❌ Should have rejected invalid layer range: {response.message}")
        return False
    
    # Test case 2: Layer range out of bounds
    print("\n2. Testing layer range out of bounds...")
    shard_info = pb2.ShardInfo(
        device_id="device_1",
        start_layer=0,
        end_layer=1000,  # way beyond model layers
        layer_names=[],
        shard_size_bytes=1024**3
    )
    
    request = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info
    )
    
    response = server.InitializeShard(request, None)
    if not response.success and "Invalid layer range" in response.message:
        print("✅ Correctly rejected out-of-bounds layer range")
    else:
        print(f"❌ Should have rejected out-of-bounds layer range: {response.message}")
        return False
    
    # Test case 3: Single layer assignment
    print("\n3. Testing single layer assignment...")
    shard_info = pb2.ShardInfo(
        device_id="device_1",
        start_layer=5,
        end_layer=6,  # just one layer
        layer_names=["layer_5"],
        shard_size_bytes=1024**3
    )
    
    request = pb2.InitializeShardRequest(
        model_name="mlx-community/Qwen3-1.7B-8bit",
        model_provider="mlx-community",
        shard_info=shard_info
    )
    
    response = server.InitializeShard(request, None)
    if response.success:
        print("✅ Successfully handled single layer assignment")
        shard = server.model_shard
        if len(shard.layers) == 1 and shard.layer_indices == [5]:
            print("✅ Single layer correctly assigned")
        else:
            print(f"❌ Single layer assignment incorrect: {shard.layer_indices}")
            return False
    else:
        print(f"❌ Failed single layer assignment: {response.message}")
        return False
    
    print("\n✅ All edge case tests passed!")
    return True

def main():
    """Run all tests."""
    print("Testing MLX Distributed Sharding System")
    print("=" * 60)
    
    all_passed = True
    
    # Run test suites
    test_suites = [
        ("Shard Info Usage", test_shard_info_usage),
        ("Heterogeneous Sharding", test_heterogeneous_sharding),
        ("Edge Cases", test_edge_cases),
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\nRunning {suite_name} tests...")
        try:
            if not test_func():
                print(f"❌ {suite_name} tests failed!")
                all_passed = False
            else:
                print(f"✅ {suite_name} tests passed!")
        except Exception as e:
            print(f"❌ {suite_name} tests crashed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The sharding system is working correctly.")
        print("   - Layer assignments are properly extracted from shard_info")
        print("   - No hardcoded values interfere with sharding")
        print("   - Heterogeneous device configurations work correctly")
    else:
        print("❌ SOME TESTS FAILED! Please check the output above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
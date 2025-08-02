#!/usr/bin/env python3
"""
Test basic component functionality before running distributed tests.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_loading():
    """Test configuration loading."""
    print("🔧 Testing configuration loading...")
    
    try:
        from core.config import ClusterConfig
        
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        print(f"✅ Loaded cluster config: {config.name}")
        print(f"   - Devices: {len(config.devices)}")
        print(f"   - Model: {config.model.name}")
        print(f"   - Coordinator: {config.coordinator_device_id}")
        
        # Test device detection
        local_device = config.get_local_device()
        if local_device:
            print(f"   - Local device: {local_device.device_id} ({local_device.role.value})")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_grpc_imports():
    """Test gRPC-related imports."""
    print("\n📡 Testing gRPC imports...")
    
    try:
        from communication.inference_pb2 import LayerRequest, LayerResponse, TensorMetadata
        from communication.inference_pb2_grpc import InferenceServiceStub
        print("✅ gRPC protobuf imports successful")
        
        # Test creating a basic message
        metadata = TensorMetadata(shape=[1, 10], dtype="float32", compressed=False)
        request = LayerRequest(
            request_id="test-123",
            input_tensor=b"dummy_data",
            layer_indices=[0, 1, 2],
            metadata=metadata
        )
        print(f"✅ Created test LayerRequest with ID: {request.request_id}")
        
        return True
    except Exception as e:
        print(f"❌ gRPC imports failed: {e}")
        return False

def test_tensor_utils():
    """Test tensor serialization utilities."""
    print("\n🔢 Testing tensor utilities...")
    
    try:
        import mlx.core as mx
        from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
        
        # Create a test tensor
        test_tensor = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print(f"✅ Created test tensor with shape: {test_tensor.shape}")
        
        # Test serialization
        serialized_data, metadata = serialize_mlx_array(test_tensor)
        print(f"✅ Serialized tensor, data size: {len(serialized_data)} bytes")
        
        # Test deserialization
        restored_tensor = deserialize_mlx_array(serialized_data, metadata)
        print(f"✅ Deserialized tensor with shape: {restored_tensor.shape}")
        
        # Verify data integrity
        if mx.allclose(test_tensor, restored_tensor):
            print("✅ Tensor serialization/deserialization verified")
            return True
        else:
            print("❌ Tensor data integrity check failed")
            return False
            
    except Exception as e:
        print(f"❌ Tensor utils test failed: {e}")
        return False

def test_model_loading():
    """Test model loading components."""
    print("\n🤖 Testing model loading...")
    
    try:
        # Import using absolute imports to avoid issues
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from model.loader import DistributedModelLoader
        from model.sharding import ModelShardingStrategy
        from core.config import ClusterConfig
        
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        
        # Test sharding strategy
        strategy = ModelShardingStrategy(config)
        print(f"✅ Created ModelShardingStrategy")
        
        # Test coverage validation
        if strategy.validate_coverage():
            print("✅ Layer coverage validation passed")
        else:
            print("❌ Layer coverage validation failed")
            return False
        
        # Test loader creation
        loader = DistributedModelLoader(config)
        print(f"✅ Created DistributedModelLoader")
        print(f"   - Model: {config.model.name}")
        print(f"   - Total layers: {config.model.total_layers}")
        
        # Test layer assignment
        local_device = config.get_local_device()
        if local_device:
            assigned_layers = config.model.get_device_layers(local_device.device_id)
            print(f"   - Assigned layers for {local_device.device_id}: {assigned_layers}")
            
            # Test shard info
            shard = strategy.get_device_shard(local_device.device_id)
            if shard:
                print(f"   - Shard info: layers {shard.start_layer}-{shard.end_layer-1}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all basic component tests."""
    print("🧪 MLX Distributed Inference - Basic Component Tests")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_grpc_imports,
        test_tensor_utils,
        test_model_loading,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic component tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
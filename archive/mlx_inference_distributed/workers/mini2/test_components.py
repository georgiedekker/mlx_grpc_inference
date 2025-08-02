#!/usr/bin/env python3
"""
Test script to verify individual components of the distributed MLX system.
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n=== Testing Imports ===")
    
    modules = [
        ("mlx", "MLX core"),
        ("grpc", "gRPC"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("yaml", "PyYAML"),
        ("mlx_lm", "MLX-LM"),
    ]
    
    failed = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
        except ImportError as e:
            print(f"❌ {description} ({module_name}): {e}")
            failed.append(module_name)
    
    return len(failed) == 0


def test_device_capabilities():
    """Test device capability detection."""
    print("\n=== Testing Device Capabilities ===")
    
    try:
        from device_capabilities import DeviceCapabilityDetector
        
        detector = DeviceCapabilityDetector()
        profile = detector.detect_capabilities()
        
        print(f"✅ Device ID: {profile.device_id}")
        print(f"✅ Model: {profile.model}")
        print(f"✅ Memory: {profile.memory_gb} GB")
        print(f"✅ GPU Cores: {profile.gpu_cores}")
        print(f"✅ CPU Cores: {profile.cpu_cores} (P: {profile.cpu_performance_cores}, E: {profile.cpu_efficiency_cores})")
        
        return True
    except Exception as e:
        print(f"❌ Failed to detect capabilities: {e}")
        return False


def test_model_abstraction():
    """Test model abstraction layer."""
    print("\n=== Testing Model Abstraction ===")
    
    try:
        from model_abstraction import ModelFactory
        
        # Test factory
        supported = ModelFactory.get_supported_models()
        print(f"✅ Supported models: {', '.join(supported)}")
        
        # Test model info (without loading)
        wrapper = ModelFactory.create_wrapper("mlx-community/Qwen3-1.7B-8bit")
        print(f"✅ Created wrapper for Qwen model")
        
        return True
    except Exception as e:
        print(f"❌ Model abstraction test failed: {e}")
        return False


def test_sharding_strategy():
    """Test sharding strategy calculations."""
    print("\n=== Testing Sharding Strategy ===")
    
    try:
        from sharding_strategy import ResourceAwareShardingPlanner, ShardingStrategy
        from device_capabilities import DeviceProfile
        from model_abstraction import ModelInfo
        
        # Create mock devices
        devices = [
            DeviceProfile(
                device_id="device1",
                hostname="device1.local",
                model="Apple M4",
                memory_gb=16.0,
                gpu_cores=10,
                cpu_cores=10,
                cpu_performance_cores=4,
                cpu_efficiency_cores=6,
                neural_engine_cores=16
            ),
            DeviceProfile(
                device_id="device2",
                hostname="device2.local",
                model="Apple M4 Pro",
                memory_gb=32.0,
                gpu_cores=20,
                cpu_cores=12,
                cpu_performance_cores=8,
                cpu_efficiency_cores=4,
                neural_engine_cores=16
            )
        ]
        
        # Create mock model info
        model_info = ModelInfo(
            name="test-model",
            architecture="qwen",
            num_layers=28,
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=50000,
            max_position_embeddings=2048,
            total_params=1_700_000_000,
            dtype="float16",
            quantization="int8"
        )
        
        # Test planner
        planner = ResourceAwareShardingPlanner()
        
        # Test different strategies
        for strategy in [ShardingStrategy.UNIFORM, ShardingStrategy.MEMORY_PROPORTIONAL]:
            plan = planner.create_plan(model_info, devices, strategy)
            print(f"\n✅ {strategy.value} strategy:")
            for assignment in plan.assignments:
                print(f"   - {assignment.device_id}: layers {assignment.start_layer}-{assignment.end_layer-1} "
                      f"({assignment.memory_utilization():.1f}% memory)")
        
        return True
    except Exception as e:
        print(f"❌ Sharding strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\n=== Testing Configuration ===")
    
    try:
        from distributed_config_v2 import DistributedConfig, create_example_config
        
        # Create example config
        config = create_example_config()
        
        # Validate
        is_valid, errors = config.validate()
        if is_valid:
            print(f"✅ Configuration valid")
        else:
            print(f"❌ Configuration errors: {errors}")
            return False
        
        # Test serialization
        config_dict = config.to_dict()
        config_loaded = DistributedConfig.from_dict(config_dict)
        
        print(f"✅ Configuration serialization works")
        print(f"✅ Devices: {len(config.devices)}")
        print(f"✅ Model: {config.model.name}")
        print(f"✅ Strategy: {config.sharding.strategy.value}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_protocol_buffers():
    """Test Protocol Buffer generation."""
    print("\n=== Testing Protocol Buffers ===")
    
    proto_file = Path("protos/distributed_inference.proto")
    if not proto_file.exists():
        print(f"❌ Protocol buffer definition not found at {proto_file}")
        return False
    
    print(f"✅ Protocol buffer definition exists")
    
    # Check if generated files exist
    pb2_file = Path("distributed_inference_pb2.py")
    pb2_grpc_file = Path("distributed_inference_pb2_grpc.py")
    
    if not pb2_file.exists() or not pb2_grpc_file.exists():
        print("⚠️  Generated Python files not found. Run ./generate_proto.sh")
        return False
    
    try:
        import distributed_inference_pb2
        import distributed_inference_pb2_grpc
        print("✅ Protocol buffer modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Failed to import protocol buffer modules: {e}")
        return False


def main():
    """Run all tests."""
    print("MLX Distributed System Component Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Device Capabilities", test_device_capabilities),
        ("Model Abstraction", test_model_abstraction),
        ("Sharding Strategy", test_sharding_strategy),
        ("Configuration", test_configuration),
        ("Protocol Buffers", test_protocol_buffers),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("  uv pip install grpcio grpcio-tools protobuf fastapi uvicorn pydantic pyyaml mlx-lm")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Generate protocol buffers: ./generate_proto.sh")
        print("  2. Create configuration: See distributed_config_example.json")
        print("  3. Launch cluster: ./launch_cluster.sh")


if __name__ == "__main__":
    main()
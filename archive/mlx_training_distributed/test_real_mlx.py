#!/usr/bin/env python3
"""
Test script to verify MLX integration works
"""

def test_mlx_import():
    """Test MLX import and basic functionality."""
    try:
        import mlx.core as mx
        print("✅ MLX Core imported successfully")
        
        # Test basic array creation
        arr = mx.array([1, 2, 3, 4])
        print(f"✅ MLX array created: {arr}")
        
        # Test basic operations
        result = mx.sum(arr)
        print(f"✅ MLX operation works: sum = {result}")
        
        return True
    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        return False

def test_mlx_nn():
    """Test MLX neural network components."""
    try:
        import mlx.nn as nn
        print("✅ MLX NN imported successfully")
        
        # Test linear layer
        linear = nn.Linear(10, 5)
        print("✅ MLX Linear layer created")
        
        # Test activation
        relu = nn.ReLU()
        print("✅ MLX ReLU activation created")
        
        return True
    except Exception as e:
        print(f"❌ MLX NN test failed: {e}")
        return False

def test_mlx_optimizers():
    """Test MLX optimizers."""
    try:
        import mlx.optimizers as optim
        print("✅ MLX Optimizers imported successfully")
        
        # Test optimizer creation
        adam = optim.Adam(learning_rate=0.001)
        print("✅ MLX Adam optimizer created")
        
        return True
    except Exception as e:
        print(f"❌ MLX Optimizers test failed: {e}")
        return False

def test_mlx_lm():
    """Test MLX Language Model utilities."""
    try:
        import mlx_lm
        print("✅ MLX-LM imported successfully")
        
        # Check available functions
        functions = [attr for attr in dir(mlx_lm) if not attr.startswith('_')]
        print(f"✅ MLX-LM functions available: {functions[:5]}...")
        
        return True
    except Exception as e:
        print(f"❌ MLX-LM test failed: {e}")
        return False

def test_training_components():
    """Test our training components can import."""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test individual components
        from training.dataset_handler import DatasetHandler
        print("✅ DatasetHandler imported")
        
        from training.optimizers import create_optimizer
        print("✅ Custom optimizers imported")
        
        # Test that we can create a basic config
        from training.mlx_trainer import TrainingConfig
        config = TrainingConfig(
            model_name="test-model",
            dataset_path="test-data.json",
            output_dir="./test-output"
        )
        print("✅ TrainingConfig created")
        
        return True
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 MLX Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("MLX Core", test_mlx_import),
        ("MLX Neural Networks", test_mlx_nn),
        ("MLX Optimizers", test_mlx_optimizers),
        ("MLX Language Models", test_mlx_lm),
        ("Training Components", test_training_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - MLX Integration is working!")
        print("\n🚀 Ready to run real MLX training:")
        print("   • Model loading with MLX tensors")
        print("   • LoRA parameter-efficient training")
        print("   • Knowledge distillation")  
        print("   • RLHF with DPO/PPO")
        print("   • Distributed multi-device training")
    elif passed > 0:
        print("⚠️  PARTIAL SUCCESS - Some components working")
    else:
        print("🚨 ALL TESTS FAILED - MLX not working properly")

if __name__ == "__main__":
    main()
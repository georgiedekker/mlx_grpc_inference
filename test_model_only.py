#!/usr/bin/env python3
"""
Test just the model loading and inference without the complex distributed components.
"""

import sys
import os
import time
import yaml
from pathlib import Path

def test_basic_model_functionality():
    """Test basic MLX model functionality."""
    print("🤖 Testing basic MLX model functionality...")
    
    try:
        # Test MLX import
        import mlx.core as mx
        print("✅ MLX imported successfully")
        
        # Test a simple MLX operation
        test_array = mx.array([[1, 2, 3], [4, 5, 6]])
        result = mx.sum(test_array)
        print(f"✅ MLX computation works: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        return False

def test_model_loading():
    """Test that we can load the target model."""
    print("\n📥 Testing model loading...")
    
    try:
        from mlx_lm import load
        
        model_name = "mlx-community/Qwen3-1.7B-8bit"
        print(f"Loading model: {model_name}")
        print("(This may take a while for first download...)")
        
        start_time = time.time()
        model, tokenizer = load(model_name)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Get model info
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"   - Number of layers: {num_layers}")
            
            # Check if it matches our config
            with open("config/cluster_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            expected_layers = config['model']['total_layers']
            
            if num_layers == expected_layers:
                print(f"   - ✅ Layer count matches config ({expected_layers})")
            else:
                print(f"   - ⚠️  Layer count mismatch: expected {expected_layers}, got {num_layers}")
        
        # Count parameters (try different methods)
        try:
            total_params = 0
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    total_params += param.size
            elif hasattr(model, 'trainable_parameters'):
                for _, param in model.trainable_parameters():
                    total_params += param.size
            else:
                # Alternative method for model info
                total_params = "unknown"
            print(f"   - Parameters: {total_params:,} ({total_params/1e6:.1f}M)" if isinstance(total_params, int) else f"   - Parameters: {total_params}")
        except Exception as e:
            print(f"   - Parameters: Unable to count ({e})")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_inference(model, tokenizer):
    """Test basic inference functionality."""
    print("\n💭 Testing inference...")
    
    try:
        from mlx_lm import generate
        
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain distributed computing."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            start_time = time.time()
            # Use simple generation with minimal parameters
            response = generate(
                model,
                tokenizer,
                prompt=prompt
            )
            inference_time = time.time() - start_time
            
            # Extract just the generated part (remove the prompt)
            if response.startswith(prompt):
                generated = response[len(prompt):].strip()
            else:
                generated = response
            
            print(f"   Response ({inference_time:.2f}s): {generated[:100]}...")
            
            # Calculate approximate speed
            token_count = len(generated.split())
            if inference_time > 0:
                tokens_per_second = token_count / inference_time
                print(f"   Speed: ~{tokens_per_second:.1f} tokens/sec")
        
        print("✅ All inference tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_distribution():
    """Test that our layer distribution makes sense."""
    print("\n📊 Testing layer distribution configuration...")
    
    try:
        with open("config/cluster_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        layer_distribution = config['model']['layer_distribution']
        total_layers = config['model']['total_layers']
        
        print(f"Total layers: {total_layers}")
        print("Layer distribution:")
        
        all_layers = set()
        for device, layers in layer_distribution.items():
            print(f"   - {device}: {layers} ({len(layers)} layers)")
            all_layers.update(layers)
        
        # Check coverage
        expected_layers = set(range(total_layers))
        missing = expected_layers - all_layers
        extra = all_layers - expected_layers
        
        if missing:
            print(f"❌ Missing layers: {sorted(missing)}")
            return False
        
        if extra:
            print(f"❌ Extra layers: {sorted(extra)}")
            return False
        
        if len(all_layers) == total_layers:
            print("✅ Layer distribution covers all layers correctly")
            return True
        else:
            print(f"❌ Layer count mismatch: expected {total_layers}, got {len(all_layers)}")
            return False
            
    except Exception as e:
        print(f"❌ Layer distribution test failed: {e}")
        return False

def test_gpu_availability():
    """Test if MLX can use the GPU."""
    print("\n🔥 Testing GPU availability...")
    
    try:
        import mlx.core as mx
        
        # Create some test data
        a = mx.random.normal((1000, 1000))
        b = mx.random.normal((1000, 1000))
        
        # Time a computation
        start_time = time.time()
        result = mx.matmul(a, b)
        mx.eval(result)  # Force evaluation
        compute_time = time.time() - start_time
        
        print(f"✅ GPU computation completed in {compute_time:.4f}s")
        print(f"   Result shape: {result.shape}")
        
        # Check if this seems reasonable for GPU speed
        if compute_time < 0.1:  # Should be very fast on M4 GPU
            print("✅ Performance suggests GPU acceleration is working")
        else:
            print("⚠️  Performance might indicate CPU fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all single device tests."""
    print("🧪 MLX Distributed Inference - Model & Performance Test")
    print("=" * 70)
    print("Testing core functionality before distributed setup...")
    print()
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic MLX functionality
    total_tests += 1
    if test_basic_model_functionality():
        tests_passed += 1
    
    # Test 2: GPU availability
    total_tests += 1
    if test_gpu_availability():
        tests_passed += 1
    
    # Test 3: Layer distribution
    total_tests += 1
    if test_layer_distribution():
        tests_passed += 1
    
    # Test 4: Model loading
    total_tests += 1
    model, tokenizer = test_model_loading()
    if model and tokenizer:
        tests_passed += 1
        
        # Test 5: Inference (only if model loaded)
        total_tests += 1
        if test_inference(model, tokenizer):
            tests_passed += 1
    else:
        total_tests += 1  # Count the skipped inference test
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready for distributed setup.")
        print("\n✅ Next steps:")
        print("1. Copy this project to mini2 and master devices")
        print("2. Set up virtual environments on all devices")
        print("3. Run cluster startup script")
        print("4. Test distributed inference")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
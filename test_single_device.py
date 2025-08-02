#!/usr/bin/env python3
"""
Test single device inference without distributed components.
"""

import sys
import os
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_loading():
    """Test that we can load the model locally."""
    print("🤖 Testing model loading...")
    
    try:
        from mlx_lm import load
        
        model_name = "mlx-community/Qwen3-1.7B-8bit"
        print(f"Loading model: {model_name}")
        
        start_time = time.time()
        model, tokenizer = load(model_name)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Get model info
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"   - Layers: {num_layers}")
        
        # Count parameters
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.size
        print(f"   - Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_simple_inference(model, tokenizer):
    """Test simple inference without distributed components."""
    print("\n💭 Testing simple inference...")
    
    try:
        from mlx_lm import generate
        
        prompt = "What is machine learning?"
        print(f"Prompt: {prompt}")
        
        start_time = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.2f}s")
        print(f"Response: {response}")
        
        # Count tokens (approximate)
        token_count = len(response.split())
        tokens_per_second = token_count / inference_time if inference_time > 0 else 0
        print(f"   - Tokens: ~{token_count}")
        print(f"   - Speed: {tokens_per_second:.1f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """Test the configuration system."""
    print("\n⚙️ Testing configuration system...")
    
    try:
        from core.config import ClusterConfig
        from model.sharding import ModelShardingStrategy
        
        config = ClusterConfig.from_yaml("config/cluster_config.yaml")
        print(f"✅ Config loaded: {config.name}")
        print(f"   - Devices: {len(config.devices)}")
        
        # Test sharding
        strategy = ModelShardingStrategy(config)
        if strategy.validate_coverage():
            print("✅ Layer sharding valid")
            
            # Show layer distribution
            for device in config.devices:
                layers = config.model.get_device_layers(device.device_id)
                print(f"   - {device.device_id}: layers {layers}")
        else:
            print("❌ Layer sharding invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run single device tests."""
    print("🧪 MLX Distributed Inference - Single Device Test")
    print("=" * 60)
    print("Testing basic functionality before distributed setup...")
    print()
    
    # Test configuration first
    if not test_config_system():
        print("\n❌ Configuration tests failed")
        sys.exit(1)
    
    # Test model loading
    model, tokenizer = test_model_loading()
    if not model or not tokenizer:
        print("\n❌ Model loading failed")
        sys.exit(1)
    
    # Test inference
    if not test_simple_inference(model, tokenizer):
        print("\n❌ Inference tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All single device tests passed!")
    print("✅ Ready for distributed setup")
    print("\nNext steps:")
    print("1. Copy project to other devices")
    print("2. Set up virtual environments on all devices")
    print("3. Run the cluster startup script")

if __name__ == "__main__":
    main()
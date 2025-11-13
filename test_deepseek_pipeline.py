#!/usr/bin/env python3
"""
Test DeepSeek pipeline with actual DeepSeek model
This should work since DeepSeek has native pipeline() support
"""
import mlx.core as mx
from mlx_lm import load
import sys

def test_deepseek():
    # Try the smallest DeepSeek model
    model_name = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    
    # Check structure
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, 'model'):
        print(f"Inner model type: {type(model.model).__name__}")
        print(f"Has pipeline: {hasattr(model.model, 'pipeline')}")
        
        if hasattr(model.model, 'pipeline'):
            print("✅ DeepSeek has native pipeline() support!")
            
            # Initialize distributed
            group = mx.distributed.init()
            if group:
                print(f"Distributed group: size={group.size()}, rank={group.rank()}")
                # This should work!
                model.model.pipeline(group)
                print("✅ Pipeline setup successful!")
            else:
                print("❌ No distributed group (need to run with mlx.launch)")
    else:
        print("❌ Model structure different than expected")

if __name__ == "__main__":
    test_deepseek()
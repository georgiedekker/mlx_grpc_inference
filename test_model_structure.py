#!/usr/bin/env python3
"""Test model structure to understand layer arrangement"""
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

print(f"Model type: {type(model)}")
print(f"Has layers: {hasattr(model, 'layers')}")
if hasattr(model, 'model'):
    print(f"Model.model type: {type(model.model)}")
    print(f"Model.model has layers: {hasattr(model.model, 'layers')}")
    if hasattr(model.model, 'layers'):
        print(f"Number of layers: {len(model.model.layers)}")
        print(f"Layer 0 type: {type(model.model.layers[0])}")
        
print("\nModel structure:")
for name, module in model.named_modules():
    if name:
        print(f"  {name}: {type(module).__name__}")
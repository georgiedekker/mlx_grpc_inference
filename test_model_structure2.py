#!/usr/bin/env python3
"""Test model structure to understand lm_head access."""
from mlx_lm import load

model_name = "mlx-community/Qwen3-1.7B-8bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name, lazy=True)

print("\nSearching for lm_head...")
# Check all attributes
for attr in dir(model):
    if 'head' in attr.lower() or 'lm' in attr.lower():
        print(f"  Found: model.{attr}")
        
# Check if it's in the dict
if 'lm_head' in model:
    print("  ✓ Found model['lm_head']")

# Print actual model keys
print(f"\nModel dict keys: {list(model.keys())[:10]}...")

# Check what happens when we access layers
print(f"\nNumber of layers: {len(model.model.layers)}")
print(f"Model has embed_tokens: {hasattr(model.model, 'embed_tokens')}")
print(f"Model has norm: {hasattr(model.model, 'norm')}")
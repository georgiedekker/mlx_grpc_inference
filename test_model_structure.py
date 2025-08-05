#!/usr/bin/env python3
"""Test model structure to understand lm_head access."""
from mlx_lm import load

model_name = "mlx-community/Qwen3-1.7B-8bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name, lazy=True)

print("\nModel structure:")
print(f"Type of model: {type(model)}")
print(f"Model attributes: {[x for x in dir(model) if not x.startswith('_')]}")

if hasattr(model, 'lm_head'):
    print(f"\n✓ model.lm_head exists")
elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
    print(f"\n✓ model.model.lm_head exists")
else:
    print("\n✗ lm_head not found in expected locations")
    
# Check what the model object actually has
print(f"\nChecking model components:")
if hasattr(model, 'model'):
    print(f"  model.model type: {type(model.model)}")
    print(f"  model.model attributes: {[x for x in dir(model.model) if not x.startswith('_') and not x.startswith('layers')][:10]}")
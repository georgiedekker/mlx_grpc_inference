#!/usr/bin/env python3
"""
Test if DeepSeek models work with distributed inference
"""
import mlx.core as mx
from mlx_lm import load

# Try to load a small DeepSeek model and check if it has pipeline
model_name = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"

print(f"Loading {model_name}...")
model, tokenizer = load(model_name)

# Check if model has pipeline method
if hasattr(model, 'model'):
    inner = model.model
    has_pipeline = hasattr(inner, 'pipeline')
    print(f"model.model has pipeline: {has_pipeline}")
    if has_pipeline:
        print(f"Model class: {type(inner).__name__}")
else:
    has_pipeline = hasattr(model, 'pipeline')
    print(f"model has pipeline: {has_pipeline}")
    if has_pipeline:
        print(f"Model class: {type(model).__name__}")

if not has_pipeline:
    print("\n❌ This model does NOT have native pipeline support")
    print("This is why distributed doesn't work properly!")
else:
    print("\n✅ This model HAS native pipeline support")
    print("This should work with distributed inference!")

# Check model size
params = sum(p.size for _, p in model.parameters())
print(f"\nModel parameters: {params/1e9:.2f}B")
memory = mx.get_active_memory() / (1024**3)
print(f"GPU memory used: {memory:.2f}GB")
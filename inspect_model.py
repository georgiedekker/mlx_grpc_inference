#!/usr/bin/env python3
"""Inspect the actual model structure to find lm_head."""
from mlx_lm import load
import mlx.core as mx

model_name = "mlx-community/Qwen3-1.7B-8bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name, lazy=True)

print("\n=== Model Structure ===")
print(f"Model type: {type(model)}")
print(f"Model class name: {model.__class__.__name__}")

# Check if model is a dictionary or object
if hasattr(model, '__dict__'):
    print(f"\nModel attributes: {list(model.__dict__.keys())}")

# Check for lm_head in various ways
print("\n=== Checking for lm_head ===")
print(f"hasattr(model, 'lm_head'): {hasattr(model, 'lm_head')}")
print(f"'lm_head' in dir(model): {'lm_head' in dir(model)}")

# If model has lm_head, check if it's None
if hasattr(model, 'lm_head'):
    print(f"model.lm_head is None: {model.lm_head is None}")
    print(f"type(model.lm_head): {type(model.lm_head)}")

# Check model.model structure
print("\n=== Checking model.model ===")
if hasattr(model, 'model'):
    print(f"model.model type: {type(model.model)}")
    print(f"model.model attributes: {[x for x in dir(model.model) if not x.startswith('_')][:15]}")
    
# Check if it's using tied embeddings
print("\n=== Checking embeddings ===")
if hasattr(model, 'args'):
    print(f"model.args: {model.args}")
    if hasattr(model.args, 'tie_word_embeddings'):
        print(f"tie_word_embeddings: {model.args.tie_word_embeddings}")

# Try to access the output projection
print("\n=== Testing output projection ===")
try:
    # Create a dummy tensor
    dummy = mx.random.normal((1, 1, 2048))  # Assuming hidden size is 2048
    
    # Try different ways to get output
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        print("✓ Can use model.lm_head")
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        print("✓ Can use model.model.embed_tokens.as_linear")
        # Test if as_linear works
        try:
            result = model.model.embed_tokens.as_linear(dummy)
            print(f"  Output shape: {result.shape}")
        except Exception as e:
            print(f"  Error with as_linear: {e}")
except Exception as e:
    print(f"Error testing output projection: {e}")

# Print the actual __call__ method if available
print("\n=== Checking __call__ method ===")
if hasattr(model, '__call__'):
    import inspect
    try:
        source = inspect.getsource(model.__call__)
        print("Found __call__ method")
        # Look for lm_head usage
        if 'lm_head' in source:
            print("✓ __call__ uses lm_head")
        if 'embed_tokens.as_linear' in source:
            print("✓ __call__ uses embed_tokens.as_linear")
    except:
        print("Could not inspect __call__ source")
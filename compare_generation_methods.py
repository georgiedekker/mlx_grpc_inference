#!/usr/bin/env python3
"""Compare manual forward pass vs MLX generate."""
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.qwen3 import Model as Qwen3Model

def test_generation_methods():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test prompt
    messages = [{"role": "user", "content": "Hello!"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens}")
    
    # Method 1: Use MLX generate (WORKS)
    print("\n=== METHOD 1: MLX GENERATE ===")
    result = generate(model, tokenizer, prompt=prompt, max_tokens=5)
    print(f"Result: {result}")
    
    # Method 2: Manual forward pass (our approach)
    print("\n=== METHOD 2: MANUAL FORWARD (OUR APPROACH) ===")
    input_ids = mx.array([tokens])
    
    # Check model structure
    print(f"Model type: {type(model)}")
    print(f"Model.model type: {type(model.model)}")
    
    # Try to call the model directly
    print("\n=== METHOD 3: DIRECT MODEL CALL ===")
    try:
        # This is how MLX likely calls it internally
        logits = model(input_ids)
        mx.eval(logits)
        print(f"Direct call logits shape: {logits.shape}")
        
        # Get prediction
        top_token_id = mx.argmax(logits[0, -1, :]).item()
        top_token = tokenizer.decode([int(top_token_id)])
        print(f"Top predicted token: '{top_token}'")
    except Exception as e:
        print(f"Direct call failed: {e}")
    
    # Method 4: Check if there's a cache or state we're missing
    print("\n=== METHOD 4: CHECK FOR CACHE/STATE ===")
    # Look for cache in model
    if hasattr(model, 'cache'):
        print(f"Model has cache: {model.cache}")
    if hasattr(model, 'state'):
        print(f"Model has state: {model.state}")
    
    # Check if model.model has a forward method
    if hasattr(model.model, '__call__'):
        print("model.model is callable")
        try:
            # Try calling model.model directly
            hidden = model.model.embed_tokens(input_ids)
            
            # Check if there's a special way to process layers
            print(f"Number of layers: {len(model.model.layers)}")
            
            # Process through model.model if it has a forward
            if hasattr(model.model, 'forward'):
                print("model.model has forward method")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_generation_methods()
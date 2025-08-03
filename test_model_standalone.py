#!/usr/bin/env python3
"""
Test the model standalone without distributed processing to verify it works correctly.
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def test_model_standalone():
    print("🧪 Testing model standalone (no distributed processing)...")
    
    # Load model
    print("📦 Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    print(f"✅ Model loaded: {len(model.layers)} layers")
    
    # Test prompt
    prompt = "user: Adaptive Multi-Teacher Multi-level Knowledge Distillation\nassistant: "
    print(f"🔤 Prompt: {repr(prompt)}")
    
    # Tokenize
    input_ids = mx.array(tokenizer.encode(prompt))
    print(f"🔢 Input tokens: {input_ids.shape} - {input_ids.tolist()[:10]}...")
    
    # Forward pass
    print("🔄 Running forward pass...")
    
    # Get embeddings
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]  # Add batch dimension
    
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"📊 Embeddings shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    
    # Process through all layers
    for i, layer in enumerate(model.model.layers):
        hidden_states = layer(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        if i < 3 or i >= len(model.model.layers) - 3:
            print(f"   Layer {i}: {hidden_states.shape}, mean: {mx.mean(hidden_states).item():.6f}")
    
    # Final norm
    hidden_states = model.model.norm(hidden_states)
    print(f"📊 After norm: {hidden_states.shape}, mean: {mx.mean(hidden_states).item():.6f}")
    
    # Get logits
    logits = model.model.embed_tokens.as_linear(hidden_states)
    print(f"📊 Logits: {logits.shape}, mean: {mx.mean(logits).item():.6f}")
    
    # Sample token
    sampler = make_sampler(temp=0.1, top_p=1.0)
    next_token = sampler(logits[:, -1:, :])
    token_id = next_token.item()
    
    # Decode
    token_text = tokenizer.decode([token_id])
    print(f"🎯 Next token: {token_id} -> {repr(token_text)}")
    
    # Generate using mlx_lm  
    print("\n🤖 Testing with mlx_lm generate...")
    try:
        response = generate(
            model, 
            tokenizer, 
            prompt="Adaptive Multi-Teacher Multi-level Knowledge Distillation",
            max_tokens=50
        )
        print(f"✅ mlx_lm output: {repr(response)}")
    except Exception as e:
        print(f"❌ mlx_lm generate failed: {e}")
        
        # Manual generation loop
        print("🔄 Trying manual generation...")
        current_prompt = "Adaptive Multi-Teacher Multi-level Knowledge Distillation"
        current_ids = mx.array(tokenizer.encode(current_prompt))[None, :]
        
        for i in range(10):  # Generate 10 tokens manually
            # Process through model
            hidden_states = model.model.embed_tokens(current_ids)
            for layer in model.model.layers:
                hidden_states = layer(hidden_states)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            hidden_states = model.model.norm(hidden_states)
            logits = model.model.embed_tokens.as_linear(hidden_states)
            
            # Sample next token
            sampler = make_sampler(temp=0.1)
            next_token = sampler(logits[:, -1:, :])
            token_id = next_token.item()
            
            # Decode and append
            token_text = tokenizer.decode([token_id])
            current_prompt += token_text
            
            # Update input for next iteration
            current_ids = mx.concatenate([current_ids, mx.array([[token_id]])], axis=1)
            
            print(f"   Token {i+1}: {token_id} -> {repr(token_text)}")
        
        print(f"✅ Manual generation: {repr(current_prompt)}")

if __name__ == "__main__":
    test_model_standalone()
#!/usr/bin/env python3
"""
Test mlx_lm generate directly to see what correct output should be.
"""

from mlx_lm import load, generate

def test_direct_generation():
    print("🧪 Testing direct mlx_lm generation...")
    
    # Load model
    model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")
    
    # Test prompt
    prompt = "Adaptive Multi-Teacher Multi-level Knowledge Distillation"
    
    print(f"🔤 Prompt: {repr(prompt)}")
    
    # Generate with mlx_lm
    print("🤖 Generating with mlx_lm...")
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=50
    )
    
    print(f"✅ Response: {repr(response)}")
    
    # Test with different chat formats
    chat_prompt1 = "user: Adaptive Multi-Teacher Multi-level Knowledge Distillation\nassistant:"
    chat_prompt2 = "Human: Adaptive Multi-Teacher Multi-level Knowledge Distillation\nAssistant:"
    chat_prompt3 = "Q: Adaptive Multi-Teacher Multi-level Knowledge Distillation\nA:"
    
    for i, chat_prompt in enumerate([chat_prompt1, chat_prompt2, chat_prompt3], 1):
        print(f"\\n🔤 Chat format {i}: {repr(chat_prompt)}")
        
        chat_response = generate(
            model,
            tokenizer, 
            prompt=chat_prompt,
            max_tokens=30
        )
        
        print(f"✅ Response {i}: {repr(chat_response)}")
    
    # Test the exact prompt our API is sending
    api_prompt = "user: Adaptive Multi-Teacher Multi-level Knowledge Distillation\\nassistant: "
    print(f"\\n🔤 API format: {repr(api_prompt)}")
    
    api_response = generate(
        model,
        tokenizer,
        prompt=api_prompt,
        max_tokens=30
    )
    
    print(f"✅ API response: {repr(api_response)}")

if __name__ == "__main__":
    test_direct_generation()
#!/usr/bin/env python3
"""
Test client for MoE distributed inference API
"""

import requests
import json
import time

def test_chat():
    """Test the chat completions endpoint"""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("🔄 Sending request to MoE API...")
    print(f"Prompt: {payload['messages'][0]['content']}")
    
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Response received in {elapsed:.2f}s")
        print(f"Model: {result['model']}")
        print(f"Response: {result['choices'][0]['message']['content']}")
        
        # Calculate tokens per second
        tokens = payload['max_tokens']
        print(f"Performance: {tokens/elapsed:.1f} tokens/sec")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def check_status():
    """Check server status"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            status = response.json()
            print("\n📊 Server Status:")
            print(f"  Model: {status['model']}")
            print(f"  Rank: {status['rank']}")
            print(f"  Layers: {status['layers']}")
            print(f"  GPU Memory: {status['gpu_memory_gb']} GB")
            print(f"  Distributed: {status['distributed']}")
            return True
    except:
        print("❌ Server not responding")
        return False

if __name__ == "__main__":
    print("MoE Distributed Inference API Test")
    print("==================================")
    
    # Check status first
    if check_status():
        print("\n" + "="*40)
        # Test chat endpoint
        test_chat()
    else:
        print("\nPlease start the server first with: ./launch_moe_api.sh")
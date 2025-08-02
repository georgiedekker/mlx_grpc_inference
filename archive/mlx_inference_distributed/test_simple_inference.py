#!/usr/bin/env python3
"""Simple test script for distributed inference with detailed logging."""

import requests
import json
import time

def test_inference():
    """Test simple inference with logging."""
    url = "http://localhost:8100/v1/chat/completions"
    
    # Simple test payload
    payload = {
        "model": "mlx-community/Qwen3-1.7B-8bit",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    print("Sending inference request...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nSuccess! Response received in {elapsed:.2f} seconds")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Calculate tokens per second if available
            if 'usage' in result:
                total_tokens = result['usage'].get('total_tokens', 0)
                if total_tokens > 0:
                    tps = total_tokens / elapsed
                    print(f"\nTokens per second: {tps:.2f}")
        else:
            print(f"\nError {response.status_code}: {response.text}")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            
    except requests.exceptions.Timeout:
        print(f"\nRequest timed out after {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    test_inference()
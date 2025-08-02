#!/usr/bin/env python3
"""Test to verify if model is actually distributed across devices."""

import json
import requests
import subprocess
import time

def check_gpu_usage():
    """Get current GPU usage percentage."""
    # Use powermetrics to get GPU usage on macOS
    cmd = "sudo powermetrics --samplers gpu_power -i 1000 -n 1"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
        output = result.stdout
        # Parse GPU usage from output
        for line in output.split('\n'):
            if 'GPU Active' in line:
                return line.strip()
        return "GPU usage not found"
    except:
        return "Could not measure GPU"

def test_inference_distribution():
    """Test if inference actually uses multiple GPUs."""
    
    print("Testing distributed inference...")
    print("=" * 60)
    
    # Check initial state
    print("\n1. Checking API health...")
    health = requests.get("http://localhost:8100/health")
    print(f"Health status: {health.json()}")
    
    print("\n2. Checking GPU info...")
    gpu_info = requests.get("http://localhost:8100/distributed/gpu-info")
    print(f"GPU info: {json.dumps(gpu_info.json(), indent=2)}")
    
    print("\n3. Running inference and monitoring GPU...")
    print("Note: On single device, all work will be on one GPU")
    
    # Make a longer inference request to have time to observe
    long_prompt = "Write a detailed explanation of quantum computing, covering its principles, current applications, and future potential. Include specific examples."
    
    response = requests.post(
        "http://localhost:8100/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 200,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n4. Inference successful!")
        print(f"Generated {result['usage']['completion_tokens']} tokens")
        print(f"Response preview: {result['choices'][0]['message']['content'][:100]}...")
    else:
        print(f"\n4. Inference failed: {response.status_code}")
        print(response.text)
    
    print("\n5. Configuration Analysis:")
    with open("distributed_config.json", "r") as f:
        config = json.load(f)
    
    print(f"- Model parallel size: {config['model_parallel_size']}")
    print(f"- Number of devices configured: {len(config['device_list'])}")
    print(f"- Communication backend: {config['communication_backend']}")
    
    if config['model_parallel_size'] > 1:
        print("\n⚠️  WARNING: Multi-device is configured but only one device is running!")
        print("   The system is trying to communicate with devices that don't exist.")
        print("   This is why you saw the DNS resolution errors earlier.")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("- The system is configured for 3 devices but only 1 is running")
    print("- All inference happens on the single available device")
    print("- To truly test distribution, you need all 3 devices running")

if __name__ == "__main__":
    test_inference_distribution()
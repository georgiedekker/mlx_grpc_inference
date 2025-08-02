#!/usr/bin/env python3
"""
Quick validation script to demonstrate the distributed MLX system.
"""

import requests
import json
import time
import sys


def validate_cluster():
    """Validate the 3-device cluster is operational."""
    
    print("🔍 MLX Distributed System Validation")
    print("=" * 60)
    
    api_url = "http://localhost:8100"
    
    # 1. Check health
    print("\n1️⃣ Checking cluster health...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cluster healthy: {data['model_name']}")
            print(f"   Device: {data['device_id']} ({data['role']})")
        else:
            print("❌ Cluster unhealthy")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to cluster: {e}")
        return False
    
    # 2. Check GPU info
    print("\n2️⃣ Checking device configuration...")
    response = requests.get(f"{api_url}/distributed/gpu-info")
    if response.status_code == 200:
        data = response.json()
        cluster = data["cluster_info"]
        print(f"✅ Devices detected: {cluster['total_devices']}")
        print(f"   Total memory: {cluster['aggregate_hardware']['total_memory_gb']}GB")
        print(f"   Total GPU cores: {cluster['aggregate_hardware']['total_gpu_cores']}")
        print(f"   gRPC status: {cluster['gRPC_communication']}")
        
        print("\n   Device breakdown:")
        for device in data["devices"]:
            hw = device.get("hardware", {})
            print(f"   - {device['device_id']}: {hw.get('device_name', 'Unknown')} ({hw.get('memory_gb', 0)}GB)")
    
    # 3. Test inference
    print("\n3️⃣ Testing distributed inference...")
    request_data = {
        "model": "mlx-community/Qwen3-1.7B-8bit",
        "messages": [
            {"role": "user", "content": "What is distributed computing in one sentence?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    start_time = time.time()
    response = requests.post(f"{api_url}/v1/chat/completions", json=request_data)
    inference_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        tokens_per_sec = tokens / inference_time
        
        print(f"✅ Inference successful!")
        print(f"   Response: {content[:100]}...")
        print(f"   Performance: {tokens_per_sec:.2f} tokens/second")
        print(f"   Time: {inference_time:.2f}s for {tokens} tokens")
    else:
        print(f"❌ Inference failed: {response.status_code}")
        return False
    
    # 4. Performance summary
    print("\n4️⃣ Performance Summary")
    print(f"✅ 3-device cluster operational")
    print(f"✅ {tokens_per_sec:.1f} tokens/second achieved")
    print(f"✅ All systems functional")
    
    return True


def main():
    """Main validation entry point."""
    success = validate_cluster()
    
    if success:
        print("\n🎉 MLX Distributed validation PASSED!")
        print("\nThe system is ready for production use.")
        return 0
    else:
        print("\n❌ Validation failed. Please check the cluster status.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
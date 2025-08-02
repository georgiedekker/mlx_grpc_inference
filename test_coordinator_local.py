#!/usr/bin/env python3
"""
Test the coordinator locally without workers to verify basic functionality.
"""

import asyncio
import subprocess
import time
import sys
import signal
import requests
from pathlib import Path

def start_coordinator():
    """Start the coordinator API server."""
    print("🚀 Starting coordinator API server...")
    
    # Use the same command as the cluster script
    cmd = [
        sys.executable, "-m", "src.coordinator.api_server", 
        "--host", "0.0.0.0", 
        "--port", "8100"
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Give it time to start
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ Coordinator started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Coordinator failed to start")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start coordinator: {e}")
        return None

def test_endpoints():
    """Test the API endpoints."""
    base_url = "http://localhost:8100"
    
    # Test health endpoint
    print("\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test cluster status
    print("\n📊 Testing cluster status endpoint...")
    try:
        response = requests.get(f"{base_url}/cluster/status", timeout=5)
        if response.status_code == 200:
            print("✅ Cluster status endpoint working")
            data = response.json()
            print(f"   Coordinator: {data.get('coordinator', {}).get('device_id', 'unknown')}")
            print(f"   Workers: {len(data.get('workers', []))}")
        else:
            print(f"❌ Cluster status failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Cluster status failed: {e}")
    
    # Test inference (this will likely fail without workers, but let's see)
    print("\n🤖 Testing inference endpoint...")
    try:
        payload = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [{"role": "user", "content": "Hello, this is a test"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Inference endpoint working")
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                print(f"   Response: {content[:100]}...")
        else:
            print(f"❌ Inference failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def main():
    """Run the local test."""
    print("🧪 MLX Distributed Inference - Local Coordinator Test")
    print("=" * 60)
    
    # Ensure we're in the right directory and environment
    env_path = Path(__file__).parent / ".venv"
    if not env_path.exists():
        print("❌ Virtual environment not found. Please run: uv venv && uv pip install -e .")
        sys.exit(1)
    
    # Start coordinator
    process = start_coordinator()
    if not process:
        print("❌ Failed to start coordinator")
        sys.exit(1)
    
    try:
        # Test endpoints
        test_endpoints()
        
        print("\n" + "=" * 60)
        print("🎉 Local coordinator test completed!")
        print("Note: Full distributed functionality requires workers on other devices.")
        
    finally:
        # Clean up
        print("\n🛑 Stopping coordinator...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✅ Coordinator stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing coordinator...")
            process.kill()
            process.wait()

if __name__ == "__main__":
    main()
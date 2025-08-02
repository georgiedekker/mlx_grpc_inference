#!/usr/bin/env python3
"""Test script for MLX Unified Training Platform."""

import time
import requests
import subprocess
import sys

def test_server():
    """Test the unified training server."""
    print("🧪 Testing MLX Unified Training Platform...")
    
    # Start server in background
    print("🚀 Starting server on port 8600...")
    server_process = subprocess.Popen(
        [sys.executable, "/Users/mini1/Movies/mlx_training_distributed/mlx_unified_training/src/api/server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("\n📡 Testing health endpoint...")
        response = requests.get("http://localhost:8600/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test API docs
        print("\n📚 Testing API documentation...")
        response = requests.get("http://localhost:8600/docs")
        if response.status_code == 200:
            print("✅ API docs available at http://localhost:8600/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
        
        # Test workflow templates
        print("\n📋 Testing workflow templates...")
        response = requests.get(
            "http://localhost:8600/v1/workflows/templates",
            headers={"X-API-Key": "mlx-unified-key"}
        )
        if response.status_code == 200:
            print("✅ Workflow templates endpoint works!")
            templates = response.json()
            if isinstance(templates, dict) and 'templates' in templates:
                template_list = templates['templates']
                print(f"Available templates: {len(template_list)} templates")
                for template in template_list:
                    print(f"  - {template['name']}: {template['description']}")
            else:
                print(f"Templates response: {templates}")
        else:
            print(f"❌ Workflow templates failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing server: {e}")
    finally:
        # Stop server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    test_server()
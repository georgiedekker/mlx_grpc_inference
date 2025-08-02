"""
Test script to verify Alpaca and ShareGPT dataset format support.
"""

import json
import os
import tempfile
from pathlib import Path
import requests
from typing import Dict, Any

# API endpoint
API_BASE_URL = "http://localhost:8200"


def create_test_datasets():
    """Create test dataset files in both formats."""
    test_data = {}
    
    # Create Alpaca format test data
    alpaca_data = [
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "input": ""
        },
        {
            "instruction": "Translate the following sentence to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        },
        {
            "instruction": "Summarize the key points",
            "input": "Machine learning has revolutionized many industries. It enables computers to identify patterns in data and make predictions. Applications include image recognition, natural language processing, and recommendation systems.",
            "output": "Key points: ML identifies patterns in data, makes predictions, and is used in image recognition, NLP, and recommendation systems.",
            "system": "You are a helpful summarization assistant."
        }
    ]
    
    # Create ShareGPT format test data
    sharegpt_data = [
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a helpful AI assistant."
                },
                {
                    "from": "human",
                    "value": "What is machine learning?"
                },
                {
                    "from": "gpt",
                    "value": "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Can you help me understand neural networks?"
                },
                {
                    "from": "gpt",
                    "value": "I'd be happy to help! Neural networks are computing systems inspired by biological neural networks in animal brains."
                },
                {
                    "from": "human",
                    "value": "What are the main components?"
                },
                {
                    "from": "gpt",
                    "value": "The main components are:\n1. Neurons (nodes)\n2. Connections (weights)\n3. Layers (input, hidden, output)\n4. Activation functions\n5. Bias terms"
                }
            ]
        }
    ]
    
    # Save test datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save Alpaca format (JSON)
        alpaca_json_path = Path(tmpdir) / "alpaca_test.json"
        with open(alpaca_json_path, 'w') as f:
            json.dump(alpaca_data, f, indent=2)
        test_data["alpaca_json"] = alpaca_json_path
        
        # Save Alpaca format (JSONL)
        alpaca_jsonl_path = Path(tmpdir) / "alpaca_test.jsonl"
        with open(alpaca_jsonl_path, 'w') as f:
            for sample in alpaca_data:
                f.write(json.dumps(sample) + '\n')
        test_data["alpaca_jsonl"] = alpaca_jsonl_path
        
        # Save ShareGPT format (JSON)
        sharegpt_json_path = Path(tmpdir) / "sharegpt_test.json"
        with open(sharegpt_json_path, 'w') as f:
            json.dump(sharegpt_data, f, indent=2)
        test_data["sharegpt_json"] = sharegpt_json_path
        
        # Save ShareGPT format (JSONL)
        sharegpt_jsonl_path = Path(tmpdir) / "sharegpt_test.jsonl"
        with open(sharegpt_jsonl_path, 'w') as f:
            for sample in sharegpt_data:
                f.write(json.dumps(sample) + '\n')
        test_data["sharegpt_jsonl"] = sharegpt_jsonl_path
    
    return test_data


def test_training_job_creation(file_path: str, format_type: str):
    """Test creating a training job with the specified dataset."""
    print(f"\n Testing {format_type} format with file: {file_path}")
    
    # Test with LoRA enabled
    request_data = {
        "model": "mlx-community/Qwen2.5-1.5B-4bit",
        "training_file": str(file_path),
        "hyperparameters": {
            "n_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "format": format_type
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/fine-tuning/jobs",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            job_data = response.json()
            print(f"✅ Successfully created training job: {job_data.get('id', 'N/A')}")
            print(f"   Status: {job_data.get('status', 'N/A')}")
            return job_data
        else:
            print(f"❌ Failed to create training job: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def test_job_status(job_id: str):
    """Check the status of a training job."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/v1/fine-tuning/jobs/{job_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            status_data = response.json()
            print(f"\n📊 Job Status for {job_id}:")
            print(f"   Status: {status_data.get('status', 'unknown')}")
            print(f"   Progress: {status_data.get('progress', {})}")
            print(f"   LoRA Info: {status_data.get('lora_info', {})}")
            return status_data
        else:
            print(f"❌ Failed to get job status: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Status request failed: {e}")
        return None


def test_api_endpoints():
    """Test various API endpoints with dataset support."""
    print("=" * 60)
    print("Testing Team B Training API - Dataset Format Support")
    print("=" * 60)
    
    # Check API status
    print("\n1. Checking API status...")
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API is running on port {status.get('port', 'unknown')}")
            print(f"   Active training jobs: {status.get('active_training_jobs', 0)}")
        else:
            print("❌ API status check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("\nMake sure Team B's API is running on port 8200")
        return
    
    # Create test datasets
    print("\n2. Creating test datasets...")
    test_datasets = create_test_datasets()
    
    # Test Alpaca format
    print("\n3. Testing Alpaca format...")
    alpaca_job = test_training_job_creation(
        test_datasets["alpaca_json"],
        "alpaca"
    )
    
    # Test ShareGPT format
    print("\n4. Testing ShareGPT format...")
    sharegpt_job = test_training_job_creation(
        test_datasets["sharegpt_json"],
        "sharegpt"
    )
    
    # Check job statuses
    if alpaca_job:
        test_job_status(alpaca_job.get("id", "unknown"))
    
    if sharegpt_job:
        test_job_status(sharegpt_job.get("id", "unknown"))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Alpaca format: " + ("✅ Supported" if alpaca_job else "❌ Not working"))
    print("- ShareGPT format: " + ("✅ Supported" if sharegpt_job else "❌ Not working"))
    print("- LoRA integration: Check job status for LoRA info")
    print("=" * 60)


def create_curl_examples():
    """Generate curl command examples for testing."""
    examples = """
# Curl Examples for Testing Team B's API

# 1. Check API Status
curl -X GET http://localhost:8200/status

# 2. Create Alpaca Format Training Job with LoRA
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "training_file": "alpaca_data.json",
    "hyperparameters": {
      "n_epochs": 3,
      "batch_size": 4,
      "learning_rate": 5e-5,
      "use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16
    }
  }'

# 3. Create ShareGPT Format Training Job with QLoRA
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "training_file": "sharegpt_data.json",
    "hyperparameters": {
      "n_epochs": 3,
      "batch_size": 4,
      "learning_rate": 5e-5,
      "use_lora": true,
      "use_qlora": true,
      "lora_r": 4,
      "format": "sharegpt"
    }
  }'

# 4. Check Training Job Status
curl -X GET http://localhost:8200/v1/fine-tuning/jobs/{job_id}

# 5. List All Training Jobs
curl -X GET http://localhost:8200/v1/fine-tuning/jobs
"""
    
    return examples


if __name__ == "__main__":
    # Run tests
    test_api_endpoints()
    
    # Print curl examples
    print("\n" + "=" * 60)
    print("Curl Command Examples:")
    print("=" * 60)
    print(create_curl_examples())
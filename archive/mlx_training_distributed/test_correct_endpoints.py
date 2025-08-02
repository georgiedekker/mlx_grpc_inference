"""
Test script for Team B's actual API endpoints.
"""

import json
import os
import tempfile
from pathlib import Path
import requests
from typing import Dict, Any
import time

# API endpoint
API_BASE_URL = "http://localhost:8200"


def test_training_with_datasets():
    """Test the actual training endpoints with dataset support."""
    print("=" * 60)
    print("Testing Team B Training API - Actual Endpoints")
    print("=" * 60)
    
    # 1. Check API status
    print("\n1. Checking API status...")
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API is running on port {status.get('port', 'unknown')}")
            print(f"   Service: {status.get('service', 'unknown')}")
            print(f"   Active training jobs: {status.get('active_training_jobs', 0)}")
            print(f"   Active provider: {status.get('llm_providers', {}).get('active_provider', 'unknown')}")
        else:
            print("❌ API status check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # 2. List available providers
    print("\n2. Listing available providers...")
    try:
        response = requests.get(f"{API_BASE_URL}/providers")
        if response.status_code == 200:
            providers = response.json()
            print(f"✅ Available providers: {providers}")
        else:
            print("❌ Failed to list providers")
    except Exception as e:
        print(f"❌ Error listing providers: {e}")
    
    # 3. Start a training job (testing current endpoint structure)
    print("\n3. Starting a training job...")
    
    # Create test Alpaca dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        alpaca_data = [
            {
                "instruction": "What is machine learning?",
                "output": "Machine learning is a branch of AI that enables systems to learn from data.",
                "input": ""
            },
            {
                "instruction": "Explain neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks.",
                "input": ""
            }
        ]
        json.dump(alpaca_data, f)
        dataset_path = f.name
    
    training_config = {
        "experiment_name": f"test_training_{int(time.time())}",
        "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
        "dataset_path": dataset_path,
        "dataset_format": "alpaca",  # Specify format
        "training_config": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            # LoRA parameters
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/train/start",
            json=training_config
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training started successfully!")
            print(f"   Experiment: {result.get('experiment_name', 'unknown')}")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Check training status
            exp_name = result.get('experiment_name')
            if exp_name:
                time.sleep(2)  # Wait a bit
                status_response = requests.get(f"{API_BASE_URL}/train/{exp_name}/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"\n📊 Training Status:")
                    print(f"   Status: {status_data.get('status', 'unknown')}")
                    print(f"   Progress: {status_data.get('progress', {})}")
                    if 'metrics' in status_data:
                        print(f"   Metrics: {status_data.get('metrics', {})}")
        else:
            print(f"❌ Failed to start training: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error starting training: {e}")
    
    finally:
        # Clean up
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)
    
    # 4. List training jobs
    print("\n4. Listing training jobs...")
    try:
        response = requests.get(f"{API_BASE_URL}/train")
        if response.status_code == 200:
            jobs = response.json()
            print(f"✅ Active training jobs: {jobs}")
        else:
            print("❌ Failed to list training jobs")
    except Exception as e:
        print(f"❌ Error listing jobs: {e}")
    
    # 5. Test text generation
    print("\n5. Testing text generation...")
    gen_request = {
        "prompt": "What is machine learning?",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=gen_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful!")
            print(f"   Provider: {result.get('provider', 'unknown')}")
            print(f"   Response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Generation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error generating text: {e}")


def create_integration_guide():
    """Create a guide for integrating the missing features."""
    guide = """
# Integration Guide for Team B

Based on the API analysis, here's how to integrate the missing features:

## 1. LoRA Integration

The training endpoint needs to accept LoRA parameters in the training_config:

```python
@app.post("/train/start")
async def start_training(request: TrainingRequest):
    # Extract LoRA config
    lora_config = {
        "use_lora": request.training_config.get("use_lora", False),
        "lora_r": request.training_config.get("lora_r", 16),
        "lora_alpha": request.training_config.get("lora_alpha", 32.0),
        "lora_dropout": request.training_config.get("lora_dropout", 0.1),
        "lora_target_modules": request.training_config.get("lora_target_modules"),
        "use_qlora": request.training_config.get("use_qlora", False)
    }
    
    # Apply LoRA to model before training
    if lora_config["use_lora"]:
        from archived_components.lora.lora import apply_lora_to_model, LoRAConfig
        model = apply_lora_to_model(model, LoRAConfig(**lora_config))
```

## 2. Dataset Format Support

Add dataset format detection and loading:

```python
def load_training_dataset(dataset_path: str, dataset_format: str = None):
    if dataset_format is None:
        # Auto-detect format
        from team_b_integration.datasets.dataset_integration import detect_dataset_format
        dataset_format = detect_dataset_format(dataset_path)
    
    if dataset_format == "alpaca":
        from archived_components.datasets.alpaca_dataset import AlpacaDataset
        return AlpacaDataset(dataset_path, config)
    elif dataset_format == "sharegpt":
        from archived_components.datasets.sharegpt_dataset import ShareGPTDataset
        return ShareGPTDataset(dataset_path, config)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
```

## 3. Update Training Status

Include LoRA and dataset info in training status:

```python
@app.get("/train/{experiment_name}/status")
async def get_training_status(experiment_name: str):
    status = get_job_status(experiment_name)
    
    # Add LoRA info if enabled
    if status.get("lora_enabled"):
        status["lora_info"] = {
            "rank": status.get("lora_r"),
            "alpha": status.get("lora_alpha"),
            "trainable_params": status.get("trainable_params"),
            "total_params": status.get("total_params")
        }
    
    # Add dataset info
    status["dataset_info"] = {
        "format": status.get("dataset_format", "unknown"),
        "total_samples": status.get("total_samples", 0)
    }
    
    return status
```

## 4. Add Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "lora": True,
            "qlora": True,
            "dataset_formats": ["alpaca", "sharegpt"],
            "providers": ["openai", "anthropic", "together", "ollama", "local_mlx"]
        },
        "active_jobs": len(active_training_jobs)
    }
```

## 5. Authentication (Basic API Key)

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != os.getenv("API_KEY", "default-key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Use as dependency
@app.post("/train/start", dependencies=[Depends(verify_api_key)])
async def start_training(...):
    ...
```
"""
    return guide


if __name__ == "__main__":
    # Run tests
    test_training_with_datasets()
    
    # Print integration guide
    print("\n" + "=" * 60)
    print("Integration Guide:")
    print("=" * 60)
    print(create_integration_guide())
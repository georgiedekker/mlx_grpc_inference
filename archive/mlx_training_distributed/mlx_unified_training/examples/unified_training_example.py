#!/usr/bin/env python3
"""
Example: Complete training pipeline using MLX Unified Training Platform

This example demonstrates how to:
1. Create a unified training pipeline
2. Configure each stage (SFT, Distillation, RLHF)
3. Execute the pipeline
4. Monitor progress
"""

import requests
import json
import time
from typing import Dict, Any

# API configuration
API_URL = "http://localhost:8600"
API_KEY = "mlx-unified-key"

def create_pipeline_example():
    """Create and run a complete training pipeline."""
    
    print("🚀 MLX Unified Training Platform - Example Pipeline")
    print("=" * 60)
    
    # Step 1: Create a pipeline
    print("\n1️⃣ Creating unified training pipeline...")
    
    pipeline_data = {
        "name": "chatbot_aligned_model",
        "stages": ["sft", "distillation", "rlhf"],
        "base_model": "mlx-community/Qwen2.5-1.5B",
        "dataset_path": "/data/chatbot_conversations.json",
        "auto_configure": True
    }
    
    response = requests.post(
        f"{API_URL}/v1/pipelines/create",
        json=pipeline_data,
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to create pipeline: {response.text}")
        return
    
    result = response.json()
    pipeline_id = result["pipeline_id"]
    print(f"✅ Pipeline created: {pipeline_id}")
    print(f"   Stages: {' → '.join(pipeline_data['stages'])}")
    
    # Step 2: Get pipeline details
    print("\n2️⃣ Pipeline configuration:")
    
    response = requests.get(
        f"{API_URL}/v1/pipelines/{pipeline_id}",
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        pipeline = response.json()
        config = pipeline["config"]
        
        if "sft_config" in config and config["sft_config"]:
            print("\n   SFT Configuration:")
            print(f"     - LoRA enabled: {config['sft_config']['use_lora']}")
            print(f"     - LoRA rank: {config['sft_config']['lora_rank']}")
            print(f"     - Epochs: {config['sft_config']['epochs']}")
        
        if "distillation_config" in config and config["distillation_config"]:
            print("\n   Distillation Configuration:")
            print(f"     - Teachers: {', '.join(config['distillation_config']['teacher_models'])}")
            print(f"     - Temperature: {config['distillation_config']['temperature']}")
            print(f"     - Alpha: {config['distillation_config']['alpha']}")
        
        if "rlhf_config" in config and config["rlhf_config"]:
            print("\n   RLHF Configuration:")
            print(f"     - Method: {config['rlhf_config']['method']}")
            print(f"     - Beta: {config['rlhf_config']['beta']}")
            print(f"     - Learning rate: {config['rlhf_config']['learning_rate']}")
    
    # Step 3: Run the pipeline
    print("\n3️⃣ Starting pipeline execution...")
    
    response = requests.post(
        f"{API_URL}/v1/pipelines/{pipeline_id}/run",
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to start pipeline: {response.text}")
        return
    
    print("✅ Pipeline started!")
    
    # Step 4: Monitor progress
    print("\n4️⃣ Monitoring pipeline progress...")
    
    while True:
        response = requests.get(
            f"{API_URL}/v1/pipelines/{pipeline_id}/status",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get status: {response.text}")
            break
        
        status = response.json()
        
        print(f"\n   Status: {status['status']}")
        print(f"   Current stage: {status['current_stage'] or 'None'}")
        print(f"   Completed stages: {', '.join(status['stages_completed']) or 'None'}")
        
        if status['progress']:
            print("\n   Progress:")
            for stage, info in status['progress'].items():
                print(f"     {stage}: {info['progress']}% ({info['status']})")
        
        if status['status'] in ["completed", "failed"]:
            break
        
        time.sleep(5)  # Check every 5 seconds
    
    print("\n" + "=" * 60)
    print(f"Pipeline {status['status'].upper()}!")


def workflow_template_example():
    """Example using workflow templates."""
    
    print("\n\n🎯 Using Workflow Templates")
    print("=" * 60)
    
    # Get available templates
    print("\n1️⃣ Available workflow templates:")
    
    response = requests.get(f"{API_URL}/v1/workflows/templates")
    if response.status_code == 200:
        templates = response.json()["templates"]
        for name, info in templates.items():
            print(f"\n   {name}:")
            print(f"     Description: {info['description']}")
            print(f"     Stages: {' → '.join(info['stages'])}")
            print(f"     Recommended for: {', '.join(info['recommended_for'])}")
    
    # Create from template
    print("\n2️⃣ Creating pipeline from 'aligned_model' template...")
    
    template_data = {
        "template_name": "aligned_model",
        "model_name": "mlx-community/Qwen2.5-1.5B",
        "dataset_path": "/data/safety_dataset.json"
    }
    
    response = requests.post(
        f"{API_URL}/v1/workflows/from-template",
        json=template_data,
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Pipeline created from template: {result['pipeline_id']}")


def direct_training_example():
    """Example of direct training endpoints."""
    
    print("\n\n🔧 Direct Training Examples")
    print("=" * 60)
    
    # Direct SFT training
    print("\n1️⃣ Direct SFT Training:")
    
    sft_request = {
        "model_name": "mlx-community/Qwen2.5-1.5B",
        "dataset_path": "/data/instructions.json",
        "use_lora": True,
        "lora_rank": 32,
        "epochs": 5,
        "learning_rate": 2e-5
    }
    
    response = requests.post(
        f"{API_URL}/v1/train/sft",
        json=sft_request,
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ SFT job started: {result['job_id']}")
    
    # Direct Distillation
    print("\n2️⃣ Direct Knowledge Distillation:")
    
    distill_request = {
        "student_model": "mlx-community/Qwen2.5-1.5B",
        "dataset_path": "/data/distillation_data.json",
        "teacher_models": ["gpt-4", "claude-3", "gemini-pro"],
        "temperature": 5.0,
        "alpha": 0.8,
        "feature_matching": True
    }
    
    response = requests.post(
        f"{API_URL}/v1/train/distill",
        json=distill_request,
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Distillation job started: {result['job_id']}")
        print(f"   Teachers: {', '.join(result['teachers'])}")
    
    # Direct RLHF
    print("\n3️⃣ Direct RLHF Training:")
    
    rlhf_request = {
        "model_name": "mlx-community/Qwen2.5-1.5B-sft",
        "method": "dpo",
        "beta": 0.1,
        "learning_rate": 1e-6,
        "preference_dataset": "/data/human_preferences.json"
    }
    
    response = requests.post(
        f"{API_URL}/v1/train/rlhf",
        json=rlhf_request,
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ RLHF job started: {result['job_id']}")
        print(f"   Method: {result['message']}")


def main():
    """Run all examples."""
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("❌ MLX Unified Training Platform is not running!")
            print(f"   Please start the server: python src/api/server.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to MLX Unified Training Platform!")
        print(f"   Please start the server on port 8600")
        return
    
    # Run examples
    create_pipeline_example()
    workflow_template_example()
    direct_training_example()
    
    print("\n\n✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
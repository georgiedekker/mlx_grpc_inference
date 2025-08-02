#!/usr/bin/env python3
"""
Team B Complete Working Example
End-to-end demonstration of LoRA training with dataset support.
"""

import os
import sys
import json
import time
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Configuration
API_BASE = "http://localhost:8200"
API_KEY = os.getenv("MLX_TRAINING_API_KEY", "test-api-key")

class TeamBExample:
    """Complete example demonstrating all integration features."""
    
    def __init__(self):
        self.api_base = API_BASE
        self.api_key = API_KEY
        self.temp_dir = Path(tempfile.mkdtemp(prefix="team_b_example_"))
        print(f"📁 Using temporary directory: {self.temp_dir}")
    
    def create_sample_datasets(self):
        """Create sample Alpaca and ShareGPT datasets."""
        print("\n📝 Creating sample datasets...")
        
        # Alpaca dataset
        alpaca_data = [
            {
                "instruction": "What is machine learning?",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Explain neural networks in simple terms.",
                "input": "",
                "output": "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) that process information in layers, learning patterns from data to make predictions or decisions."
            },
            {
                "instruction": "What is the difference between supervised and unsupervised learning?",
                "input": "",
                "output": "Supervised learning uses labeled data where the correct answers are provided during training. Unsupervised learning works with unlabeled data, discovering hidden patterns without predefined answers."
            },
            {
                "instruction": "Write a Python function to calculate factorial.",
                "input": "5",
                "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nresult = factorial(5)  # Returns 120"
            },
            {
                "instruction": "What are the main types of machine learning?",
                "input": "",
                "output": "The three main types are: 1) Supervised Learning - learns from labeled examples, 2) Unsupervised Learning - finds patterns in unlabeled data, and 3) Reinforcement Learning - learns through trial and error with rewards."
            }
        ]
        
        alpaca_path = self.temp_dir / "alpaca_demo.json"
        with open(alpaca_path, 'w') as f:
            json.dump(alpaca_data, f, indent=2)
        print(f"✅ Created Alpaca dataset: {alpaca_path}")
        
        # ShareGPT dataset
        sharegpt_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Can you explain what LoRA is in the context of fine-tuning?"},
                    {"from": "assistant", "value": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices. Instead of updating all parameters, LoRA adds small trainable matrices (typically rank 4-32) that learn task-specific adaptations. This reduces memory usage by 90%+ and speeds up training significantly."},
                    {"from": "human", "value": "How much memory does it save?"},
                    {"from": "assistant", "value": "LoRA typically saves 85-95% of GPU memory compared to full fine-tuning. For example, fine-tuning a 7B parameter model might require 28GB+ of memory, but with LoRA (rank 8), you might only need 6-8GB. The exact savings depend on the rank and target modules chosen."}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "What's the benefit of using QLoRA?"},
                    {"from": "assistant", "value": "QLoRA (Quantized LoRA) combines LoRA with 4-bit quantization, providing even greater memory efficiency. Benefits include: 1) Enables fine-tuning large models on consumer GPUs, 2) Reduces memory by 95%+ compared to full fine-tuning, 3) Maintains near-full performance despite quantization, 4) Makes 13B-70B model fine-tuning accessible on single GPUs."}
                ]
            }
        ]
        
        sharegpt_path = self.temp_dir / "sharegpt_demo.json"
        with open(sharegpt_path, 'w') as f:
            json.dump(sharegpt_data, f, indent=2)
        print(f"✅ Created ShareGPT dataset: {sharegpt_path}")
        
        return alpaca_path, sharegpt_path
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        print("\n🏥 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.api_base}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   - Service: {data.get('service')}")
                print(f"   - Status: {data.get('status')}")
                
                features = data.get('features', {})
                if features.get('lora'):
                    print("   - LoRA: ✅ Supported")
                if features.get('qlora'):
                    print("   - QLoRA: ✅ Supported")
                if 'alpaca' in features.get('dataset_formats', []):
                    print("   - Dataset Formats: ✅ Alpaca, ShareGPT")
                
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_dataset_validation(self, dataset_path: Path, expected_format: str):
        """Test dataset validation endpoint."""
        print(f"\n🔍 Validating {expected_format} dataset...")
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/datasets/validate",
                json={"file_path": str(dataset_path), "sample_size": 3}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('valid') and data.get('format') == expected_format:
                    print(f"✅ Dataset validation passed")
                    print(f"   - Format: {data.get('format')}")
                    print(f"   - Samples: {data.get('total_samples')}")
                    print(f"   - Valid: {data.get('valid')}")
                    
                    # Show sample preview
                    if data.get('sample_preview'):
                        print("   - Sample preview available")
                    
                    return True
                else:
                    print(f"❌ Validation failed or incorrect format detected")
                    return False
            else:
                print(f"❌ Validation endpoint returned: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            return False
    
    def create_lora_training_job(self, dataset_path: Path, use_qlora: bool = False):
        """Create a LoRA training job."""
        job_type = "QLoRA" if use_qlora else "LoRA"
        print(f"\n🚀 Creating {job_type} training job...")
        
        config = {
            "experiment_name": f"demo_{job_type.lower()}_{int(time.time())}",
            "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
            "epochs": 1,
            "learning_rate": 5e-5,
            "warmup_steps": 10,
            "save_steps": 50,
            "lora": {
                "use_lora": True,
                "lora_r": 4 if use_qlora else 8,
                "lora_alpha": 8.0 if use_qlora else 16.0,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "use_qlora": use_qlora
            },
            "dataset": {
                "dataset_path": str(dataset_path),
                "dataset_format": "alpaca",
                "batch_size": 2,
                "max_seq_length": 512,
                "shuffle": True
            },
            "output_dir": str(self.temp_dir / "outputs")
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/fine-tuning/jobs",
                json=config,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {job_type} training job created successfully!")
                print(f"   - Job ID: {data.get('id')}")
                print(f"   - Experiment: {data.get('experiment_name')}")
                print(f"   - LoRA enabled: {data.get('lora_enabled')}")
                
                if data.get('lora_details'):
                    details = data['lora_details']
                    print(f"   - LoRA rank: {details.get('lora_config', {}).get('r')}")
                    print(f"   - Memory savings: {details.get('memory_savings_pct', 0):.1f}%")
                    print(f"   - Compression: {details.get('compression_ratio', 0):.1f}x")
                
                print(f"   - Dataset: {data.get('dataset_info', {}).get('total_samples')} samples")
                print(f"   - Estimated time: {data.get('estimated_completion_time')}")
                
                return data.get('id')
            else:
                print(f"❌ Job creation failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"   - Error: {error.get('detail', 'Unknown error')}")
                except:
                    pass
                return None
                
        except Exception as e:
            print(f"❌ Job creation error: {str(e)}")
            return None
    
    def check_job_status(self, job_id: str):
        """Check training job status."""
        print(f"\n📊 Checking job status: {job_id}")
        
        headers = {"X-API-Key": self.api_key}
        
        try:
            response = requests.get(
                f"{self.api_base}/v1/fine-tuning/jobs/{job_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Job status retrieved")
                print(f"   - Status: {data.get('status')}")
                print(f"   - Progress: {data.get('progress', {}).get('percentage', 0):.1f}%")
                
                if data.get('metrics'):
                    metrics = data['metrics']
                    if metrics.get('loss') is not None:
                        print(f"   - Loss: {metrics['loss']:.4f}")
                    if metrics.get('learning_rate') is not None:
                        print(f"   - Learning rate: {metrics['learning_rate']:.2e}")
                
                return data
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Status check error: {str(e)}")
            return None
    
    def demonstrate_all_features(self):
        """Run complete demonstration of all features."""
        print("\n" + "="*60)
        print("🎯 Team B Complete Integration Demo")
        print("="*60)
        
        # Create sample datasets
        alpaca_path, sharegpt_path = self.create_sample_datasets()
        
        # Test health endpoint
        health_ok = self.test_health_endpoint()
        
        # Test dataset validation
        alpaca_valid = self.test_dataset_validation(alpaca_path, "alpaca")
        sharegpt_valid = self.test_dataset_validation(sharegpt_path, "sharegpt")
        
        # Create LoRA training job
        lora_job_id = self.create_lora_training_job(alpaca_path, use_qlora=False)
        
        # Create QLoRA training job
        qlora_job_id = self.create_lora_training_job(alpaca_path, use_qlora=True)
        
        # Check job status
        if lora_job_id:
            time.sleep(2)  # Wait a bit for job to start
            self.check_job_status(lora_job_id)
        
        # Print curl examples
        self.print_curl_examples(alpaca_path, lora_job_id)
        
        # Summary
        print("\n" + "="*60)
        print("📋 Integration Summary")
        print("="*60)
        
        results = {
            "Health Check": "✅" if health_ok else "❌",
            "Alpaca Validation": "✅" if alpaca_valid else "❌",
            "ShareGPT Validation": "✅" if sharegpt_valid else "❌",
            "LoRA Job Creation": "✅" if lora_job_id else "❌",
            "QLoRA Job Creation": "✅" if qlora_job_id else "❌"
        }
        
        for feature, status in results.items():
            print(f"{feature:<25} {status}")
        
        # Calculate success rate
        success_count = sum(1 for v in results.values() if v == "✅")
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        print(f"\nSuccess Rate: {success_count}/{total_count} ({success_rate:.0f}%)")
        
        if success_rate == 100:
            print("\n🎉 Congratulations! All features are working perfectly!")
            print("Team B has successfully integrated LoRA/QLoRA and dataset support!")
        elif success_rate >= 60:
            print("\n⚠️  Most features are working. Check the failed components.")
        else:
            print("\n❌ Several features need implementation. Review the integration guide.")
    
    def print_curl_examples(self, dataset_path: Path, job_id: str = None):
        """Print curl command examples."""
        print("\n" + "="*60)
        print("🔧 Curl Command Examples")
        print("="*60)
        
        print("\n# 1. Health Check")
        print(f"curl -X GET {self.api_base}/health")
        
        print("\n# 2. Validate Dataset")
        print(f"""curl -X POST {self.api_base}/v1/datasets/validate \\
  -H "Content-Type: application/json" \\
  -d '{{"file_path": "{dataset_path}", "sample_size": 3}}'""")
        
        print("\n# 3. Create LoRA Training Job")
        print(f"""curl -X POST {self.api_base}/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: {self.api_key}" \\
  -d '{{
    "experiment_name": "my_lora_experiment",
    "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
    "epochs": 3,
    "lora": {{
      "use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16.0
    }},
    "dataset": {{
      "dataset_path": "{dataset_path}",
      "batch_size": 8
    }}
  }}'""")
        
        if job_id:
            print("\n# 4. Check Job Status")
            print(f"""curl -X GET {self.api_base}/v1/fine-tuning/jobs/{job_id} \\
  -H "X-API-Key: {self.api_key}\"""")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {self.temp_dir}")
        except:
            pass


def main():
    """Main function."""
    print("🚀 Team B Integration Example")
    print("This script demonstrates all integrated features")
    print("-" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=2)
        print("✅ API server detected at http://localhost:8200")
    except:
        print("❌ API server not detected at http://localhost:8200")
        print("Please start Team B's API server first:")
        print("  cd mlx_distributed_training")
        print("  python app.py  # or uvicorn app:app --port 8200")
        sys.exit(1)
    
    # Run example
    example = TeamBExample()
    
    try:
        example.demonstrate_all_features()
    finally:
        example.cleanup()
    
    print("\n✅ Example completed!")
    print("\nNext steps:")
    print("1. Review any failed components")
    print("2. Implement missing endpoints using team_b_api_modifications.py")
    print("3. Run team_b_validation_script.py for detailed validation")
    print("4. Start training real models with LoRA!")

if __name__ == "__main__":
    main()
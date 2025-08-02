#!/usr/bin/env python3
"""
Team B Complete Working Example
A full end-to-end example that Team B can run immediately to test their integration.
"""

import json
import yaml
import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

# ============================================================================
# SAMPLE DATA CREATION
# ============================================================================

def create_sample_alpaca_dataset(output_path: str = "team_b_alpaca_sample.json") -> str:
    """Create a sample Alpaca dataset for testing."""
    
    sample_data = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves."
        },
        {
            "instruction": "Explain the concept of neural networks.",
            "input": "",
            "output": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. They consist of interconnected nodes (neurons) that process and transmit information. Each connection has a weight that adjusts as learning proceeds, allowing the network to recognize patterns and make predictions."
        },
        {
            "instruction": "What is the difference between supervised and unsupervised learning?",
            "input": "",
            "output": "Supervised learning uses labeled training data to learn a mapping function from inputs to outputs, like classification and regression tasks. Unsupervised learning finds hidden patterns in data without labeled examples, such as clustering and dimensionality reduction."
        },
        {
            "instruction": "Define deep learning.",
            "input": "",
            "output": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep networks) to progressively extract higher-level features from raw input. It has been particularly successful in tasks like image recognition, natural language processing, and speech recognition."
        },
        {
            "instruction": "What is overfitting in machine learning?",
            "input": "",
            "output": "Overfitting occurs when a machine learning model learns the training data too well, including noise and specific details that don't generalize to new data. This results in high accuracy on training data but poor performance on validation or test data."
        },
        {
            "instruction": "Explain gradient descent optimization.",
            "input": "",
            "output": "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning models. It iteratively adjusts model parameters in the direction of steepest decrease of the cost function, using the gradient (derivative) to find the optimal values."
        },
        {
            "instruction": "What is the purpose of regularization?",
            "input": "",
            "output": "Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. It encourages simpler models by constraining or shrinking model parameters, helping the model generalize better to unseen data."
        },
        {
            "instruction": "Define cross-validation.",
            "input": "",
            "output": "Cross-validation is a technique for assessing how well a machine learning model will generalize to new data. It involves dividing the dataset into multiple folds, training on some folds while validating on others, and averaging the results to get a more robust estimate of model performance."
        },
        {
            "instruction": "What is feature engineering?",
            "input": "",
            "output": "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve machine learning model performance. It involves domain knowledge to transform data into a format that better represents the underlying problem for the algorithm."
        },
        {
            "instruction": "Explain the bias-variance tradeoff.",
            "input": "",
            "output": "The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between a model's ability to minimize bias (error from oversimplifying) and variance (error from being too sensitive to small fluctuations). The goal is to find the optimal balance to minimize total error."
        }
    ]
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created Alpaca sample dataset: {output_path} ({len(sample_data)} samples)")
    return output_path


def create_sample_sharegpt_dataset(output_path: str = "team_b_sharegpt_sample.json") -> str:
    """Create a sample ShareGPT dataset for testing."""
    
    sample_data = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Can you explain what LoRA is in machine learning?"
                },
                {
                    "from": "gpt", 
                    "value": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large language models. Instead of updating all model parameters, LoRA adds trainable low-rank matrices to existing weights, dramatically reducing the number of trainable parameters while maintaining performance."
                },
                {
                    "from": "human",
                    "value": "What are the main advantages of using LoRA?"
                },
                {
                    "from": "gpt",
                    "value": "The main advantages of LoRA include: 1) Significantly reduced memory usage during fine-tuning, 2) Faster training times, 3) Smaller model checkpoints (only LoRA weights need to be saved), 4) Easy switching between different LoRA adapters for different tasks, and 5) Maintained performance comparable to full fine-tuning."  
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a helpful AI assistant specializing in explaining technical concepts."
                },
                {
                    "from": "human", 
                    "value": "What is quantization in neural networks?"
                },
                {
                    "from": "gpt",
                    "value": "Quantization in neural networks is a technique that reduces the precision of model weights and activations from full precision (like 32-bit floats) to lower precision formats (like 8-bit integers or 4-bit). This reduces model size and memory usage while maintaining most of the model's performance."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "How does distributed training work?"
                },
                {
                    "from": "gpt",
                    "value": "Distributed training involves splitting the training process across multiple devices or machines. Common approaches include data parallelism (splitting data across devices) and model parallelism (splitting the model itself). This allows training larger models faster by leveraging multiple GPUs or compute nodes."
                },
                {
                    "from": "human", 
                    "value": "What about gradient synchronization?"
                },
                {
                    "from": "gpt",
                    "value": "Gradient synchronization ensures all devices have consistent model updates. In synchronous training, all devices wait for each other to complete their forward/backward passes before updating. In asynchronous training, updates happen independently, which can be faster but may lead to staleness issues."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "What is MLX?"
                },
                {
                    "from": "gpt", 
                    "value": "MLX is Apple's machine learning framework optimized for Apple Silicon (M1, M2, M3 chips). It provides efficient implementations of neural network operations that leverage the unified memory architecture and specialized hardware features of Apple's chips for both training and inference."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Explain attention mechanisms in transformers."
                },
                {
                    "from": "gpt",
                    "value": "Attention mechanisms allow models to focus on relevant parts of the input when making predictions. In transformers, self-attention computes relationships between all positions in a sequence simultaneously. Each position can attend to all positions in the previous layer, creating rich contextual representations."
                },
                {
                    "from": "human",
                    "value": "What's the difference between self-attention and cross-attention?"
                },
                {
                    "from": "gpt",
                    "value": "Self-attention computes attention within the same sequence (queries, keys, and values come from the same input). Cross-attention computes attention between different sequences (queries from one sequence, keys and values from another), commonly used in encoder-decoder architectures for tasks like translation."
                }
            ]
        }
    ]
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created ShareGPT sample dataset: {output_path} ({len(sample_data)} samples)")
    return output_path


def create_sample_training_config(output_path: str = "team_b_training_config.yaml") -> str:
    """Create a sample training configuration."""
    
    config = {
        "model": "mlx-community/Qwen2.5-1.5B-4bit",
        "experiment_name": "team_b_lora_test_" + str(int(time.time())),
        "training_file": "team_b_alpaca_sample.json",
        
        "dataset_config": {
            "dataset_format": "alpaca",
            "batch_size": 4,
            "max_seq_length": 1024,
            "shuffle": True,
            "validation_split": 0.0
        },
        
        "lora_config": {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_modules_to_save": [],
            "lora_bias": "none",
            "use_qlora": True,
            "qlora_compute_dtype": "float16"
        },
        
        "hyperparameters": {
            "n_epochs": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 10,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "save_steps": 50,
            "logging_steps": 5,
            "seed": 42
        },
        
        "output_dir": "./team_b_outputs",
        "save_total_limit": 3
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"✅ Created training config: {output_path}")
    return output_path


# ============================================================================
# API TESTING FUNCTIONS
# ============================================================================

class TeamBAPITester:
    """Complete API testing for Team B integration."""
    
    def __init__(self, api_base_url: str = "http://localhost:8200"):
        self.api_base_url = api_base_url
        self.test_files = []
        
    def setup_test_data(self):
        """Create all test data files."""
        print("🔧 Setting up test data...")
        
        # Create sample datasets
        alpaca_file = create_sample_alpaca_dataset()
        sharegpt_file = create_sample_sharegpt_dataset()
        config_file = create_sample_training_config()
        
        self.test_files = [alpaca_file, sharegpt_file, config_file]
        
        print("✅ Test data setup complete")
        return {
            "alpaca_dataset": alpaca_file,
            "sharegpt_dataset": sharegpt_file,
            "training_config": config_file
        }
    
    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        print("🏥 Testing API health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                features = health_data.get("features", {})
                
                print(f"✅ API is healthy")
                print(f"   LoRA support: {features.get('lora', False)}")
                print(f"   Dataset formats: {features.get('dataset_formats', [])}")
                print(f"   Service version: {health_data.get('version', 'unknown')}")
                
                return {"success": True, "data": health_data}
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_dataset_validation(self, dataset_path: str) -> Dict[str, Any]:
        """Test dataset validation endpoint."""
        print(f"🔍 Testing dataset validation: {dataset_path}")
        
        try:
            response = requests.post(
                f"{self.api_base_url}/v1/datasets/validate",
                json={"file_path": dataset_path, "max_samples": 50},
                timeout=30
            )
            
            if response.status_code == 200:
                validation_data = response.json()
                
                print(f"✅ Dataset validation successful")
                print(f"   Valid: {validation_data.get('valid', False)}")
                print(f"   Format: {validation_data.get('format', 'unknown')}")
                print(f"   Samples: {validation_data.get('total_samples', 0)}")
                print(f"   Errors: {len(validation_data.get('errors', []))}")
                print(f"   Warnings: {len(validation_data.get('warnings', []))}")
                
                return {"success": True, "data": validation_data}
            else:
                print(f"❌ Dataset validation failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_training_job_creation(self, config_path: str) -> Dict[str, Any]:
        """Test training job creation."""
        print(f"🚀 Testing training job creation with config: {config_path}")
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create training request
            training_request = {
                "model": config["model"],
                "experiment_name": config["experiment_name"],
                "training_file": config["training_file"],
                "dataset_config": config["dataset_config"],
                "lora_config": config["lora_config"],
                **config["hyperparameters"]
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/fine-tuning/jobs",
                json=training_request,
                timeout=60
            )
            
            if response.status_code == 200:
                job_data = response.json()
                
                print(f"✅ Training job created successfully")
                print(f"   Job ID: {job_data.get('job_id', 'unknown')}")
                print(f"   Experiment: {job_data.get('experiment_name', 'unknown')}")
                print(f"   Status: {job_data.get('status', 'unknown')}")
                print(f"   LoRA enabled: {job_data.get('lora_info', {}).get('lora_enabled', False)}")
                
                return {"success": True, "data": job_data}
            else:
                print(f"❌ Training job creation failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data}")
                except:
                    print(f"   Response: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Training job creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_job_status(self, job_id: str) -> Dict[str, Any]:
        """Test training job status retrieval."""
        print(f"📊 Testing job status: {job_id}")
        
        try:
            response = requests.get(
                f"{self.api_base_url}/v1/fine-tuning/jobs/{job_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                status_data = response.json()
                
                print(f"✅ Job status retrieved successfully")
                print(f"   Status: {status_data.get('status', 'unknown')}")
                print(f"   Progress: {status_data.get('progress', {})}")
                print(f"   Metrics: {status_data.get('metrics', {})}")
                
                return {"success": True, "data": status_data}
            else:
                print(f"❌ Job status retrieval failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Job status retrieval failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_model_list(self) -> Dict[str, Any]:
        """Test model list endpoint."""
        print("📋 Testing model list...")
        
        try:
            response = requests.get(f"{self.api_base_url}/v1/models", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                
                print(f"✅ Model list retrieved successfully")
                print(f"   Available models: {len(models)}")
                
                for model in models[:3]:  # Show first 3 models
                    print(f"   - {model.get('id', 'unknown')}")
                    if 'recommended_lora_config' in model:
                        lora_config = model['recommended_lora_config']
                        print(f"     LoRA: r={lora_config.get('lora_r', 'unknown')}, alpha={lora_config.get('lora_alpha', 'unknown')}")
                
                return {"success": True, "data": models_data}
            else:
                print(f"❌ Model list failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Model list failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_complete_test(self) -> Dict[str, Any]:
        """Run complete end-to-end test."""
        print("=" * 80)
        print("🧪 TEAM B COMPLETE INTEGRATION TEST")
        print("=" * 80)
        
        results = {
            "timestamp": time.time(),
            "api_url": self.api_base_url,
            "tests": {},
            "overall_success": False
        }
        
        try:
            # Setup test data
            test_data = self.setup_test_data()
            
            # Run tests
            tests = [
                ("API Health", lambda: self.test_api_health()),
                ("Model List", lambda: self.test_model_list()),
                ("Alpaca Dataset Validation", lambda: self.test_dataset_validation(test_data["alpaca_dataset"])),
                ("ShareGPT Dataset Validation", lambda: self.test_dataset_validation(test_data["sharegpt_dataset"])),
                ("Training Job Creation", lambda: self.test_training_job_creation(test_data["training_config"])),
            ]
            
            for test_name, test_func in tests:
                print(f"\n{'=' * 60}")
                result = test_func()
                results["tests"][test_name] = result
                
                if not result["success"]:
                    print(f"❌ {test_name} failed - stopping test suite")
                    break
                
                time.sleep(1)  # Small delay between tests
            
            # Test job status if training job was created
            if results["tests"].get("Training Job Creation", {}).get("success"):
                job_data = results["tests"]["Training Job Creation"]["data"]
                job_id = job_data.get("job_id")
                
                if job_id:
                    print(f"\n{'=' * 60}")
                    time.sleep(2)  # Wait a bit for job to start
                    status_result = self.test_job_status(job_id)
                    results["tests"]["Job Status"] = status_result
            
            # Calculate overall success
            successful_tests = sum(1 for test in results["tests"].values() if test["success"])
            total_tests = len(results["tests"])
            results["overall_success"] = successful_tests == total_tests
            results["success_rate"] = successful_tests / total_tests * 100
            
            # Print summary
            print(f"\n{'=' * 80}")
            print("📊 TEST SUMMARY")
            print("=" * 80)
            print(f"Tests passed: {successful_tests}/{total_tests} ({results['success_rate']:.1f}%)")
            
            if results["overall_success"]:
                print("🎉 ALL TESTS PASSED! Your integration is working correctly.")
            else:
                print("⚠️  Some tests failed. Check the error messages above.")
                print("\n🔧 Troubleshooting tips:")
                print("- Ensure your API server is running on port 8200")
                print("- Check that you've applied all the API modifications")
                print("- Run the validation script: python team_b_validation_script.py")
                print("- Check server logs for detailed error messages")
            
            return results
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            results["error"] = "Interrupted by user"
            return results
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            results["error"] = str(e)
            return results
        finally:
            # Cleanup test files
            self.cleanup_test_files()
    
    def cleanup_test_files(self):
        """Clean up created test files."""
        print(f"\n🧹 Cleaning up test files...")
        
        for file_path in self.test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"   Removed: {file_path}")
            except Exception as e:
                print(f"   Failed to remove {file_path}: {e}")


# ============================================================================
# CURL COMMAND EXAMPLES
# ============================================================================

def generate_curl_examples() -> str:
    """Generate curl command examples for testing."""
    
    curl_examples = """
# ============================================================================
# CURL COMMAND EXAMPLES FOR TESTING TEAM B API
# ============================================================================

# 1. Health Check
curl -X GET http://localhost:8200/health

# 2. List Available Models
curl -X GET http://localhost:8200/v1/models

# 3. Validate Alpaca Dataset
curl -X POST http://localhost:8200/v1/datasets/validate \\
  -H "Content-Type: application/json" \\
  -d '{
    "file_path": "team_b_alpaca_sample.json",
    "max_samples": 10
  }'

# 4. Get Supported Dataset Formats
curl -X GET http://localhost:8200/v1/datasets/formats

# 5. Create LoRA Training Job (Alpaca)
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "experiment_name": "curl_test_alpaca_lora",
    "training_file": "team_b_alpaca_sample.json",
    "dataset_config": {
      "dataset_format": "alpaca",
      "batch_size": 2,
      "max_seq_length": 512
    },
    "lora_config": {
      "use_lora": true,
      "lora_r": 4,
      "lora_alpha": 8.0,
      "lora_dropout": 0.1,
      "use_qlora": false
    },
    "n_epochs": 1,
    "learning_rate": 1e-4,
    "warmup_steps": 5,
    "save_steps": 20,
    "logging_steps": 5
  }'

# 6. Create LoRA Training Job (ShareGPT)
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "experiment_name": "curl_test_sharegpt_lora",
    "training_file": "team_b_sharegpt_sample.json",
    "dataset_config": {
      "dataset_format": "sharegpt",
      "batch_size": 2,
      "max_seq_length": 512
    },
    "lora_config": {
      "use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16.0,
      "use_qlora": true
    },
    "n_epochs": 1,
    "learning_rate": 5e-5
  }'

# 7. Check Training Job Status (replace JOB_ID with actual job ID)
curl -X GET http://localhost:8200/v1/fine-tuning/jobs/JOB_ID

# 8. List All Training Jobs (if endpoint exists)
curl -X GET http://localhost:8200/v1/fine-tuning/jobs

# ============================================================================
# TESTING SEQUENCE
# ============================================================================

# Step 1: Check if API is running
curl -X GET http://localhost:8200/health

# Step 2: Create test datasets
python team_b_complete_example.py --create-data-only

# Step 3: Validate datasets
curl -X POST http://localhost:8200/v1/datasets/validate \\
  -H "Content-Type: application/json" \\
  -d '{"file_path": "team_b_alpaca_sample.json"}'

# Step 4: Start a simple training job
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "experiment_name": "quick_test",
    "training_file": "team_b_alpaca_sample.json",
    "lora_config": {"use_lora": true, "lora_r": 4},
    "n_epochs": 1,
    "dataset_config": {"batch_size": 1}
  }'

# ============================================================================
"""
    
    return curl_examples


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface for the complete example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Team B Complete Integration Example")
    parser.add_argument("--api-url", default="http://localhost:8200", help="API base URL")
    parser.add_argument("--create-data-only", action="store_true", help="Only create test data files")
    parser.add_argument("--test-api-only", action="store_true", help="Only test API endpoints")
    parser.add_argument("--generate-curl", action="store_true", help="Generate curl examples")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files")
    
    args = parser.parse_args()
    
    if args.generate_curl:
        print(generate_curl_examples())
        return 0
    
    if args.cleanup:
        test_files = [
            "team_b_alpaca_sample.json",
            "team_b_sharegpt_sample.json", 
            "team_b_training_config.yaml"
        ]
        
        print("🧹 Cleaning up test files...")
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   Removed: {file_path}")
        
        print("✅ Cleanup complete")
        return 0
    
    if args.create_data_only:
        print("📝 Creating test data files only...")
        create_sample_alpaca_dataset()
        create_sample_sharegpt_dataset()
        create_sample_training_config()
        print("✅ Test data created successfully")
        return 0
    
    # Run complete test
    tester = TeamBAPITester(api_base_url=args.api_url)
    
    if args.test_api_only:
        # Create minimal test data for API testing
        tester.setup_test_data()
        
        # Test individual endpoints
        tester.test_api_health()
        tester.test_model_list()
        tester.test_dataset_validation("team_b_alpaca_sample.json")
        
        return 0
    
    # Run complete end-to-end test
    results = tester.run_complete_test()
    
    # Exit with appropriate code
    return 0 if results["overall_success"] else 1


if __name__ == "__main__":
    exit(main())
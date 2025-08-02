"""
End-to-end training test for Team B's API with LoRA and dataset support.
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8200"
API_KEY = os.getenv("MLX_TRAINING_API_KEY", "test-key")


class TrainingTestSuite:
    """Comprehensive test suite for training features."""
    
    def __init__(self):
        self.test_results = []
        self.example_dir = Path(__file__).parent / "examples"
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"     {details}")
    
    def test_api_health(self) -> bool:
        """Test API health endpoint."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "API Health Check",
                    True,
                    f"Service: {data.get('service', 'unknown')}, Features: {data.get('features', {})}"
                )
                return True
            else:
                self.log_result("API Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("API Health Check", False, str(e))
            return False
    
    def test_dataset_validation(self, format_type: str) -> bool:
        """Test dataset validation endpoint."""
        dataset_path = self.example_dir / f"{format_type}_example.json"
        
        if not dataset_path.exists():
            self.log_result(f"Dataset Validation ({format_type})", False, "Example file not found")
            return False
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/datasets/validate",
                json={"file_path": str(dataset_path)},
                headers={"X-API-Key": API_KEY},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    f"Dataset Validation ({format_type})",
                    data.get("valid", False),
                    f"Format: {data.get('format')}, Samples: {data.get('total_samples')}"
                )
                return data.get("valid", False)
            else:
                self.log_result(
                    f"Dataset Validation ({format_type})",
                    False,
                    f"Status: {response.status_code}"
                )
                return False
                
        except Exception as e:
            self.log_result(f"Dataset Validation ({format_type})", False, str(e))
            return False
    
    def test_training_job(
        self, 
        format_type: str, 
        use_lora: bool = True,
        use_qlora: bool = False
    ) -> Optional[str]:
        """Test creating and monitoring a training job."""
        dataset_path = self.example_dir / f"{format_type}_example.json"
        
        config = {
            "experiment_name": f"test_{format_type}_{'qlora' if use_qlora else 'lora' if use_lora else 'full'}_{int(time.time())}",
            "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
            "epochs": 1,
            "learning_rate": 5e-5,
            "lora": {
                "use_lora": use_lora,
                "lora_r": 4 if use_qlora else 8,
                "lora_alpha": 8 if use_qlora else 16,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "use_qlora": use_qlora
            },
            "dataset": {
                "dataset_path": str(dataset_path),
                "dataset_format": format_type,
                "batch_size": 2,
                "max_seq_length": 512  # Smaller for testing
            },
            "output_dir": f"./test_outputs/{format_type}_{int(time.time())}"
        }
        
        test_name = f"Training Job ({format_type}, {'QLoRA' if use_qlora else 'LoRA' if use_lora else 'Full'})"
        
        try:
            # Start training
            response = requests.post(
                f"{API_BASE_URL}/train/start",
                json=config,
                headers={"X-API-Key": API_KEY},
                timeout=30
            )
            
            if response.status_code != 200:
                self.log_result(test_name, False, f"Failed to start: {response.status_code}")
                return None
            
            job_data = response.json()
            job_id = job_data.get("job_id") or job_data.get("experiment_name")
            
            self.log_result(
                test_name + " - Start",
                True,
                f"Job ID: {job_id}, LoRA: {job_data.get('lora_enabled', False)}"
            )
            
            # Monitor training progress
            max_checks = 10
            for i in range(max_checks):
                time.sleep(3)
                
                status_response = requests.get(
                    f"{API_BASE_URL}/train/{job_id}/status",
                    headers={"X-API-Key": API_KEY},
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    self.log_result(
                        test_name + f" - Progress Check {i+1}",
                        True,
                        f"Status: {status.get('status')}, Progress: {status.get('progress', {})}"
                    )
                    
                    # Check for LoRA info
                    if "lora_details" in status:
                        self.log_result(
                            test_name + " - LoRA Info",
                            True,
                            f"Rank: {status['lora_details'].get('rank')}, Alpha: {status['lora_details'].get('alpha')}"
                        )
                    
                    # Check if completed or failed
                    if status.get("status") in ["completed", "failed", "error"]:
                        final_success = status.get("status") == "completed"
                        self.log_result(
                            test_name + " - Completion",
                            final_success,
                            f"Final status: {status.get('status')}"
                        )
                        return job_id if final_success else None
                
            self.log_result(test_name + " - Timeout", False, "Training didn't complete in time")
            return None
            
        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")
            return None
    
    def test_model_inference(self, job_id: Optional[str] = None):
        """Test inference with trained model."""
        test_name = "Model Inference" + (f" (Job: {job_id})" if job_id else "")
        
        try:
            inference_request = {
                "prompt": "What is machine learning?",
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            if job_id:
                inference_request["model"] = f"fine-tuned/{job_id}"
            
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json=inference_request,
                headers={"X-API-Key": API_KEY},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log_result(
                    test_name,
                    True,
                    f"Generated {len(result.get('text', '').split())} words"
                )
                return True
            else:
                self.log_result(test_name, False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result(test_name, False, str(e))
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print("=" * 60)
        print("Team B End-to-End Training Test Suite")
        print("=" * 60)
        
        # Check if API supports new endpoints
        if not self.test_api_health():
            print("\n⚠️  Note: Health endpoint not found. Team B needs to implement it.")
        
        # Test dataset validation
        print("\n--- Dataset Validation Tests ---")
        alpaca_valid = self.test_dataset_validation("alpaca")
        sharegpt_valid = self.test_dataset_validation("sharegpt")
        
        if not alpaca_valid and not sharegpt_valid:
            print("\n⚠️  Note: Dataset validation endpoint not found. Team B needs to implement it.")
        
        # Test training jobs
        print("\n--- Training Job Tests ---")
        
        # Test Alpaca with LoRA
        alpaca_job = self.test_training_job("alpaca", use_lora=True, use_qlora=False)
        
        # Test ShareGPT with QLoRA
        sharegpt_job = self.test_training_job("sharegpt", use_lora=True, use_qlora=True)
        
        # Test inference
        print("\n--- Inference Tests ---")
        self.test_model_inference()
        
        if alpaca_job:
            self.test_model_inference(alpaca_job)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate test summary report."""
        print("\n" + "=" * 60)
        print("Test Summary Report")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Feature checklist
        print("\n--- Feature Implementation Status ---")
        features = {
            "Health Endpoint": any(r["test"] == "API Health Check" and r["passed"] for r in self.test_results),
            "Dataset Validation": any("Dataset Validation" in r["test"] and r["passed"] for r in self.test_results),
            "LoRA Training": any("LoRA" in r["test"] and "Start" in r["test"] and r["passed"] for r in self.test_results),
            "QLoRA Training": any("QLoRA" in r["test"] and "Start" in r["test"] and r["passed"] for r in self.test_results),
            "Alpaca Format": any("alpaca" in r["test"] and r["passed"] for r in self.test_results),
            "ShareGPT Format": any("sharegpt" in r["test"] and r["passed"] for r in self.test_results),
        }
        
        for feature, implemented in features.items():
            status = "✅" if implemented else "❌"
            print(f"{status} {feature}")
        
        # Implementation notes
        print("\n--- Implementation Notes for Team B ---")
        if not features["Health Endpoint"]:
            print("1. Add /health endpoint following the implementation guide")
        if not features["Dataset Validation"]:
            print("2. Add /datasets/validate endpoint for format detection")
        if not features["LoRA Training"]:
            print("3. Integrate LoRA support from archived_components/lora/lora.py")
        if not features["Alpaca Format"] or not features["ShareGPT Format"]:
            print("4. Integrate dataset loaders from archived_components/datasets/")
        
        print("\nRefer to team_b_integration/IMPLEMENTATION_GUIDE.md for detailed instructions.")
        print("=" * 60)


def create_sample_config_file():
    """Create a sample configuration file for testing."""
    config = {
        "experiment_name": "sample_lora_training",
        "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
        "dataset_path": "alpaca_example.json",
        "dataset_format": "alpaca",
        "training_params": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "save_steps": 500
        },
        "lora_params": {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "use_qlora": False
        },
        "output_dir": "./lora_model_output"
    }
    
    with open("sample_training_config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created sample_training_config.yaml")


if __name__ == "__main__":
    # Run comprehensive tests
    test_suite = TrainingTestSuite()
    test_suite.run_all_tests()
    
    # Create sample config
    print("\n" + "=" * 60)
    create_sample_config_file()
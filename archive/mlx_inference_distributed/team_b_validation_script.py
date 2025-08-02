#!/usr/bin/env python3
"""
Team B Integration Validation Script
Comprehensive testing script to verify LoRA and dataset integration works correctly.
"""

import sys
import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """Test result information."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: float = 0.0

class TeamBValidator:
    """Comprehensive validation for Team B integration."""
    
    def __init__(self, api_base_url: str = "http://localhost:8200"):
        self.api_base_url = api_base_url
        self.results: List[TestResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and record results."""
        self.log(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, tuple):
                passed, message, details = result
            else:
                passed, message, details = result, "Test completed", None
            
            test_result = TestResult(
                name=test_name,
                passed=passed,
                message=message,
                details=details,
                duration=duration
            )
            
            self.results.append(test_result)
            status = "✅ PASS" if passed else "❌ FAIL"
            self.log(f"{status} {test_name}: {message} ({duration:.2f}s)")
            
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                name=test_name,
                passed=False,
                message=f"Exception: {str(e)}",
                duration=duration
            )
            
            self.results.append(test_result)
            self.log(f"❌ FAIL {test_name}: {str(e)} ({duration:.2f}s)", "ERROR")
            
            return test_result
    
    def test_file_structure(self) -> tuple:
        """Test that all required files are in place."""
        team_b_dir = Path("/Users/mini1/Movies/mlx_distributed_training")
        
        required_files = [
            "src/mlx_distributed_training/training/lora/lora.py",
            "src/mlx_distributed_training/integration/lora_integration.py",
            "src/mlx_distributed_training/datasets/alpaca_dataset.py",
            "src/mlx_distributed_training/datasets/sharegpt_dataset.py",
            "src/mlx_distributed_training/integration/dataset_integration.py",
            "examples/alpaca_example.json",
            "examples/sharegpt_example.json",
            "test_integration.py"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = team_b_dir / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        passed = len(missing_files) == 0
        message = f"{len(existing_files)}/{len(required_files)} files found"
        
        details = {
            "existing_files": existing_files,
            "missing_files": missing_files,
            "team_b_directory": str(team_b_dir),
            "directory_exists": team_b_dir.exists()
        }
        
        return passed, message, details
    
    def test_python_imports(self) -> tuple:
        """Test that Python imports work correctly."""
        team_b_dir = Path("/Users/mini1/Movies/mlx_distributed_training")
        
        # Test import script
        test_script = f'''
import sys
sys.path.insert(0, "{team_b_dir}/src")

try:
    # Test LoRA imports
    from mlx_distributed_training.training.lora.lora import (
        LoRAConfig, LoRALayer, LoRALinear, apply_lora_to_model
    )
    print("✅ LoRA imports successful")
    
    # Test dataset imports
    from mlx_distributed_training.datasets.alpaca_dataset import AlpacaDataset
    from mlx_distributed_training.datasets.sharegpt_dataset import ShareGPTDataset
    print("✅ Dataset imports successful")
    
    # Test integration imports
    from mlx_distributed_training.integration.lora_integration import (
        LoRATrainingConfig, create_lora_enabled_trainer
    )
    from mlx_distributed_training.integration.dataset_integration import (
        validate_dataset, detect_dataset_format
    )
    print("✅ Integration imports successful")
    
    print("ALL_IMPORTS_SUCCESSFUL")
    
except ImportError as e:
    print(f"❌ Import failed: {{e}}")
    sys.exit(1)
        '''
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = "ALL_IMPORTS_SUCCESSFUL" in result.stdout
            
            if success:
                return True, "All imports successful", {"stdout": result.stdout}
            else:
                return False, "Import failed", {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return False, "Import test timed out", {}
        except Exception as e:
            return False, f"Import test error: {str(e)}", {}
    
    def test_api_server_running(self) -> tuple:
        """Test that the API server is running and responsive."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                return True, f"API server running (status: {health_data.get('status', 'unknown')})", health_data
            else:
                return False, f"API server returned status {response.status_code}", {"response": response.text}
                
        except requests.exceptions.ConnectionError:
            return False, "API server not reachable - is it running on port 8200?", {}
        except requests.exceptions.Timeout:
            return False, "API server timeout", {}
        except Exception as e:
            return False, f"API server test error: {str(e)}", {}
    
    def test_dataset_validation_endpoint(self) -> tuple:
        """Test the dataset validation endpoint."""
        # Create a simple test dataset
        test_dataset = [
            {
                "instruction": "What is 2+2?",
                "output": "2+2 equals 4.",
                "input": ""
            },
            {
                "instruction": "What is the capital of France?",
                "output": "The capital of France is Paris."
            }
        ]
        
        # Write test dataset
        test_file = "/tmp/team_b_test_dataset.json"
        with open(test_file, 'w') as f:
            json.dump(test_dataset, f, indent=2)
        
        try:
            # Test validation endpoint
            response = requests.post(
                f"{self.api_base_url}/v1/datasets/validate",
                json={"file_path": test_file},
                timeout=30
            )
            
            if response.status_code == 200:
                validation_data = response.json()
                is_valid = validation_data.get("valid", False)
                detected_format = validation_data.get("format", "unknown")
                
                success = is_valid and detected_format == "alpaca"
                message = f"Dataset validation {'passed' if success else 'failed'} (format: {detected_format})"
                
                return success, message, validation_data
            else:
                return False, f"Validation endpoint returned status {response.status_code}", {"response": response.text}
                
        except Exception as e:
            return False, f"Dataset validation test error: {str(e)}", {}
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_training_job_creation(self) -> tuple:
        """Test creating a training job with LoRA."""
        # Create minimal test dataset
        test_dataset = [
            {
                "instruction": "What is machine learning?",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Explain neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process and transmit information."
            }
        ]
        
        test_file = "/tmp/team_b_training_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_dataset, f, indent=2)
        
        training_request = {
            "model": "mlx-community/Qwen2.5-1.5B-4bit",
            "experiment_name": f"validation_test_{int(time.time())}",
            "training_file": test_file,
            "dataset_config": {
                "dataset_format": "alpaca",
                "batch_size": 2,
                "max_seq_length": 512
            },
            "lora_config": {
                "use_lora": True,
                "lora_r": 4,
                "lora_alpha": 8.0,
                "lora_dropout": 0.1,
                "use_qlora": False
            },
            "n_epochs": 1,
            "learning_rate": 1e-4
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/v1/fine-tuning/jobs",
                json=training_request,
                timeout=60
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")
                lora_enabled = job_data.get("lora_info", {}).get("lora_enabled", False)
                
                success = job_id is not None and lora_enabled
                message = f"Training job created {'with LoRA' if lora_enabled else 'without LoRA'} (ID: {job_id})"
                
                return success, message, job_data
            else:
                return False, f"Training job creation returned status {response.status_code}", {"response": response.text}
                
        except Exception as e:
            return False, f"Training job creation test error: {str(e)}", {}
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_health_endpoint_features(self) -> tuple:
        """Test that health endpoint reports correct features."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                features = health_data.get("features", {})
                
                required_features = ["lora", "dataset_formats", "distributed"]
                missing_features = [f for f in required_features if not features.get(f, False)]
                
                lora_available = features.get("lora", False)
                datasets_available = len(features.get("dataset_formats", [])) > 0
                
                success = len(missing_features) == 0 and lora_available and datasets_available
                message = f"Features: LoRA={lora_available}, Datasets={datasets_available}"
                
                details = {
                    "features": features,
                    "missing_features": missing_features,
                    "all_features_available": success
                }
                
                return success, message, details
            else:
                return False, f"Health endpoint returned status {response.status_code}", {}
                
        except Exception as e:
            return False, f"Health endpoint test error: {str(e)}", {}
    
    def test_model_list_endpoint(self) -> tuple:
        """Test model list endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/v1/models", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                
                has_models = len(models) > 0
                has_lora_configs = any("recommended_lora_config" in model for model in models)
                
                success = has_models and has_lora_configs
                message = f"Found {len(models)} models with LoRA configs"
                
                return success, message, {"models": models}
            else:
                return False, f"Models endpoint returned status {response.status_code}", {}
                
        except Exception as e:
            return False, f"Models endpoint test error: {str(e)}", {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        self.log("🚀 Starting Team B Integration Validation", "INFO")
        self.log(f"API Base URL: {self.api_base_url}")
        
        # Define tests to run
        tests = [
            ("File Structure", self.test_file_structure),
            ("Python Imports", self.test_python_imports),
            ("API Server Running", self.test_api_server_running),
            ("Health Endpoint Features", self.test_health_endpoint_features),
            ("Model List Endpoint", self.test_model_list_endpoint),
            ("Dataset Validation Endpoint", self.test_dataset_validation_endpoint),
            ("Training Job Creation", self.test_training_job_creation),
        ]
        
        # Run tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(0.5)  # Small delay between tests
        
        # Generate summary
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_duration = sum(r.duration for r in self.results)
        
        summary = {
            "total_tests": len(self.results),
            "passed": len(passed_tests),
            "failed": len(failed_tests),
            "success_rate": len(passed_tests) / len(self.results) * 100,
            "total_duration": total_duration,
            "all_passed": len(failed_tests) == 0,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("TEAM B INTEGRATION VALIDATION SUMMARY")
        print("=" * 80)
        
        print(f"📊 Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        print(f"⏱️  Total duration: {summary['total_duration']:.2f}s")
        
        if summary['all_passed']:
            print("🎉 ALL TESTS PASSED! Integration is working correctly.")
        else:
            print("⚠️  Some tests failed. See details below.")
        
        print("\n📋 Test Details:")
        print("-" * 40)
        
        for result in summary['results']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['name']}: {result['message']} ({result['duration']:.2f}s)")
            
            if not result['passed'] and result.get('details'):
                print(f"    Details: {result['details']}")
        
        if not summary['all_passed']:
            print("\n🔧 Troubleshooting:")
            print("-" * 20)
            
            failed_tests = [r for r in summary['results'] if not r['passed']]
            
            for result in failed_tests:
                print(f"\n❌ {result['name']}:")
                
                if "File Structure" in result['name']:
                    print("  - Run the auto-setup script: ./team_b_auto_setup.sh")
                    print("  - Check that the mlx_distributed directory exists")
                
                elif "Python Imports" in result['name']:
                    print("  - Ensure all files were copied correctly")
                    print("  - Check Python path and dependencies")
                
                elif "API Server" in result['name']:
                    print("  - Start your API server on port 8200")
                    print("  - Check server logs for errors")
                
                elif "Endpoint" in result['name']:
                    print("  - Update your API with the provided code modifications")
                    print("  - Restart your API server after making changes")
        
        print(f"\n{'=' * 80}")

def main():
    """Run validation with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Team B Integration Validation Script")
    parser.add_argument("--api-url", default="http://localhost:8200", help="API base URL")
    parser.add_argument("--json-output", action="store_true", help="Output results as JSON")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    
    args = parser.parse_args()
    
    validator = TeamBValidator(api_base_url=args.api_url)
    
    if args.quick:
        # Run only file structure and import tests
        validator.run_test("File Structure", validator.test_file_structure)
        validator.run_test("Python Imports", validator.test_python_imports)
        validator.run_test("API Server Running", validator.test_api_server_running)
    else:
        # Run all tests
        summary = validator.run_all_tests()
    
    if args.quick:
        passed = all(r.passed for r in validator.results)
        print(f"\n🚀 Quick validation {'✅ passed' if passed else '❌ failed'}")
        return 0 if passed else 1
    
    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        validator.print_summary(summary)
    
    return 0 if summary['all_passed'] else 1

if __name__ == "__main__":
    sys.exit(main())
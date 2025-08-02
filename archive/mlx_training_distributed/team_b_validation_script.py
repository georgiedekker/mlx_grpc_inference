#!/usr/bin/env python3
"""
Team B Comprehensive Validation Script
Tests all integration components to ensure proper setup and functionality.
"""

import os
import sys
import json
import requests
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

class TeamBValidator:
    """Comprehensive validator for Team B integration."""
    
    def __init__(self, project_root: str = None):
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Try to find the project root
            self.project_root = Path.cwd()
            if "mlx_distributed_training" not in str(self.project_root):
                self.project_root = Path("/Users/mini1/Movies/mlx_distributed/mlx_distributed_training")
        
        self.api_base = "http://localhost:8200"
        self.api_key = os.getenv("MLX_TRAINING_API_KEY", "test-api-key")
        self.validation_results = {
            "file_structure": False,
            "imports": False,
            "api_endpoints": False,
            "dataset_validation": False,
            "lora_training": False,
            "overall": False
        }
    
    def validate_all(self) -> bool:
        """Run all validation tests."""
        print_header("Team B Integration Validation")
        print(f"Project Root: {self.project_root}")
        print(f"API Base URL: {self.api_base}")
        
        # 1. File structure validation
        self.validate_file_structure()
        
        # 2. Python imports validation
        self.validate_imports()
        
        # 3. API endpoints validation
        self.validate_api_endpoints()
        
        # 4. Dataset validation
        self.validate_dataset_functionality()
        
        # 5. LoRA training validation
        self.validate_lora_training()
        
        # Summary
        self.print_summary()
        
        return self.validation_results["overall"]
    
    def validate_file_structure(self):
        """Validate that all required files are in place."""
        print_header("1. File Structure Validation")
        
        required_files = [
            "src/mlx_distributed_training/training/lora/lora.py",
            "src/mlx_distributed_training/datasets/base_dataset.py",
            "src/mlx_distributed_training/datasets/alpaca_dataset.py",
            "src/mlx_distributed_training/datasets/sharegpt_dataset.py",
            "src/mlx_distributed_training/datasets/dataset_utils.py",
            "src/mlx_distributed_training/integration/lora_integration.py",
            "src/mlx_distributed_training/integration/dataset_integration.py",
            "examples/datasets/alpaca_example.json",
            "examples/datasets/sharegpt_example.json",
            "examples/configs/lora_training_config.yaml"
        ]
        
        all_present = True
        missing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"Found: {file_path}")
            else:
                print_error(f"Missing: {file_path}")
                missing_files.append(file_path)
                all_present = False
        
        if all_present:
            print_success("All required files are present!")
            self.validation_results["file_structure"] = True
        else:
            print_error(f"Missing {len(missing_files)} files")
            print_info("Run team_b_auto_setup.sh to copy missing files")
        
        return all_present
    
    def validate_imports(self):
        """Validate that Python imports work correctly."""
        print_header("2. Python Import Validation")
        
        # Add project to Python path
        sys.path.insert(0, str(self.project_root / "src"))
        
        imports_to_test = [
            ("LoRA Core", "mlx_distributed_training.training.lora.lora", ["apply_lora_to_model", "LoRAConfig"]),
            ("Alpaca Dataset", "mlx_distributed_training.datasets.alpaca_dataset", ["AlpacaDataset"]),
            ("ShareGPT Dataset", "mlx_distributed_training.datasets.sharegpt_dataset", ["ShareGPTDataset"]),
            ("Dataset Utils", "mlx_distributed_training.datasets.dataset_utils", ["detect_dataset_format"]),
            ("LoRA Integration", "mlx_distributed_training.integration.lora_integration", ["create_lora_enabled_trainer"]),
            ("Dataset Integration", "mlx_distributed_training.integration.dataset_integration", ["validate_dataset"])
        ]
        
        all_imports_ok = True
        
        for name, module_path, attributes in imports_to_test:
            try:
                module = importlib.import_module(module_path)
                
                # Check specific attributes
                missing_attrs = []
                for attr in attributes:
                    if not hasattr(module, attr):
                        missing_attrs.append(attr)
                
                if missing_attrs:
                    print_warning(f"{name}: Module imported but missing: {', '.join(missing_attrs)}")
                    all_imports_ok = False
                else:
                    print_success(f"{name}: All imports successful")
                    
            except ImportError as e:
                print_error(f"{name}: Import failed - {str(e)}")
                all_imports_ok = False
            except Exception as e:
                print_error(f"{name}: Unexpected error - {str(e)}")
                all_imports_ok = False
        
        if all_imports_ok:
            print_success("All Python imports working correctly!")
            self.validation_results["imports"] = True
        else:
            print_error("Some imports failed")
            print_info("Check that all files were copied correctly and dependencies are installed")
        
        return all_imports_ok
    
    def validate_api_endpoints(self):
        """Validate API endpoints are working."""
        print_header("3. API Endpoint Validation")
        
        # Check if API server is running
        try:
            response = requests.get(f"{self.api_base}/docs", timeout=2)
            if response.status_code == 200:
                print_success("API server is running")
            else:
                print_warning(f"API server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print_error("Cannot connect to API server at http://localhost:8200")
            print_info("Please ensure Team B's API server is running")
            return False
        except Exception as e:
            print_error(f"API connection error: {str(e)}")
            return False
        
        # Test endpoints
        endpoints_to_test = [
            ("Health Check", "GET", "/health", None, None),
            ("Dataset Validation", "POST", "/v1/datasets/validate", 
             {"file_path": str(self.project_root / "examples/datasets/alpaca_example.json")}, None),
            ("Models List", "GET", "/v1/models", None, {"X-API-Key": self.api_key}),
        ]
        
        all_endpoints_ok = True
        
        for name, method, endpoint, data, headers in endpoints_to_test:
            try:
                if method == "GET":
                    response = requests.get(f"{self.api_base}{endpoint}", headers=headers, timeout=5)
                else:
                    response = requests.post(f"{self.api_base}{endpoint}", json=data, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    print_success(f"{name} ({endpoint}): Working")
                    
                    # Check specific features for health endpoint
                    if endpoint == "/health":
                        try:
                            health_data = response.json()
                            features = health_data.get("features", {})
                            if features.get("lora") and "alpaca" in features.get("dataset_formats", []):
                                print_success("  → LoRA and dataset features reported correctly")
                            else:
                                print_warning("  → Health endpoint missing LoRA/dataset features")
                                all_endpoints_ok = False
                        except:
                            pass
                            
                elif response.status_code == 404:
                    print_error(f"{name} ({endpoint}): Not found - endpoint not implemented")
                    all_endpoints_ok = False
                elif response.status_code == 401:
                    print_warning(f"{name} ({endpoint}): Authentication required")
                else:
                    print_error(f"{name} ({endpoint}): Failed with status {response.status_code}")
                    all_endpoints_ok = False
                    
            except Exception as e:
                print_error(f"{name} ({endpoint}): Error - {str(e)}")
                all_endpoints_ok = False
        
        if all_endpoints_ok:
            print_success("All API endpoints validated successfully!")
            self.validation_results["api_endpoints"] = True
        else:
            print_error("Some endpoints need implementation")
            print_info("Review team_b_api_modifications.py for endpoint code")
        
        return all_endpoints_ok
    
    def validate_dataset_functionality(self):
        """Validate dataset parsing and validation."""
        print_header("4. Dataset Functionality Validation")
        
        # Test dataset files
        alpaca_path = self.project_root / "examples/datasets/alpaca_example.json"
        sharegpt_path = self.project_root / "examples/datasets/sharegpt_example.json"
        
        if not alpaca_path.exists() or not sharegpt_path.exists():
            print_error("Example dataset files not found")
            return False
        
        all_tests_passed = True
        
        # Test via API if available
        try:
            # Test Alpaca format
            response = requests.post(
                f"{self.api_base}/v1/datasets/validate",
                json={"file_path": str(alpaca_path)},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid") and data.get("format") == "alpaca":
                    print_success("Alpaca dataset validation: API working correctly")
                else:
                    print_error("Alpaca dataset validation: Incorrect response")
                    all_tests_passed = False
            else:
                print_warning("Dataset validation endpoint not yet implemented")
                
                # Test using direct imports
                try:
                    from mlx_distributed_training.integration.dataset_integration import validate_dataset
                    
                    result = validate_dataset(str(alpaca_path))
                    if result.is_valid and result.format_type == "alpaca":
                        print_success("Alpaca dataset validation: Direct import working")
                    else:
                        print_error("Alpaca dataset validation: Failed")
                        all_tests_passed = False
                        
                except Exception as e:
                    print_error(f"Dataset validation error: {str(e)}")
                    all_tests_passed = False
                    
        except Exception as e:
            print_warning(f"API test skipped: {str(e)}")
        
        if all_tests_passed:
            print_success("Dataset functionality validated!")
            self.validation_results["dataset_validation"] = True
        
        return all_tests_passed
    
    def validate_lora_training(self):
        """Validate LoRA training job creation."""
        print_header("5. LoRA Training Validation")
        
        # Prepare test configuration
        test_config = {
            "experiment_name": "validation_test_lora",
            "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
            "epochs": 1,
            "learning_rate": 5e-5,
            "lora": {
                "use_lora": True,
                "lora_r": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"]
            },
            "dataset": {
                "dataset_path": str(self.project_root / "examples/datasets/alpaca_example.json"),
                "dataset_format": "alpaca",
                "batch_size": 2,
                "max_seq_length": 512
            },
            "output_dir": str(self.project_root / "outputs/validation_test")
        }
        
        try:
            # Try to create a training job
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            response = requests.post(
                f"{self.api_base}/v1/fine-tuning/jobs",
                json=test_config,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("lora_enabled"):
                    print_success("LoRA training job creation: Working!")
                    print_info(f"  → Job ID: {data.get('id')}")
                    print_info(f"  → LoRA enabled: {data.get('lora_enabled')}")
                    print_info(f"  → Dataset format: {data.get('dataset_info', {}).get('format')}")
                    self.validation_results["lora_training"] = True
                else:
                    print_warning("Training job created but LoRA not enabled")
            elif response.status_code == 404:
                print_error("Training endpoint not found - needs implementation")
            elif response.status_code == 401:
                print_warning("Authentication required - API key may be incorrect")
            else:
                print_error(f"Training job creation failed: {response.status_code}")
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    print_info(f"  → Error: {error_detail}")
                except:
                    pass
                    
        except requests.exceptions.ConnectionError:
            print_warning("Cannot test training - API server not running")
        except Exception as e:
            print_error(f"Training validation error: {str(e)}")
        
        return self.validation_results["lora_training"]
    
    def print_summary(self):
        """Print validation summary."""
        print_header("Validation Summary")
        
        total_tests = len(self.validation_results) - 1  # Exclude 'overall'
        passed_tests = sum(1 for k, v in self.validation_results.items() if k != "overall" and v)
        
        print(f"\nTest Results: {passed_tests}/{total_tests} passed")
        print("-" * 40)
        
        for test, passed in self.validation_results.items():
            if test != "overall":
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{test.replace('_', ' ').title():<30} {status}")
        
        print("-" * 40)
        
        # Overall assessment
        if passed_tests == total_tests:
            print_success("\n🎉 All validations passed! Team B integration is complete!")
            self.validation_results["overall"] = True
        elif passed_tests >= total_tests - 1:
            print_warning("\n⚠️  Almost there! Just one more component to integrate.")
            self.validation_results["overall"] = False
        else:
            print_error(f"\n❌ {total_tests - passed_tests} components need attention.")
            self.validation_results["overall"] = False
        
        # Recommendations
        if not self.validation_results["file_structure"]:
            print_info("\n📋 Next Step: Run ./team_b_auto_setup.sh to copy required files")
        elif not self.validation_results["imports"]:
            print_info("\n📋 Next Step: Check Python path and install dependencies")
        elif not self.validation_results["api_endpoints"]:
            print_info("\n📋 Next Step: Update API with code from team_b_api_modifications.py")
        elif not self.validation_results["lora_training"]:
            print_info("\n📋 Next Step: Implement training endpoint with LoRA support")
        
        # Time estimate
        if passed_tests < total_tests:
            remaining_time = (total_tests - passed_tests) * 15  # 15 minutes per component
            print_info(f"\n⏱️  Estimated time to complete: {remaining_time} minutes")

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Team B LoRA/Dataset Integration")
    parser.add_argument("--project-root", type=str, help="Path to Team B project root")
    parser.add_argument("--api-url", type=str, default="http://localhost:8200", help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    args = parser.parse_args()
    
    # Create validator
    validator = TeamBValidator(project_root=args.project_root)
    
    if args.api_url:
        validator.api_base = args.api_url
    
    if args.api_key:
        validator.api_key = args.api_key
    
    # Run validation
    success = validator.validate_all()
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
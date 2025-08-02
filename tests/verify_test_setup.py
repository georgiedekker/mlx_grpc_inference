#!/usr/bin/env python3
"""
Verify test setup and dependencies are working correctly.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_imports():
    """Check that all required imports work."""
    print("🔍 Checking imports...")
    
    required_modules = [
        'pytest',
        'pytest_asyncio',
        'pytest_cov',
        'mlx.core',
        'numpy',
        'grpc',
        'yaml',
        'pydantic',
        'fastapi',
        'lz4'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        return False
    else:
        print("\n✅ All required modules available")
        return True


def check_test_files():
    """Check that test files are properly structured."""
    print("\n📁 Checking test file structure...")
    
    test_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "tests/performance",
        "tests/fixtures"
    ]
    
    for test_dir in test_dirs:
        dir_path = Path(test_dir)
        if dir_path.exists():
            test_files = list(dir_path.glob("test_*.py"))
            print(f"  ✅ {test_dir}: {len(test_files)} test files")
        else:
            print(f"  ❌ {test_dir}: directory not found")
            return False
    
    # Check for key test files
    key_files = [
        "tests/conftest.py",
        "tests/unit/test_orchestrator.py",
        "tests/unit/test_layer_processor.py",
        "tests/unit/test_grpc_client.py",
        "tests/integration/test_distributed_forward.py",
        "tests/e2e/test_full_inference.py",
        "tests/performance/test_benchmarks.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}: not found")
            return False
    
    return True


def run_sample_tests():
    """Run a few sample tests to verify everything works."""
    print("\n🧪 Running sample tests...")
    
    # Test layer processor (should work)
    cmd = ["uv", "run", "python", "-m", "pytest", 
           "tests/unit/test_layer_processor.py::TestLayerProcessor::test_init", "-v"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✅ Layer processor test passed")
        else:
            print(f"  ❌ Layer processor test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ Test timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error running test: {e}")
        return False
    
    return True


def check_config_files():
    """Check configuration files."""
    print("\n⚙️  Checking configuration files...")
    
    config_files = [
        "pytest.ini",
        "pyproject.toml",
        "tests/fixtures/test_cluster_config.yaml",
        "tests/fixtures/single_device_config.yaml",
        "tests/fixtures/performance_config.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}: not found")
            return False
    
    return True


def check_pytest_config():
    """Check pytest configuration."""
    print("\n🔧 Checking pytest configuration...")
    
    cmd = ["uv", "run", "python", "-m", "pytest", "--collect-only", "-q"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        # Check for critical errors (not just any stderr output)
        critical_errors = ["ImportError", "SyntaxError", "ModuleNotFoundError"]
        has_critical_error = any(error in result.stderr for error in critical_errors)
        
        if has_critical_error or result.returncode != 0:
            print(f"  ❌ Pytest collection failed: {result.stderr}")
            return False
        else:
            # Count tests collected
            test_lines = [line for line in result.stdout.split('\n') if '::' in line and 'test_' in line]
            test_count = len(test_lines)
            print(f"  ✅ Collected {test_count} tests successfully")
    except Exception as e:
        print(f"  ❌ Error checking pytest config: {e}")
        return False
    
    return True


def main():
    """Run all verification checks."""
    print("🚀 Verifying test setup for MLX Distributed Inference System")
    print("=" * 60)
    
    checks = [
        ("Import dependencies", check_imports),
        ("Test file structure", check_test_files),
        ("Configuration files", check_config_files),
        ("Pytest configuration", check_pytest_config),
        ("Sample test execution", run_sample_tests)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"  ❌ {check_name}: Exception - {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All test setup verification checks passed!")
        print("\nYou can now run tests with:")
        print("  • make test (all tests)")
        print("  • make test-unit (unit tests only)")
        print("  • make test-fast (exclude slow tests)")
        print("  • uv run python tests/run_tests.py --suite unit")
        return 0
    else:
        print("💥 Some verification checks failed!")
        print("\nPlease fix the issues above before running tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script for MLX Training servers
"""

import requests
import json
import time
from typing import Dict, Any

def test_server(port: int, name: str) -> Dict[str, Any]:
    """Test a server on given port."""
    base_url = f"http://localhost:{port}"
    results = {"name": name, "port": port, "tests": []}
    
    print(f"\n🧪 Testing {name} on port {port}")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            results["tests"].append({
                "test": "Health Check",
                "status": "✅ PASS",
                "details": {
                    "service": health.get("service"),
                    "version": health.get("version"),
                    "implementation": health.get("implementation", "unknown")
                }
            })
            print(f"✅ Health Check: {health['service']} v{health['version']}")
            if "implementation" in health:
                print(f"   Implementation: {health['implementation']}")
        else:
            results["tests"].append({
                "test": "Health Check", 
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Health Check: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Health Check",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Health Check: {str(e)}")
        return results
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            root_info = response.json()
            endpoints = root_info.get("endpoints", {})
            results["tests"].append({
                "test": "Root Info",
                "status": "✅ PASS",
                "details": {"endpoints": len(str(endpoints))}
            })
            print(f"✅ Root Info: {len(endpoints)} endpoint categories")
        else:
            results["tests"].append({
                "test": "Root Info",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Root Info: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Root Info",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Root Info: {str(e)}")
    
    # Test 3: API Documentation
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            results["tests"].append({
                "test": "API Docs",
                "status": "✅ PASS"
            })
            print(f"✅ API Docs: Available at {base_url}/docs")
        else:
            results["tests"].append({
                "test": "API Docs",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ API Docs: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "API Docs",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ API Docs: {str(e)}")
    
    return results

def test_mlx_training_specific(port: int) -> Dict[str, Any]:
    """Test MLX Training Framework specific endpoints."""
    base_url = f"http://localhost:{port}"
    results = {"tests": []}
    
    print(f"\n🔧 Testing MLX Training Framework Specific Features")
    print("-" * 50)
    
    # Test optimizers endpoint
    try:
        response = requests.get(f"{base_url}/v1/optimizers", timeout=5)
        if response.status_code == 200:
            optimizers = response.json()
            opt_count = len(optimizers.get("optimizers", {}))
            results["tests"].append({
                "test": "Optimizers List",
                "status": "✅ PASS",
                "details": {"optimizer_count": opt_count}
            })
            print(f"✅ Optimizers: {opt_count} available")
            
            # Test specific optimizer
            response = requests.get(f"{base_url}/v1/optimizers/adamw", timeout=5)
            if response.status_code == 200:
                results["tests"].append({
                    "test": "Optimizer Details",
                    "status": "✅ PASS"
                })
                print(f"✅ AdamW Config: Available")
            else:
                results["tests"].append({
                    "test": "Optimizer Details",
                    "status": f"❌ FAIL ({response.status_code})"
                })
        else:
            results["tests"].append({
                "test": "Optimizers List",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Optimizers: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Optimizers List",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Optimizers: {str(e)}")
    
    # Test dataset formats
    try:
        response = requests.get(f"{base_url}/v1/datasets/formats", timeout=5)
        if response.status_code == 200:
            formats = response.json().get("formats", {})
            results["tests"].append({
                "test": "Dataset Formats",
                "status": "✅ PASS",
                "details": {"format_count": len(formats)}
            })
            print(f"✅ Dataset Formats: {len(formats)} supported")
        else:
            results["tests"].append({
                "test": "Dataset Formats",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Dataset Formats: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Dataset Formats",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Dataset Formats: {str(e)}")
    
    # Test training jobs list
    try:
        response = requests.get(f"{base_url}/v1/training/jobs", timeout=5)
        if response.status_code == 200:
            jobs = response.json()
            results["tests"].append({
                "test": "Training Jobs",
                "status": "✅ PASS",
                "details": {"job_count": jobs.get("total", 0)}
            })
            print(f"✅ Training Jobs: {jobs.get('total', 0)} jobs")
        else:
            results["tests"].append({
                "test": "Training Jobs",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Training Jobs: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Training Jobs",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Training Jobs: {str(e)}")
    
    return results

def test_unified_platform_specific(port: int) -> Dict[str, Any]:
    """Test Unified Platform specific endpoints."""
    base_url = f"http://localhost:{port}"
    results = {"tests": []}
    
    print(f"\n🚀 Testing Unified Platform Specific Features")
    print("-" * 50)
    
    # Test workflow templates
    try:
        response = requests.get(f"{base_url}/v1/workflows/templates", timeout=5)
        if response.status_code == 200:
            templates = response.json().get("templates", {})
            results["tests"].append({
                "test": "Workflow Templates",
                "status": "✅ PASS",
                "details": {"template_count": len(templates)}
            })
            print(f"✅ Workflow Templates: {len(templates)} available")
            
            # Print template names
            for name, info in templates.items():
                print(f"   - {info['name']}: {info['description'][:50]}...")
                
        else:
            results["tests"].append({
                "test": "Workflow Templates",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Workflow Templates: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Workflow Templates",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Workflow Templates: {str(e)}")
    
    # Test pipelines list
    try:
        response = requests.get(f"{base_url}/v1/pipelines", timeout=5)
        if response.status_code == 200:
            pipelines = response.json()
            results["tests"].append({
                "test": "Pipelines List",
                "status": "✅ PASS",
                "details": {"pipeline_count": pipelines.get("total", 0)}
            })
            print(f"✅ Pipelines: {pipelines.get('total', 0)} pipelines")
        else:
            results["tests"].append({
                "test": "Pipelines List",
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Pipelines: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Pipelines List",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Pipelines: {str(e)}")
    
    # Test jobs endpoint
    try:
        response = requests.get(f"{base_url}/v1/jobs", timeout=5)
        if response.status_code == 200:
            jobs = response.json()
            results["tests"].append({
                "test": "Jobs List",
                "status": "✅ PASS",
                "details": {"job_count": jobs.get("total", 0)}
            })
            print(f"✅ Jobs: {jobs.get('total', 0)} jobs")
        else:
            results["tests"].append({
                "test": "Jobs List", 
                "status": f"❌ FAIL ({response.status_code})"
            })
            print(f"❌ Jobs: HTTP {response.status_code}")
    except Exception as e:
        results["tests"].append({
            "test": "Jobs List",
            "status": f"❌ ERROR: {str(e)}"
        })
        print(f"❌ Jobs: {str(e)}")
    
    return results

def main():
    """Run comprehensive tests on both servers."""
    print("🧪 MLX Training Framework Test Suite")
    print("=" * 60)
    
    # Test both servers
    results = []
    
    # Test MLX Training Framework (8500)
    mlx_results = test_server(8500, "MLX Training Framework")
    if any("✅ PASS" in test["status"] for test in mlx_results["tests"]):
        mlx_specific = test_mlx_training_specific(8500)
        mlx_results["tests"].extend(mlx_specific["tests"])
    results.append(mlx_results)
    
    # Test Unified Platform (8600)
    unified_results = test_server(8600, "MLX Unified Training Platform")
    if any("✅ PASS" in test["status"] for test in unified_results["tests"]):
        unified_specific = test_unified_platform_specific(8600)
        unified_results["tests"].extend(unified_specific["tests"])
    results.append(unified_results)
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    
    for server_result in results:
        passed = len([t for t in server_result["tests"] if "✅ PASS" in t["status"]])
        total = len(server_result["tests"])
        
        print(f"\n🖥️  {server_result['name']} (Port {server_result['port']})")
        print(f"   Tests Passed: {passed}/{total}")
        
        if passed == total:
            print(f"   Status: ✅ ALL TESTS PASSED")
        elif passed > 0:
            print(f"   Status: ⚠️  PARTIAL SUCCESS")
        else:
            print(f"   Status: ❌ ALL TESTS FAILED")
            
        # Show failed tests
        failed = [t for t in server_result["tests"] if "❌" in t["status"]]
        if failed:
            print(f"   Failed Tests:")
            for test in failed:
                print(f"     - {test['test']}: {test['status']}")
    
    # Overall status
    total_passed = sum(len([t for t in r["tests"] if "✅ PASS" in t["status"]]) for r in results)
    total_tests = sum(len(r["tests"]) for r in results)
    
    print(f"\n🎯 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
    elif total_passed > 0:
        print("⚠️  SOME SYSTEMS OPERATIONAL")
    else:
        print("🚨 NO SYSTEMS OPERATIONAL")

if __name__ == "__main__":
    main()
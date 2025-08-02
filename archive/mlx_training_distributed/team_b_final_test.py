#!/usr/bin/env python3
"""
Team B Final Test Suite - Proving 1/5 → 5/5 Success
This test mimics the original failing test scenario and proves all endpoints now work.
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8200"
API_KEY = "test-api-key"

def test_original_failing_scenario():
    """Test the original scenario that was giving Team B 1/5 success."""
    print("🧪 TEAM B FINAL TEST - PROVING 1/5 → 5/5 SUCCESS")
    print("=" * 60)
    
    results = {"passed": 0, "total": 5, "details": []}
    
    # TEST 1: Basic server connectivity (this was the 1/5 that passed)
    print("\n1️⃣  Testing basic server connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            results["passed"] += 1
            results["details"].append("✅ Server connectivity - PASS")
            print("   ✅ Server is running and accessible")
        else:
            results["details"].append("❌ Server connectivity - FAIL") 
            print("   ❌ Server not accessible")
    except Exception as e:
        results["details"].append(f"❌ Server connectivity - FAIL: {e}")
        print(f"   ❌ Server error: {e}")
    
    # TEST 2: Health endpoint with LoRA features (this was missing before)
    print("\n2️⃣  Testing health endpoint with LoRA features...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            has_lora = data.get("features", {}).get("lora", False)
            has_qlora = data.get("features", {}).get("qlora", False)
            has_formats = "alpaca" in data.get("features", {}).get("dataset_formats", [])
            
            if has_lora and has_qlora and has_formats:
                results["passed"] += 1
                results["details"].append("✅ Health endpoint features - PASS")
                print("   ✅ Health endpoint reports LoRA, QLoRA, and dataset format support")
            else:
                results["details"].append("❌ Health endpoint features - FAIL")
                print("   ❌ Health endpoint missing LoRA/dataset features")
        else:
            results["details"].append("❌ Health endpoint features - FAIL")
            print("   ❌ Health endpoint failed")
    except Exception as e:
        results["details"].append(f"❌ Health endpoint features - FAIL: {e}")
        print(f"   ❌ Health endpoint error: {e}")
    
    # TEST 3: Dataset validation endpoint (this was missing before)
    print("\n3️⃣  Testing dataset validation endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/datasets/validate",
            json={"file_path": "/tmp/team_b_test_alpaca.json"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            is_valid = data.get("valid", False)
            format_detected = data.get("format") == "alpaca"
            
            if is_valid and format_detected:
                results["passed"] += 1
                results["details"].append("✅ Dataset validation - PASS")
                print("   ✅ Dataset validation works with format detection")
            else:
                results["details"].append("❌ Dataset validation - FAIL")
                print("   ❌ Dataset validation failed")
        else:
            results["details"].append("❌ Dataset validation - FAIL")
            print("   ❌ Dataset validation endpoint failed")
    except Exception as e:
        results["details"].append(f"❌ Dataset validation - FAIL: {e}")
        print(f"   ❌ Dataset validation error: {e}")
    
    # TEST 4: LoRA training job creation (this was missing before)
    print("\n4️⃣  Testing LoRA training job creation...")
    try:
        job_request = {
            "experiment_name": f"test_final_{int(time.time())}",
            "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
            "epochs": 1,
            "learning_rate": 5e-5,
            "lora": {
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32.0,
                "lora_dropout": 0.1,
                "use_qlora": True
            },
            "dataset": {
                "dataset_path": "/tmp/team_b_test_alpaca.json",
                "batch_size": 4
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/fine-tuning/jobs",
            json=job_request,
            headers={"X-API-Key": API_KEY},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            has_job_id = "id" in data
            lora_enabled = data.get("lora_enabled", False)
            has_memory_savings = data.get("lora_details", {}).get("memory_savings_pct", 0) > 0
            
            if has_job_id and lora_enabled and has_memory_savings:
                results["passed"] += 1
                results["details"].append("✅ LoRA training job creation - PASS")
                print(f"   ✅ LoRA training job created: {data.get('id')}")
                print(f"   ✅ Memory savings: {data.get('lora_details', {}).get('memory_savings_pct')}%")
                
                # Store job ID for next test
                global last_job_id
                last_job_id = data.get("id")
            else:
                results["details"].append("❌ LoRA training job creation - FAIL")
                print("   ❌ LoRA training job creation failed")
        else:
            results["details"].append("❌ LoRA training job creation - FAIL")
            print(f"   ❌ Training job creation failed: {response.status_code}")
    except Exception as e:
        results["details"].append(f"❌ LoRA training job creation - FAIL: {e}")
        print(f"   ❌ Training job creation error: {e}")
    
    # TEST 5: Training job status tracking (this was missing before)
    print("\n5️⃣  Testing training job status tracking...")
    try:
        if 'last_job_id' in globals():
            response = requests.get(
                f"{API_BASE_URL}/v1/fine-tuning/jobs/{last_job_id}",
                headers={"X-API-Key": API_KEY},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                has_status = "status" in data
                has_progress = "progress" in data
                has_metrics = "metrics" in data
                
                if has_status and has_progress and has_metrics:
                    results["passed"] += 1
                    results["details"].append("✅ Training job status tracking - PASS")
                    print(f"   ✅ Job status: {data.get('status')}")
                    print(f"   ✅ Progress: {data.get('progress', {}).get('percentage')}%")
                    print(f"   ✅ GPU memory: {data.get('metrics', {}).get('gpu_memory_gb')}GB")
                else:
                    results["details"].append("❌ Training job status tracking - FAIL")
                    print("   ❌ Job status tracking incomplete")
            else:
                results["details"].append("❌ Training job status tracking - FAIL")
                print("   ❌ Job status endpoint failed")
        else:
            results["details"].append("❌ Training job status tracking - FAIL (no job to track)")
            print("   ❌ No job ID to track (previous test failed)")
    except Exception as e:
        results["details"].append(f"❌ Training job status tracking - FAIL: {e}")
        print(f"   ❌ Job status tracking error: {e}")
    
    # FINAL RESULTS
    print("\n" + "=" * 60)
    print("🏆 FINAL TEST RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Tests passed: {results['passed']}/{results['total']}")
    print(f"📈 Success rate: {results['passed']/results['total']*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for detail in results["details"]:
        print(f"   {detail}")
    
    if results["passed"] == 5:
        print(f"\n🎉 SUCCESS! Team B achieved 5/5 test success!")
        print(f"🚀 From 1/5 (basic server only) to 5/5 (full LoRA/dataset support)!")
        print(f"✨ All LoRA features, dataset validation, and training jobs working!")
    else:
        print(f"\n⚠️  Still need work: {5 - results['passed']} tests failing")
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    # Wait a moment for server to be ready
    time.sleep(1)
    
    # Run the comprehensive test
    results = test_original_failing_scenario()
    
    # Exit with appropriate code
    exit(0 if results["passed"] == 5 else 1)
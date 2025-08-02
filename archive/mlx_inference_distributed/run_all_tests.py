#!/usr/bin/env python3
"""
Comprehensive test runner for MLX Distributed.
Runs unit tests, integration tests, and generates coverage reports.
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime


class TestRunner:
    """Run all tests and generate reports."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "coverage": {},
            "performance": {}
        }
    
    def run_unit_tests(self):
        """Run unit tests with coverage."""
        print("\n" + "="*60)
        print("🧪 Running Unit Tests")
        print("="*60)
        
        test_files = [
            "tests/test_hardware_detection.py",
            "tests/test_distributed_comm.py",
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n📋 Testing {test_file}...")
                
                cmd = [
                    sys.executable, "-m", "pytest", test_file,
                    "-v", "--tb=short",
                    "--cov=.", "--cov-report=term-missing"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                self.results["tests"][test_file] = {
                    "returncode": result.returncode,
                    "passed": result.returncode == 0,
                    "output": result.stdout[-1000:] if result.stdout else ""  # Last 1000 chars
                }
                
                if result.returncode == 0:
                    print(f"✅ {test_file} passed")
                else:
                    print(f"❌ {test_file} failed")
                    print(result.stdout)
                    print(result.stderr)
    
    def check_cluster_status(self):
        """Check if the cluster is running."""
        import requests
        
        try:
            response = requests.get("http://localhost:8100/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "="*60)
        print("🔗 Running Integration Tests")
        print("="*60)
        
        # Check if cluster is running
        if not self.check_cluster_status():
            print("⚠️  Cluster not running. Starting cluster...")
            
            # Start the cluster
            start_script = "./start_3device_cluster.sh"
            if os.path.exists(start_script):
                print("Starting 3-device cluster...")
                subprocess.run(["bash", start_script], capture_output=True)
                time.sleep(15)  # Wait for cluster to start
                
                if not self.check_cluster_status():
                    print("❌ Failed to start cluster")
                    return
            else:
                print("❌ Cluster start script not found")
                return
        
        print("✅ Cluster is running")
        
        # Run integration tests
        test_file = "tests/integration/test_3device_cluster.py"
        if os.path.exists(test_file):
            print(f"\n📋 Running {test_file}...")
            
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "-s"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            self.results["tests"][test_file] = {
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else ""
            }
            
            if result.returncode == 0:
                print(f"✅ Integration tests passed")
            else:
                print(f"❌ Integration tests failed")
                print(result.stdout)
                print(result.stderr)
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks."""
        print("\n" + "="*60)
        print("📊 Running Performance Benchmarks")
        print("="*60)
        
        if not self.check_cluster_status():
            print("❌ Cluster not running. Skipping benchmarks.")
            return
        
        import requests
        
        # Benchmark 1: Single inference latency
        print("\n📏 Benchmark 1: Single Inference Latency")
        prompts = [
            "Hello",
            "What is machine learning?",
            "Explain quantum computing in simple terms."
        ]
        
        latencies = []
        for prompt in prompts:
            start = time.time()
            response = requests.post(
                "http://localhost:8100/v1/chat/completions",
                json={
                    "model": "mlx-community/Qwen3-1.7B-8bit",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50
                }
            )
            latency = time.time() - start
            
            if response.status_code == 200:
                tokens = response.json()["usage"]["completion_tokens"]
                latencies.append({
                    "prompt_length": len(prompt.split()),
                    "latency": latency,
                    "tokens": tokens,
                    "tokens_per_second": tokens / latency
                })
                print(f"  Prompt: {len(prompt.split())} words, Latency: {latency:.2f}s, Tokens/s: {tokens/latency:.2f}")
        
        self.results["performance"]["single_inference"] = latencies
        
        # Benchmark 2: Throughput test
        print("\n📏 Benchmark 2: Throughput Test")
        num_requests = 5
        start = time.time()
        
        import concurrent.futures
        
        def make_request(i):
            return requests.post(
                "http://localhost:8100/v1/chat/completions",
                json={
                    "model": "mlx-community/Qwen3-1.7B-8bit",
                    "messages": [{"role": "user", "content": f"Count to {i}"}],
                    "max_tokens": 20
                }
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start
        successful = sum(1 for r in responses if r.status_code == 200)
        
        print(f"  Completed {successful}/{num_requests} requests in {total_time:.2f}s")
        print(f"  Throughput: {successful/total_time:.2f} requests/second")
        
        self.results["performance"]["throughput"] = {
            "requests": num_requests,
            "successful": successful,
            "total_time": total_time,
            "requests_per_second": successful / total_time
        }
        
        # Benchmark 3: Memory usage
        print("\n📏 Benchmark 3: Memory Usage")
        response = requests.get("http://localhost:8100/distributed/gpu-info")
        if response.status_code == 200:
            data = response.json()
            for device in data["devices"]:
                if "system_memory" in device:
                    print(f"  {device['device_id']}: {device['system_memory'].get('used_percent', 'N/A')}% memory used")
    
    def generate_report(self):
        """Generate test report."""
        print("\n" + "="*60)
        print("📄 Test Report Summary")
        print("="*60)
        
        # Count results
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"].values() if t["passed"])
        
        print(f"\n✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests} tests")
        
        # Performance summary
        if "single_inference" in self.results["performance"]:
            avg_tokens_per_sec = sum(
                l["tokens_per_second"] for l in self.results["performance"]["single_inference"]
            ) / len(self.results["performance"]["single_inference"])
            print(f"\n🚀 Average Performance: {avg_tokens_per_sec:.2f} tokens/second")
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📊 Detailed report saved to: {report_file}")
        
        return passed_tests == total_tests
    
    def run_all(self):
        """Run all tests."""
        print("🚀 MLX Distributed - Comprehensive Test Suite")
        print("=" * 60)
        
        # Install test dependencies if needed
        print("📦 Checking test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "pytest-asyncio"], 
                      capture_output=True)
        
        # Run tests
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_benchmarks()
        
        # Generate report
        success = self.generate_report()
        
        if success:
            print("\n✅ All tests passed! 🎉")
            return 0
        else:
            print("\n❌ Some tests failed. Check the report for details.")
            return 1


def main():
    """Main entry point."""
    runner = TestRunner()
    return runner.run_all()


if __name__ == "__main__":
    sys.exit(main())
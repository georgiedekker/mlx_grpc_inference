"""
Integration tests for 3-device distributed cluster.
"""

import pytest
import requests
import json
import time
import subprocess
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Test3DeviceCluster:
    """Integration tests for 3-device cluster operations."""
    
    @classmethod
    def setup_class(cls):
        """Setup test cluster before all tests."""
        cls.api_url = "http://localhost:8100"
        cls.cluster_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "start_3device_cluster.sh"
        )
        
    def test_cluster_health(self):
        """Test that all 3 devices are healthy."""
        response = requests.get(f"{self.api_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "mlx-community/Qwen3-1.7B-8bit"
    
    def test_gpu_info_shows_3_devices(self):
        """Test that GPU info endpoint shows all 3 devices."""
        response = requests.get(f"{self.api_url}/distributed/gpu-info")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check cluster info
        cluster_info = data["cluster_info"]
        assert cluster_info["total_devices"] == 3
        assert cluster_info["healthy_devices"] == 3
        assert cluster_info["world_size"] == 3
        assert cluster_info["gRPC_communication"] == "Active"
        
        # Check aggregate hardware
        hardware = cluster_info["aggregate_hardware"]
        assert hardware["total_gpu_cores"] == 36  # 10 + 10 + 16
        assert hardware["total_cpu_cores"] == 32  # 10 + 10 + 12
        assert hardware["total_memory_gb"] == 80.0  # 16 + 16 + 48
        assert hardware["total_neural_engine_cores"] == 48  # 16 + 16 + 16
        
        # Check individual devices
        devices = data["devices"]
        assert len(devices) == 3
        
        # Check device types
        device_names = {d["device_id"]: d["hardware"]["device_name"] for d in devices}
        assert device_names["mini1"] == "Mac mini M4"
        assert device_names["mini2"] == "Mac mini M4"
        assert device_names["master"] == "MacBook Pro M4 Pro"
    
    def test_distributed_inference(self):
        """Test distributed inference across 3 devices."""
        request_data = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        
        # Check that we got a response
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        assert "4" in content or "four" in content.lower()
    
    def test_memory_distribution(self):
        """Test that memory is distributed across devices."""
        response = requests.get(f"{self.api_url}/distributed/gpu-info")
        data = response.json()
        
        devices = data["devices"]
        
        # Check memory allocation
        mini1_memory = devices[0]["system_memory"]["used_percent"]
        mini2_memory = devices[1]["system_memory"]["total_gb"]
        master_memory = devices[2]["system_memory"]["total_gb"]
        
        # Verify heterogeneous memory
        assert mini2_memory == 16.0
        assert master_memory == 48.0
        
        # Master device should handle larger model portions
        assert devices[2]["capabilities"]["max_recommended_model_size_gb"] > \
               devices[0]["capabilities"]["max_recommended_model_size_gb"]
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent inference requests."""
        import concurrent.futures
        
        def make_request(prompt):
            request_data = {
                "model": "mlx-community/Qwen3-1.7B-8bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 30,
                "temperature": 0.5
            }
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            return response.status_code, response.json() if response.status_code == 200 else None
        
        prompts = [
            "What is the capital of France?",
            "What is 10 times 10?",
            "Complete: The quick brown fox"
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, prompt) for prompt in prompts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, data in results:
            assert status_code == 200
            assert data is not None
            assert len(data["choices"][0]["message"]["content"]) > 0
    
    def test_device_failure_handling(self):
        """Test cluster behavior when a device fails."""
        # This test would simulate device failure
        # For now, we'll check that the cluster reports device status correctly
        
        response = requests.get(f"{self.api_url}/distributed/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "operational"
        assert data["world_size"] == 3
        
        # Check all devices are listed
        devices = data["devices"]
        device_ids = {d["device_id"] for d in devices}
        assert device_ids == {"mini1", "mini2", "master"}
    
    def test_model_info_endpoint(self):
        """Test the models endpoint."""
        response = requests.get(f"{self.api_url}/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "mlx-community/Qwen3-1.7B-8bit"
    
    def test_performance_metrics(self):
        """Test that performance metrics are available."""
        # Make an inference request first
        request_data = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data
        )
        inference_time = time.time() - start_time
        
        assert response.status_code == 200
        data = response.json()
        
        # Check usage stats
        assert "usage" in data
        assert "completion_tokens" in data["usage"]
        assert data["usage"]["completion_tokens"] > 0
        
        # Inference should be reasonably fast with distribution
        assert inference_time < 10.0  # Should complete within 10 seconds
    
    def test_grpc_communication(self):
        """Test that gRPC communication is working between devices."""
        # The GPU info endpoint will fail if gRPC isn't working
        response = requests.get(f"{self.api_url}/distributed/gpu-info")
        data = response.json()
        
        # If we can see all 3 devices, gRPC is working
        assert data["cluster_info"]["gRPC_communication"] == "Active"
        assert len(data["devices"]) == 3
        
        # Check that worker devices are identified correctly
        for device in data["devices"]:
            if device["device_id"] != "mini1":
                assert device["role"] == "worker"
                assert "Worker node" in device.get("note", "")
    
    def test_openai_compatibility(self):
        """Test OpenAI API compatibility."""
        # Test with OpenAI-style parameters
        request_data = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is MLX?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check OpenAI-compatible response format
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data
        
        # Check choice format
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        assert choice["message"]["role"] == "assistant"


class TestClusterPerformance:
    """Performance-specific tests for the cluster."""
    
    @classmethod
    def setup_class(cls):
        """Setup for performance tests."""
        cls.api_url = "http://localhost:8100"
    
    def test_tokens_per_second(self):
        """Measure tokens per second performance."""
        request_data = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [
                {"role": "user", "content": "Write a short story about a robot learning to paint. Make it exactly 100 words long."}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_data
        )
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        
        # Calculate tokens per second
        completion_tokens = data["usage"]["completion_tokens"]
        duration = end_time - start_time
        tokens_per_second = completion_tokens / duration
        
        print(f"\nPerformance: {tokens_per_second:.2f} tokens/second")
        print(f"Generated {completion_tokens} tokens in {duration:.2f} seconds")
        
        # With 3 devices, should achieve reasonable performance
        assert tokens_per_second > 5.0  # At least 5 tokens/second
    
    def test_memory_efficiency(self):
        """Test memory usage across devices."""
        response = requests.get(f"{self.api_url}/distributed/gpu-info")
        data = response.json()
        
        # Check that memory is being used efficiently
        master_device = next(d for d in data["devices"] if d["device_id"] == "mini1")
        memory_used = master_device["system_memory"]["used_percent"]
        
        print(f"\nMaster device memory usage: {memory_used}%")
        
        # Memory usage should be reasonable (not maxed out)
        assert memory_used < 90.0  # Less than 90% memory usage
    
    def test_scalability_metrics(self):
        """Test scalability with different prompt sizes."""
        prompt_sizes = [10, 50, 100, 200]  # Word counts
        results = []
        
        for size in prompt_sizes:
            prompt = " ".join(["word"] * size)
            request_data = {
                "model": "mlx-community/Qwen3-1.7B-8bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.5
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=request_data
            )
            duration = time.time() - start_time
            
            assert response.status_code == 200
            results.append((size, duration))
        
        print("\nScalability results:")
        for size, duration in results:
            print(f"  {size} words: {duration:.2f}s")
        
        # Response time should scale reasonably with input size
        # Not strictly linear due to parallelization benefits
        assert all(duration < 20.0 for _, duration in results)


if __name__ == "__main__":
    # Run specific test class or all tests
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        pytest.main([__file__, "-v", "-k", "TestClusterPerformance"])
    else:
        pytest.main([__file__, "-v"])
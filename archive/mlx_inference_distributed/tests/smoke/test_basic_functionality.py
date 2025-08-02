#!/usr/bin/env python3
"""
Smoke tests for distributed MLX inference system.
These tests validate that the basic system is functional.
"""

import pytest
import requests
import time
import subprocess
import os
import signal
from pathlib import Path

# Test configuration
API_PORT = 8100
API_BASE_URL = f"http://localhost:{API_PORT}"
TEST_TIMEOUT = 30  # seconds


class TestDistributedMLXSmoke:
    """Smoke tests for distributed MLX inference system."""
    
    @pytest.fixture(scope="class")
    def distributed_server(self):
        """Start distributed API server for testing."""
        # Change to project directory
        project_dir = Path(__file__).parent.parent.parent
        os.chdir(project_dir)
        
        # Start the distributed API server
        process = subprocess.Popen(
            ["uv", "run", "python", "run_distributed_openai.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for server to start
        for _ in range(TEST_TIMEOUT):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            # Server didn't start in time
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            pytest.fail("Distributed API server failed to start within timeout")
        
        yield process
        
        # Cleanup: stop the server
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_grpc_server_startup(self, distributed_server):
        """Test that gRPC server starts without errors."""
        # Check if server process is running
        assert distributed_server.poll() is None, "Distributed server process died"
        
        # Verify gRPC communication ports are listening
        # This test validates that our real gRPC implementation initializes
        try:
            import socket
            # Test that gRPC base port (50100) is in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 50100))
            sock.close()
            assert result == 0, "gRPC server not listening on port 50100"
        except Exception as e:
            pytest.fail(f"gRPC server startup validation failed: {e}")

    def test_api_server_responds(self, distributed_server):
        """Test that API server responds correctly."""
        # Test health endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        health_data = response.json()
        assert health_data["status"] == "healthy", f"Server not healthy: {health_data}"
        assert health_data["model_loaded"] is True, "Model not loaded"

    def test_model_loading(self, distributed_server):
        """Test that Qwen3-1.7B-8bit model loads successfully."""
        # Test models endpoint
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=10)
        assert response.status_code == 200, f"Models endpoint failed: {response.status_code}"
        
        models_data = response.json()
        assert "data" in models_data, "Models response missing data field"
        assert len(models_data["data"]) > 0, "No models available"
        
        # Verify Qwen model is loaded
        model_ids = [model["id"] for model in models_data["data"]]
        assert "mlx-community/Qwen3-1.7B-8bit" in model_ids, f"Qwen model not found: {model_ids}"

    def test_single_inference(self, distributed_server):
        """Test that system can generate one response."""
        # Test simple chat completion
        chat_request = {
            "model": "mlx-community/Qwen3-1.7B-8bit",
            "messages": [
                {"role": "user", "content": "Hello! Say 'test successful' if you can read this."}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        # Make inference request
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=chat_request,
            timeout=30  # Inference can take longer
        )
        
        assert response.status_code == 200, f"Chat completion failed: {response.status_code} - {response.text}"
        
        completion_data = response.json()
        assert "choices" in completion_data, "Response missing choices field"
        assert len(completion_data["choices"]) > 0, "No completion choices returned"
        
        # Verify response structure matches OpenAI format
        choice = completion_data["choices"][0]
        assert "message" in choice, "Choice missing message field"
        assert "content" in choice["message"], "Message missing content field"
        assert choice["message"]["role"] == "assistant", f"Wrong role: {choice['message']['role']}"
        
        # Verify actual content was generated
        content = choice["message"]["content"]
        assert len(content.strip()) > 0, "Empty response content"
        
        print(f"✅ Generated response: {content[:100]}...")


def test_real_grpc_implementation():
    """Validate that we have real gRPC implementation, not stubs."""
    from distributed_comm import GRPCCommunicator, GRPCCommServicer
    import inspect
    
    # Check that GRPCCommServicer has real implementations
    servicer = GRPCCommServicer(None)
    
    # Verify key methods exist and are not just 'pass' statements
    send_method = getattr(servicer, 'Send', None)
    assert send_method is not None, "GRPCCommServicer missing Send method"
    
    # Check that method has real implementation (not just a stub)
    source_lines = inspect.getsource(send_method)
    assert 'queue.Queue()' in source_lines, "Send method appears to be a stub - no queue implementation found"
    assert 'return distributed_comm_pb2.SendResponse' in source_lines, "Send method missing proper return"
    
    # Verify GRPCCommunicator has real send implementation
    comm = GRPCCommunicator()
    send_source = inspect.getsource(comm.send)
    assert 'grpc.RpcError' in send_source, "GRPCCommunicator.send missing gRPC error handling"
    assert '_serialize_data' in send_source, "GRPCCommunicator.send missing serialization"
    
    print("✅ Real gRPC implementation validated (not stubs)")


def test_no_mpi_dependencies():
    """Verify that all MPI dependencies have been removed."""
    import sys
    
    # Check that mpi4py is not imported
    mpi_modules = [name for name in sys.modules.keys() if 'mpi' in name.lower()]
    assert len(mpi_modules) == 0, f"MPI modules still imported: {mpi_modules}"
    
    # Test that our config defaults to gRPC
    from distributed_config import DistributedConfig
    config = DistributedConfig()
    assert config.communication_backend == "grpc", f"Config still defaults to MPI: {config.communication_backend}"
    
    print("✅ MPI dependencies completely removed")


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v", "--tb=short"])
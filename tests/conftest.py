"""
Test configuration and fixtures for distributed MLX inference system.
"""

import asyncio
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List
import mlx.core as mx

from src.core.config import ClusterConfig, DeviceConfig, DeviceRole, ModelConfig, PerformanceConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_model_name():
    """Name of test model for validation."""
    return "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_cluster_config():
    """Mock cluster configuration for testing."""
    devices = [
        DeviceConfig(
            device_id="coordinator",
            hostname="localhost",
            rank=0,
            role=DeviceRole.COORDINATOR,
            grpc_port=50051,
            api_port=8000
        ),
        DeviceConfig(
            device_id="worker_1",
            hostname="localhost",
            rank=1,
            role=DeviceRole.WORKER,
            grpc_port=50052,
            api_port=8001
        ),
        DeviceConfig(
            device_id="worker_2",
            hostname="localhost",
            rank=2,
            role=DeviceRole.WORKER,
            grpc_port=50053,
            api_port=8002
        )
    ]
    
    model_config = ModelConfig(
        name="test_model",
        path="test/path",
        layer_distribution={
            "coordinator": [0, 1, 2],
            "worker_1": [3, 4, 5],
            "worker_2": [6, 7, 8]
        },
        max_sequence_length=2048,
        context_window=4096
    )
    
    performance_config = PerformanceConfig(
        request_timeout_seconds=30.0,
        max_concurrent_requests=10,
        tensor_compression=True,
        memory_limit_gb=8.0
    )
    
    return ClusterConfig(
        devices=devices,
        model=model_config,
        performance=performance_config
    )


@pytest.fixture
def cluster_config_file(mock_cluster_config, temp_dir):
    """Create temporary cluster configuration file."""
    config_file = temp_dir / "cluster_config.yaml"
    config_dict = {
        "devices": [
            {
                "device_id": device.device_id,
                "hostname": device.hostname,
                "rank": device.rank,
                "role": device.role.value,
                "grpc_port": device.grpc_port,
                "api_port": device.api_port
            } for device in mock_cluster_config.devices
        ],
        "model": {
            "name": mock_cluster_config.model.name,
            "path": mock_cluster_config.model.path,
            "layer_distribution": mock_cluster_config.model.layer_distribution,
            "max_sequence_length": mock_cluster_config.model.max_sequence_length,
            "context_window": mock_cluster_config.model.context_window
        },
        "performance": {
            "request_timeout_seconds": mock_cluster_config.performance.request_timeout_seconds,
            "max_concurrent_requests": mock_cluster_config.performance.max_concurrent_requests,
            "tensor_compression": mock_cluster_config.performance.tensor_compression,
            "memory_limit_gb": mock_cluster_config.performance.memory_limit_gb
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    return config_file


@pytest.fixture
def mock_mlx_model():
    """Create mock MLX model for testing."""
    model = MagicMock()
    
    # Mock model structure
    model.model = MagicMock()
    model.model.embed_tokens = MagicMock()
    model.model.layers = []
    model.model.norm = MagicMock()
    model.lm_head = MagicMock()
    
    # Create mock layers
    for i in range(9):
        layer = MagicMock()
        layer.input_layernorm = MagicMock()
        layer.self_attn = MagicMock()
        layer.post_attention_layernorm = MagicMock()
        layer.mlp = MagicMock()
        model.model.layers.append(layer)
    
    # Mock embeddings return tensor
    model.model.embed_tokens.return_value = mx.ones((1, 10, 512))
    
    # Mock layer outputs
    for layer in model.model.layers:
        layer.input_layernorm.return_value = mx.ones((1, 10, 512))
        layer.self_attn.return_value = mx.ones((1, 10, 512))
        layer.post_attention_layernorm.return_value = mx.ones((1, 10, 512))
        layer.mlp.return_value = mx.ones((1, 10, 512))
    
    # Mock final layers
    model.model.norm.return_value = mx.ones((1, 10, 512))
    model.lm_head.return_value = mx.ones((1, 10, 32000))  # vocab size
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
    tokenizer.decode.return_value = "mocked response"
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def sample_input_tensor():
    """Create sample input tensor for testing."""
    return mx.array([[1, 2, 3, 4, 5]])  # Shape: [batch_size, sequence_length]


@pytest.fixture
def sample_hidden_states():
    """Create sample hidden states tensor for testing."""
    return mx.ones((1, 5, 512))  # Shape: [batch_size, sequence_length, hidden_size]


@pytest.fixture
def mock_grpc_client():
    """Create mock gRPC client for testing."""
    client = AsyncMock()
    
    # Mock health check
    client.health_check.return_value = {
        'healthy': True,
        'device_id': 'test_device',
        'timestamp': '2024-01-01T00:00:00Z'
    }
    
    # Mock device info
    client.get_device_info.return_value = {
        'device_id': 'test_device',
        'hostname': 'localhost',
        'rank': 1,
        'role': 'worker',
        'assigned_layers': [3, 4, 5],
        'capabilities': {'memory_gb': 8.0},
        'gpu_utilization': 0.5,
        'memory_usage_gb': 2.0
    }
    
    return client


@pytest.fixture
def mock_connection_pool():
    """Create mock connection pool for testing."""
    pool = MagicMock()
    
    # Mock client retrieval
    mock_client = mock_grpc_client()
    pool.get_client.return_value = mock_client
    pool.get_next_device_client.return_value = mock_client
    
    return pool


@pytest.fixture
def test_inference_request():
    """Create test inference request."""
    from src.coordinator.orchestrator import InferenceRequest
    
    return InferenceRequest(
        request_id="test_request_123",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "tokens_per_second": 25.5,
        "latency_ms": 150.0,
        "memory_usage_gb": 2.5,
        "gpu_utilization": 0.75,
        "throughput_requests_per_minute": 40
    }


@pytest.fixture(autouse=True)
def cleanup_mlx():
    """Cleanup MLX state after each test."""
    yield
    # Clear any MLX cached data
    try:
        mx.metal.clear_cache()
    except:
        pass


# Helper functions for test utilities
def create_test_tensor(shape: tuple, dtype=mx.float32) -> mx.array:
    """Create test tensor with specified shape."""
    return mx.random.normal(shape).astype(dtype)


def assert_tensor_shape(tensor: mx.array, expected_shape: tuple):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_dtype(tensor: mx.array, expected_dtype):
    """Assert tensor has expected dtype."""
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"


# Test data generators
def generate_test_messages(count: int = 1) -> List[Dict[str, str]]:
    """Generate test message sequences."""
    base_messages = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}],
        [{"role": "user", "content": "Write a short poem about AI"}],
        [{"role": "user", "content": "Explain quantum computing in simple terms"}]
    ]
    
    if count <= len(base_messages):
        return base_messages[:count]
    
    # Generate additional messages if needed
    result = base_messages.copy()
    for i in range(count - len(base_messages)):
        result.append([{"role": "user", "content": f"Test message {i+1}"}])
    
    return result
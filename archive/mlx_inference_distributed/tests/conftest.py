"""
Test configuration for distributed MLX inference system.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_model_path():
    """Path to test model for validation."""
    return "mlx-community/Qwen3-1.7B-8bit"

@pytest.fixture(scope="session") 
def cluster_config():
    """Standard cluster configuration for tests."""
    return {
        "master_hostname": "localhost",
        "master_port": 8100,
        "model_name": "mlx-community/Qwen3-1.7B-8bit",
        "world_size": 2,
        "communication_backend": "grpc"
    }

@pytest.fixture(scope="session")
def project_root():
    """Path to project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def temp_config_file(cluster_config):
    """Create temporary config file for testing."""
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cluster_config, f, indent=2)
        return f.name
"""
Training MLX Adapters - Integration layer for external systems
"""

from .distributed_integration import (
    get_integration_status,
    is_distributed_available,
    create_inference_adapter
)

__all__ = [
    "get_integration_status",
    "is_distributed_available", 
    "create_inference_adapter"
]
"""
Training MLX - Distributed Training System for MLX Models

A clean, production-ready implementation for distributed training with MLX.
Supports supervised fine-tuning, RLHF, and efficient training techniques.
"""

__version__ = "0.1.0"
__author__ = "MLX Training Team"

# Core components
from . import api
from . import cli
from . import training
from . import adapters
from . import security

__all__ = [
    "api",
    "cli", 
    "training",
    "adapters",
    "security",
    "__version__",
]
"""
Generation module for MLX Distributed Inference

Provides optimized text generation with KV-caching and distributed processing.
"""

from .optimized_generator import OptimizedGenerator

__all__ = ["OptimizedGenerator"]
"""Utilities for heterogeneous distributed inference."""

# Import only if available
try:
    from .logging_config import DiagnosticLogger, setup_logging, log_performance
    __all__ = ['DiagnosticLogger', 'setup_logging', 'log_performance']
except ImportError:
    # MLX not available, skip logging imports
    __all__ = []
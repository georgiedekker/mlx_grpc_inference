"""
File-based coordination system for distributed inference
"""

from .file_based_coordinator import (
    FileBasedCoordinator,
    DistributedInferenceManager,
    NodeInfo
)

__all__ = [
    'FileBasedCoordinator',
    'DistributedInferenceManager', 
    'NodeInfo'
]
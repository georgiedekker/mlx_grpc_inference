"""
Management module for MLX Distributed Inference

Provides device management, coordinator migration, and cluster orchestration.
"""

from .device_manager import DeviceManager, DeviceStatus

__all__ = ["DeviceManager", "DeviceStatus"]
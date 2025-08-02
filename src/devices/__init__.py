"""
Device abstraction layer for distributed MLX inference.

This module provides device abstractions that separate device-specific logic
from the orchestrator and enable clean device lifecycle management.
"""

from .base_device import BaseDevice, DeviceState, DeviceHealth
from .coordinator_device import CoordinatorDevice
from .worker_device import WorkerDevice
from .device_manager import DeviceManager
from .device_factory import DeviceFactory

__all__ = [
    "BaseDevice",
    "DeviceState", 
    "DeviceHealth",
    "CoordinatorDevice",
    "WorkerDevice",
    "DeviceManager",
    "DeviceFactory",
]
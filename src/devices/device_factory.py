"""
Device factory for creating distributed MLX inference devices.

This module provides a factory pattern for creating appropriate device instances
based on configuration and role specifications.
"""

import logging
from typing import Type, Dict, Any

from .base_device import BaseDevice
from .coordinator_device import CoordinatorDevice
from .worker_device import WorkerDevice
from ..core.config import ClusterConfig, DeviceConfig, DeviceRole

logger = logging.getLogger(__name__)


class DeviceFactory:
    """
    Factory class for creating device instances.
    
    The DeviceFactory provides a centralized way to create device instances
    based on their configuration and role, ensuring proper device types
    are instantiated with correct parameters.
    """
    
    def __init__(self, config: ClusterConfig):
        """
        Initialize the device factory.
        
        Args:
            config: Cluster configuration
        """
        self.config = config
        
        # Register device types
        self._device_types: Dict[DeviceRole, Type[BaseDevice]] = {
            DeviceRole.COORDINATOR: CoordinatorDevice,
            DeviceRole.WORKER: WorkerDevice,
        }
        
        logger.info(f"Initialized device factory for cluster: {config.name}")
    
    def create_device(self, device_config: DeviceConfig) -> BaseDevice:
        """
        Create a device instance based on the device configuration.
        
        Args:
            device_config: Configuration for the device to create
            
        Returns:
            Device instance of the appropriate type
            
        Raises:
            ValueError: If the device role is not supported
            TypeError: If device creation fails
        """
        try:
            # Get the device class for this role
            device_class = self._device_types.get(device_config.role)
            
            if not device_class:
                supported_roles = list(self._device_types.keys())
                raise ValueError(
                    f"Unsupported device role: {device_config.role}. "
                    f"Supported roles: {[role.value for role in supported_roles]}"
                )
            
            # Validate device configuration
            self._validate_device_config(device_config)
            
            # Create device instance
            device = device_class(self.config, device_config)
            
            logger.info(
                f"Created {device_config.role.value} device: {device_config.device_id} "
                f"({device_class.__name__})"
            )
            
            return device
            
        except Exception as e:
            logger.error(f"Failed to create device {device_config.device_id}: {e}")
            raise
    
    def create_coordinator(self, device_config: DeviceConfig) -> CoordinatorDevice:
        """
        Create a coordinator device instance.
        
        Args:
            device_config: Configuration for the coordinator device
            
        Returns:
            CoordinatorDevice instance
            
        Raises:
            ValueError: If the device is not configured as coordinator
        """
        if device_config.role != DeviceRole.COORDINATOR:
            raise ValueError(f"Device {device_config.device_id} is not configured as coordinator")
        
        device = self.create_device(device_config)
        return device  # Type checker knows this is CoordinatorDevice due to role check
    
    def create_worker(self, device_config: DeviceConfig) -> WorkerDevice:
        """
        Create a worker device instance.
        
        Args:
            device_config: Configuration for the worker device
            
        Returns:
            WorkerDevice instance
            
        Raises:
            ValueError: If the device is not configured as worker
        """
        if device_config.role != DeviceRole.WORKER:
            raise ValueError(f"Device {device_config.device_id} is not configured as worker")
        
        device = self.create_device(device_config)
        return device  # Type checker knows this is WorkerDevice due to role check
    
    def create_all_devices(self) -> Dict[str, BaseDevice]:
        """
        Create all devices from the cluster configuration.
        
        Returns:
            Dictionary mapping device IDs to device instances
        """
        devices = {}
        
        logger.info(f"Creating all devices for cluster: {self.config.name}")
        
        for device_config in self.config.devices:
            try:
                device = self.create_device(device_config)
                devices[device_config.device_id] = device
                
            except Exception as e:
                logger.error(f"Failed to create device {device_config.device_id}: {e}")
                # Clean up already created devices
                for created_device in devices.values():
                    try:
                        # Note: devices should implement a synchronous cleanup method
                        # For now, we just clear the reference
                        pass
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up device during factory failure: {cleanup_error}")
                
                devices.clear()
                raise
        
        logger.info(f"Created {len(devices)} devices successfully")
        return devices
    
    def register_device_type(self, role: DeviceRole, device_class: Type[BaseDevice]) -> None:
        """
        Register a custom device type for a specific role.
        
        This allows extending the factory with custom device implementations.
        
        Args:
            role: Device role to register
            device_class: Device class to use for this role
        """
        if not issubclass(device_class, BaseDevice):
            raise TypeError(f"Device class must inherit from BaseDevice: {device_class}")
        
        self._device_types[role] = device_class
        logger.info(f"Registered custom device type {device_class.__name__} for role {role.value}")
    
    def get_supported_roles(self) -> list[DeviceRole]:
        """Get list of supported device roles."""
        return list(self._device_types.keys())
    
    def _validate_device_config(self, device_config: DeviceConfig) -> None:
        """
        Validate device configuration before creation.
        
        Args:
            device_config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation
        if not device_config.device_id:
            raise ValueError("Device ID cannot be empty")
        
        if not device_config.hostname:
            raise ValueError("Device hostname cannot be empty")
        
        if device_config.api_port <= 0 or device_config.api_port > 65535:
            raise ValueError(f"Invalid API port: {device_config.api_port}")
        
        if device_config.grpc_port <= 0 or device_config.grpc_port > 65535:
            raise ValueError(f"Invalid gRPC port: {device_config.grpc_port}")
        
        if device_config.rank < 0:
            raise ValueError(f"Invalid device rank: {device_config.rank}")
        
        # Role-specific validation
        if device_config.role == DeviceRole.COORDINATOR:
            self._validate_coordinator_config(device_config)
        elif device_config.role == DeviceRole.WORKER:
            self._validate_worker_config(device_config)
    
    def _validate_coordinator_config(self, device_config: DeviceConfig) -> None:
        """
        Validate coordinator-specific configuration.
        
        Args:
            device_config: Coordinator configuration to validate
        """
        # Check if this is the designated coordinator
        if device_config.device_id != self.config.coordinator_device_id:
            raise ValueError(
                f"Device {device_config.device_id} is configured as coordinator "
                f"but cluster coordinator is {self.config.coordinator_device_id}"
            )
        
        # Coordinator should have rank 0
        if device_config.rank != 0:
            logger.warning(f"Coordinator {device_config.device_id} has rank {device_config.rank}, expected 0")
    
    def _validate_worker_config(self, device_config: DeviceConfig) -> None:
        """
        Validate worker-specific configuration.
        
        Args:
            device_config: Worker configuration to validate
        """
        # Worker should have positive rank
        if device_config.rank <= 0:
            raise ValueError(f"Worker {device_config.device_id} must have positive rank, got {device_config.rank}")
        
        # Check if worker has assigned layers
        assigned_layers = self.config.model.get_device_layers(device_config.device_id)
        if not assigned_layers:
            logger.warning(f"Worker {device_config.device_id} has no assigned layers")
        
        # Validate layer assignments
        total_layers = self.config.model.total_layers
        for layer_idx in assigned_layers:
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(
                    f"Worker {device_config.device_id} assigned invalid layer {layer_idx}. "
                    f"Valid range: 0-{total_layers-1}"
                )


# Convenience functions for direct device creation

def create_coordinator_device(config: ClusterConfig, device_config: DeviceConfig) -> CoordinatorDevice:
    """
    Convenience function to create a coordinator device.
    
    Args:
        config: Cluster configuration
        device_config: Coordinator device configuration
        
    Returns:
        CoordinatorDevice instance
    """
    factory = DeviceFactory(config)
    return factory.create_coordinator(device_config)


def create_worker_device(config: ClusterConfig, device_config: DeviceConfig) -> WorkerDevice:
    """
    Convenience function to create a worker device.
    
    Args:
        config: Cluster configuration
        device_config: Worker device configuration
        
    Returns:
        WorkerDevice instance
    """
    factory = DeviceFactory(config)
    return factory.create_worker(device_config)


def create_local_device(config: ClusterConfig) -> BaseDevice:
    """
    Convenience function to create a device for the local machine.
    
    Args:
        config: Cluster configuration
        
    Returns:
        Device instance for the local machine
        
    Raises:
        ValueError: If local device configuration cannot be determined
    """
    local_device_config = config.get_local_device()
    if not local_device_config:
        raise ValueError("Cannot determine local device configuration")
    
    factory = DeviceFactory(config)
    return factory.create_device(local_device_config)
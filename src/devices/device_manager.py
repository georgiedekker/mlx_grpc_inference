"""
Device manager for distributed MLX inference cluster.

This module provides centralized device lifecycle management, health monitoring,
and coordination across the distributed inference cluster.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

from .base_device import BaseDevice, DeviceState, DeviceHealth
from .coordinator_device import CoordinatorDevice
from .worker_device import WorkerDevice
from ..core.config import ClusterConfig, DeviceConfig, DeviceRole

logger = logging.getLogger(__name__)


class ClusterState(Enum):
    """State of the entire cluster."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ClusterHealth:
    """Health status of the entire cluster."""
    state: ClusterState
    total_devices: int
    healthy_devices: int
    coordinator_healthy: bool
    worker_health_ratio: float
    last_health_check: float
    error_message: Optional[str] = None


class DeviceManager:
    """
    Manages the lifecycle and health of all devices in the cluster.
    
    The DeviceManager is responsible for:
    - Creating and initializing devices
    - Monitoring device health and performance
    - Handling device failures and recovery
    - Coordinating cluster-wide operations
    - Managing device state transitions
    """
    
    def __init__(self, config: ClusterConfig):
        """
        Initialize the device manager.
        
        Args:
            config: Cluster configuration
        """
        self.config = config
        self.devices: Dict[str, BaseDevice] = {}
        
        # Cluster state
        self._cluster_state = ClusterState.UNINITIALIZED
        self._state_lock = asyncio.Lock()
        
        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._last_cluster_health_check = 0.0
        
        # Device tracking
        self._failed_devices: Set[str] = set()
        self._recovery_attempts: Dict[str, int] = {}
        
        logger.info(f"Initialized device manager for cluster: {config.name}")
    
    @property
    def cluster_state(self) -> ClusterState:
        """Get the current cluster state."""
        return self._cluster_state
    
    @property
    def is_ready(self) -> bool:
        """Check if the cluster is ready to process requests."""
        return self._cluster_state == ClusterState.READY
    
    async def initialize_cluster(self) -> None:
        """
        Initialize all devices in the cluster.
        
        This method will create and initialize all configured devices,
        ensuring the cluster is ready for operation.
        """
        async with self._state_lock:
            if self._cluster_state != ClusterState.UNINITIALIZED:
                logger.warning("Cluster already initialized or initializing")
                return
            
            self._cluster_state = ClusterState.INITIALIZING
        
        try:
            logger.info("Initializing cluster devices...")
            
            # Create all devices
            await self._create_devices()
            
            # Initialize coordinator first
            await self._initialize_coordinator()
            
            # Initialize workers
            await self._initialize_workers()
            
            # Verify cluster health
            cluster_health = await self.check_cluster_health()
            if not self._is_cluster_healthy(cluster_health):
                raise RuntimeError("Cluster health check failed after initialization")
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            async with self._state_lock:
                self._cluster_state = ClusterState.READY
            
            logger.info("Cluster initialization completed successfully")
            
        except Exception as e:
            async with self._state_lock:
                self._cluster_state = ClusterState.ERROR
            
            logger.error(f"Failed to initialize cluster: {e}")
            await self._cleanup_devices()
            raise
    
    async def shutdown_cluster(self) -> None:
        """
        Gracefully shutdown all devices in the cluster.
        
        This method will stop health monitoring and shutdown all devices
        in the correct order.
        """
        async with self._state_lock:
            if self._cluster_state in {ClusterState.SHUTTING_DOWN, ClusterState.SHUTDOWN}:
                logger.warning("Cluster already shutting down or shutdown")
                return
            
            self._cluster_state = ClusterState.SHUTTING_DOWN
        
        try:
            logger.info("Shutting down cluster...")
            
            # Stop health monitoring
            await self._stop_health_monitoring()
            
            # Shutdown devices in reverse order (workers first, then coordinator)
            await self._shutdown_workers()
            await self._shutdown_coordinator()
            
            # Cleanup all devices
            await self._cleanup_devices()
            
            async with self._state_lock:
                self._cluster_state = ClusterState.SHUTDOWN
            
            logger.info("Cluster shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")
            raise
    
    async def check_cluster_health(self) -> ClusterHealth:
        """
        Check the health of the entire cluster.
        
        Returns:
            ClusterHealth object containing cluster health information
        """
        try:
            healthy_devices = 0
            coordinator_healthy = False
            total_devices = len(self.devices)
            
            # Check each device
            for device_id, device in self.devices.items():
                try:
                    health = await device.check_health()
                    if health.is_healthy:
                        healthy_devices += 1
                        
                        if device.role == DeviceRole.COORDINATOR:
                            coordinator_healthy = True
                
                except Exception as e:
                    logger.warning(f"Health check failed for device {device_id}: {e}")
            
            # Calculate worker health ratio
            workers = [d for d in self.devices.values() if d.role == DeviceRole.WORKER]
            healthy_workers = sum(1 for d in workers if d.is_healthy)
            worker_health_ratio = healthy_workers / len(workers) if workers else 1.0
            
            # Determine cluster state
            if not coordinator_healthy:
                cluster_state = ClusterState.ERROR
                error_message = "Coordinator is not healthy"
            elif worker_health_ratio < 0.5:
                cluster_state = ClusterState.DEGRADED
                error_message = f"Only {worker_health_ratio:.1%} of workers are healthy"
            elif healthy_devices == total_devices:
                cluster_state = ClusterState.READY
                error_message = None
            else:
                cluster_state = ClusterState.DEGRADED
                error_message = f"{total_devices - healthy_devices} devices are unhealthy"
            
            self._last_cluster_health_check = time.time()
            
            return ClusterHealth(
                state=cluster_state,
                total_devices=total_devices,
                healthy_devices=healthy_devices,
                coordinator_healthy=coordinator_healthy,
                worker_health_ratio=worker_health_ratio,
                last_health_check=self._last_cluster_health_check,
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Error checking cluster health: {e}")
            return ClusterHealth(
                state=ClusterState.ERROR,
                total_devices=len(self.devices),
                healthy_devices=0,
                coordinator_healthy=False,
                worker_health_ratio=0.0,
                last_health_check=time.time(),
                error_message=str(e)
            )
    
    async def get_device_health(self, device_id: str) -> Optional[DeviceHealth]:
        """
        Get health information for a specific device.
        
        Args:
            device_id: ID of the device
            
        Returns:
            DeviceHealth object or None if device not found
        """
        device = self.devices.get(device_id)
        if not device:
            return None
        
        try:
            return await device.check_health()
        except Exception as e:
            logger.error(f"Error checking health for device {device_id}: {e}")
            return None
    
    def get_device(self, device_id: str) -> Optional[BaseDevice]:
        """Get a device by its ID."""
        return self.devices.get(device_id)
    
    def get_coordinator(self) -> Optional[CoordinatorDevice]:
        """Get the coordinator device."""
        for device in self.devices.values():
            if isinstance(device, CoordinatorDevice):
                return device
        return None
    
    def get_workers(self) -> List[WorkerDevice]:
        """Get all worker devices."""
        return [device for device in self.devices.values() if isinstance(device, WorkerDevice)]
    
    def get_healthy_workers(self) -> List[WorkerDevice]:
        """Get all healthy worker devices."""
        return [device for device in self.get_workers() if device.is_healthy]
    
    async def _create_devices(self) -> None:
        """Create all devices from configuration."""
        from .device_factory import DeviceFactory
        
        logger.info("Creating devices from configuration...")
        
        device_factory = DeviceFactory(self.config)
        
        for device_config in self.config.devices:
            try:
                device = device_factory.create_device(device_config)
                self.devices[device_config.device_id] = device
                logger.info(f"Created {device_config.role.value} device: {device_config.device_id}")
                
            except Exception as e:
                logger.error(f"Failed to create device {device_config.device_id}: {e}")
                raise
    
    async def _initialize_coordinator(self) -> None:
        """Initialize the coordinator device."""
        coordinator = self.get_coordinator()
        if not coordinator:
            raise RuntimeError("No coordinator device found")
        
        logger.info("Initializing coordinator device...")
        await coordinator.initialize()
        logger.info("Coordinator device initialized")
    
    async def _initialize_workers(self) -> None:
        """Initialize all worker devices."""
        workers = self.get_workers()
        if not workers:
            logger.warning("No worker devices to initialize")
            return
        
        logger.info(f"Initializing {len(workers)} worker devices...")
        
        # Initialize workers in parallel
        initialization_tasks = []
        for worker in workers:
            task = asyncio.create_task(worker.initialize())
            initialization_tasks.append(task)
        
        # Wait for all workers to initialize
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Check for failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                worker_id = workers[i].device_id
                logger.error(f"Failed to initialize worker {worker_id}: {result}")
                self._failed_devices.add(worker_id)
        
        successful_workers = len(workers) - len([r for r in results if isinstance(r, Exception)])
        logger.info(f"Initialized {successful_workers}/{len(workers)} worker devices")
    
    async def _shutdown_coordinator(self) -> None:
        """Shutdown the coordinator device."""
        coordinator = self.get_coordinator()
        if coordinator:
            logger.info("Shutting down coordinator device...")
            try:
                await coordinator.shutdown()
                logger.info("Coordinator device shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down coordinator: {e}")
    
    async def _shutdown_workers(self) -> None:
        """Shutdown all worker devices."""
        workers = self.get_workers()
        if not workers:
            return
        
        logger.info(f"Shutting down {len(workers)} worker devices...")
        
        # Shutdown workers in parallel
        shutdown_tasks = []
        for worker in workers:
            task = asyncio.create_task(worker.shutdown())
            shutdown_tasks.append(task)
        
        # Wait for all workers to shutdown
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        logger.info("Worker devices shutdown completed")
    
    async def _cleanup_devices(self) -> None:
        """Cleanup all device references."""
        logger.info("Cleaning up devices...")
        self.devices.clear()
        self._failed_devices.clear()
        self._recovery_attempts.clear()
    
    async def _start_health_monitoring(self) -> None:
        """Start the health monitoring task."""
        if self._health_monitor_task:
            logger.warning("Health monitoring already started")
            return
        
        logger.info("Starting health monitoring...")
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
    
    async def _stop_health_monitoring(self) -> None:
        """Stop the health monitoring task."""
        if self._health_monitor_task:
            logger.info("Stopping health monitoring...")
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        logger.info("Health monitoring started")
        
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                
                try:
                    cluster_health = await self.check_cluster_health()
                    
                    # Update cluster state based on health
                    async with self._state_lock:
                        if cluster_health.state != self._cluster_state:
                            logger.info(f"Cluster state changed: {self._cluster_state} -> {cluster_health.state}")
                            self._cluster_state = cluster_health.state
                    
                    # Handle device failures
                    await self._handle_device_failures()
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
                
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
            raise
    
    async def _handle_device_failures(self) -> None:
        """Handle device failures and attempt recovery."""
        for device_id, device in self.devices.items():
            if not device.is_healthy and device_id not in self._failed_devices:
                logger.warning(f"Device {device_id} became unhealthy")
                self._failed_devices.add(device_id)
                
                # Attempt recovery for workers
                if device.role == DeviceRole.WORKER:
                    await self._attempt_device_recovery(device_id)
    
    async def _attempt_device_recovery(self, device_id: str) -> None:
        """Attempt to recover a failed device."""
        max_recovery_attempts = 3
        
        recovery_count = self._recovery_attempts.get(device_id, 0)
        if recovery_count >= max_recovery_attempts:
            logger.error(f"Device {device_id} exceeded maximum recovery attempts")
            return
        
        logger.info(f"Attempting recovery for device {device_id} (attempt {recovery_count + 1})")
        
        try:
            device = self.devices.get(device_id)
            if not device:
                return
            
            # Try to reinitialize the device
            await device.shutdown()
            await asyncio.sleep(5.0)  # Wait before retry
            await device.initialize()
            
            # Check if recovery was successful
            health = await device.check_health()
            if health.is_healthy:
                logger.info(f"Device {device_id} recovery successful")
                self._failed_devices.discard(device_id)
                self._recovery_attempts.pop(device_id, None)
            else:
                raise RuntimeError("Device still unhealthy after recovery")
                
        except Exception as e:
            self._recovery_attempts[device_id] = recovery_count + 1
            logger.error(f"Device {device_id} recovery failed: {e}")
    
    def _is_cluster_healthy(self, cluster_health: ClusterHealth) -> bool:
        """Check if cluster health is acceptable."""
        return (
            cluster_health.coordinator_healthy and
            cluster_health.worker_health_ratio >= 0.5  # At least 50% of workers healthy
        )
"""
Device coordination for managing distributed cluster health and communication.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..core.config import ClusterConfig, DeviceRole
from ..communication.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Device status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"


@dataclass
class DeviceInfo:
    """Information about a device in the cluster."""
    device_id: str
    status: DeviceStatus
    last_health_check: float
    response_time_ms: float
    memory_usage: Dict[str, float]
    layer_assignments: List[int]
    error_message: Optional[str] = None


class DeviceCoordinator:
    """Coordinates device health, connections, and communication in the cluster."""
    
    def __init__(self, 
                 config: ClusterConfig,
                 connection_pool: ConnectionPool,
                 health_check_interval: float = 30.0):
        """
        Initialize device coordinator.
        
        Args:
            config: Cluster configuration
            connection_pool: Pool of gRPC connections
            health_check_interval: Interval between health checks in seconds
        """
        self.config = config
        self.connection_pool = connection_pool
        self.health_check_interval = health_check_interval
        
        self.device_info: Dict[str, DeviceInfo] = {}
        self.local_device = config.get_local_device()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Initialize device info
        self._initialize_device_info()
        
        logger.info(f"DeviceCoordinator initialized for {len(config.devices)} devices")
    
    def _initialize_device_info(self):
        """Initialize device information for all devices in cluster."""
        for device in self.config.devices:
            layer_assignments = self.config.model.get_device_layers(device.device_id)
            
            self.device_info[device.device_id] = DeviceInfo(
                device_id=device.device_id,
                status=DeviceStatus.UNKNOWN,
                last_health_check=0.0,
                response_time_ms=0.0,
                memory_usage={},
                layer_assignments=layer_assignments or []
            )
    
    async def start_monitoring(self):
        """Start device monitoring and health checks."""
        if self._health_check_task is not None:
            logger.warning("Device monitoring already started")
            return
        
        logger.info("Starting device monitoring")
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_monitoring(self):
        """Stop device monitoring."""
        if self._health_check_task is None:
            return
        
        logger.info("Stopping device monitoring")
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    async def _health_check_loop(self):
        """Continuous health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def perform_health_checks(self) -> Dict[str, DeviceInfo]:
        """
        Perform health checks on all devices.
        
        Returns:
            Updated device information
        """
        health_check_tasks = []
        
        # Create health check tasks for all remote devices
        for device_id in self.device_info.keys():
            if device_id != self.local_device.device_id:
                task = asyncio.create_task(
                    self._check_device_health(device_id)
                )
                health_check_tasks.append(task)
        
        # Wait for all health checks to complete
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
        
        # Update local device status
        self._update_local_device_status()
        
        # Log cluster health summary
        self._log_cluster_health()
        
        return self.device_info.copy()
    
    async def _check_device_health(self, device_id: str):
        """Check health of a specific device."""
        device_info = self.device_info[device_id]
        
        try:
            start_time = time.time()
            client = self.connection_pool.get_client(device_id)
            
            if not client:
                device_info.status = DeviceStatus.DISCONNECTED
                device_info.error_message = "No gRPC connection"
                return
            
            # Perform health check
            health_result = client.health_check()
            response_time = (time.time() - start_time) * 1000
            
            # Update device info based on health check result
            if health_result.get('healthy', False):
                device_info.status = DeviceStatus.HEALTHY
                device_info.response_time_ms = response_time
                device_info.memory_usage = health_result.get('memory_usage', {})
                device_info.error_message = None
            else:
                device_info.status = DeviceStatus.UNHEALTHY
                device_info.error_message = health_result.get('error', 'Unknown health issue')
            
            device_info.last_health_check = time.time()
            
        except Exception as e:
            device_info.status = DeviceStatus.UNHEALTHY
            device_info.error_message = str(e)
            device_info.last_health_check = time.time()
            logger.warning(f"Health check failed for {device_id}: {e}")
    
    def _update_local_device_status(self):
        """Update status for the local device."""
        local_device_id = self.local_device.device_id
        local_info = self.device_info[local_device_id]
        
        local_info.status = DeviceStatus.HEALTHY
        local_info.last_health_check = time.time()
        local_info.response_time_ms = 0.0  # Local device
        # TODO: Add local memory usage monitoring
        local_info.memory_usage = {}
    
    def _log_cluster_health(self):
        """Log summary of cluster health status."""
        healthy_count = sum(1 for info in self.device_info.values() 
                          if info.status == DeviceStatus.HEALTHY)
        total_count = len(self.device_info)
        
        logger.debug(f"Cluster health: {healthy_count}/{total_count} devices healthy")
        
        # Log details for unhealthy devices
        for device_id, info in self.device_info.items():
            if info.status != DeviceStatus.HEALTHY:
                logger.warning(f"Device {device_id} is {info.status.value}: {info.error_message}")
    
    def get_healthy_devices(self) -> List[str]:
        """Get list of healthy device IDs."""
        return [
            device_id for device_id, info in self.device_info.items()
            if info.status == DeviceStatus.HEALTHY
        ]
    
    def get_unhealthy_devices(self) -> List[str]:
        """Get list of unhealthy device IDs."""
        return [
            device_id for device_id, info in self.device_info.items()
            if info.status != DeviceStatus.HEALTHY
        ]
    
    def is_device_healthy(self, device_id: str) -> bool:
        """Check if a specific device is healthy."""
        info = self.device_info.get(device_id)
        return info is not None and info.status == DeviceStatus.HEALTHY
    
    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get information about a specific device."""
        return self.device_info.get(device_id)
    
    def get_cluster_status(self) -> Dict[str, any]:
        """Get comprehensive cluster status information."""
        healthy_devices = self.get_healthy_devices()
        unhealthy_devices = self.get_unhealthy_devices()
        
        # Calculate total layers across healthy devices
        healthy_layers = []
        for device_id in healthy_devices:
            info = self.device_info[device_id]
            healthy_layers.extend(info.layer_assignments)
        
        return {
            'total_devices': len(self.device_info),
            'healthy_devices': len(healthy_devices),
            'unhealthy_devices': len(unhealthy_devices),
            'healthy_device_ids': healthy_devices,
            'unhealthy_device_ids': unhealthy_devices,
            'total_layers_healthy': len(set(healthy_layers)),
            'cluster_operational': len(unhealthy_devices) == 0,
            'coordinator_device': self.local_device.device_id,
            'last_health_check': max(info.last_health_check for info in self.device_info.values())
        }
    
    async def reconnect_device(self, device_id: str) -> bool:
        """
        Attempt to reconnect to a specific device.
        
        Args:
            device_id: ID of device to reconnect
            
        Returns:
            True if reconnection successful
        """
        logger.info(f"Attempting to reconnect to device {device_id}")
        
        device_info = self.device_info.get(device_id)
        if not device_info:
            logger.error(f"Device {device_id} not found in cluster configuration")
            return False
        
        # Update status to connecting
        device_info.status = DeviceStatus.CONNECTING
        
        try:
            # Attempt reconnection through connection pool
            success = self.connection_pool.reconnect_device(device_id)
            
            if success:
                # Perform immediate health check
                await self._check_device_health(device_id)
                logger.info(f"Successfully reconnected to {device_id}")
                return device_info.status == DeviceStatus.HEALTHY
            else:
                device_info.status = DeviceStatus.DISCONNECTED
                device_info.error_message = "Reconnection failed"
                return False
                
        except Exception as e:
            device_info.status = DeviceStatus.UNHEALTHY
            device_info.error_message = f"Reconnection error: {e}"
            logger.error(f"Failed to reconnect to {device_id}: {e}")
            return False
    
    def validate_pipeline_health(self) -> Tuple[bool, List[str]]:
        """
        Validate that all devices required for the inference pipeline are healthy.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check that all devices are healthy
        unhealthy_devices = self.get_unhealthy_devices()
        if unhealthy_devices:
            issues.append(f"Unhealthy devices: {unhealthy_devices}")
        
        # Check that all layers are covered by healthy devices
        all_layers = set()
        covered_layers = set()
        
        for device_id, layers in self.config.model.layer_distribution.items():
            if layers:
                all_layers.update(layers)
                if self.is_device_healthy(device_id):
                    covered_layers.update(layers)
        
        missing_layers = all_layers - covered_layers
        if missing_layers:
            issues.append(f"Missing layers due to unhealthy devices: {sorted(missing_layers)}")
        
        # Check device connectivity
        for device_id in self.config.get_worker_device_ids():
            if not self.connection_pool.is_connected(device_id):
                issues.append(f"No connection to worker device: {device_id}")
        
        return len(issues) == 0, issues
    
    def get_device_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all devices."""
        metrics = {}
        
        for device_id, info in self.device_info.items():
            metrics[device_id] = {
                'response_time_ms': info.response_time_ms,
                'last_health_check_age_s': time.time() - info.last_health_check,
                'status_numeric': 1.0 if info.status == DeviceStatus.HEALTHY else 0.0,
                'assigned_layer_count': len(info.layer_assignments)
            }
            
            # Add memory metrics if available
            if info.memory_usage:
                metrics[device_id].update(info.memory_usage)
        
        return metrics
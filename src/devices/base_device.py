"""
Base device abstraction for distributed MLX inference.

This module defines the abstract base class for all devices in the cluster,
providing common interfaces for device lifecycle, health monitoring, and communication.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, List
import mlx.core as mx
import psutil

from ..core.config import ClusterConfig, DeviceConfig, DeviceRole
from ..model.loader import DistributedModelLoader
try:
    from ..model.inference_fixed import FixedLayerProcessor as LayerProcessor
except ImportError:
    from ..model.inference import LayerProcessor

logger = logging.getLogger(__name__)


class DeviceState(Enum):
    """State of a device in the cluster."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class DeviceHealth:
    """Health status of a device."""
    is_healthy: bool
    state: DeviceState
    uptime_seconds: float
    memory_usage_percent: float
    gpu_utilization_percent: float
    error_message: Optional[str] = None
    last_heartbeat: Optional[float] = None
    request_count: int = 0
    average_response_time_ms: float = 0.0


@dataclass
class DeviceMetrics:
    """Performance metrics for a device."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    last_request_time: Optional[float] = None


class BaseDevice(ABC):
    """
    Abstract base class for all devices in the distributed inference cluster.
    
    This class provides common functionality for device lifecycle management,
    health monitoring, resource management, and communication interfaces.
    """
    
    def __init__(self, config: ClusterConfig, device_config: DeviceConfig):
        """
        Initialize the base device.
        
        Args:
            config: Cluster configuration
            device_config: Configuration for this specific device
        """
        self.config = config
        self.device_config = device_config
        self.device_id = device_config.device_id
        self.role = device_config.role
        
        # Device state
        self._state = DeviceState.UNINITIALIZED
        self._start_time = time.time()
        self._error_message: Optional[str] = None
        self._metrics = DeviceMetrics()
        
        # Model components
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.layer_processor: Optional[LayerProcessor] = None
        
        # Synchronization
        self._state_lock = asyncio.Lock()
        
        logger.info(f"Initialized {self.role.value} device: {self.device_id}")
    
    @property
    def state(self) -> DeviceState:
        """Get the current device state."""
        return self._state
    
    @property
    def is_ready(self) -> bool:
        """Check if the device is ready to process requests."""
        return self._state == DeviceState.READY
    
    @property
    def is_healthy(self) -> bool:
        """Check if the device is healthy."""
        return self._state in {DeviceState.READY, DeviceState.BUSY, DeviceState.INITIALIZING}
    
    async def initialize(self) -> None:
        """
        Initialize the device and its components.
        
        This method should be called before the device can process requests.
        """
        async with self._state_lock:
            if self._state != DeviceState.UNINITIALIZED:
                logger.warning(f"Device {self.device_id} already initialized or initializing")
                return
            
            self._state = DeviceState.INITIALIZING
            
        try:
            logger.info(f"Initializing device {self.device_id}...")
            
            # Perform device-specific initialization
            await self._initialize_device()
            
            # Load model components
            await self._load_model()
            
            # Initialize communication
            await self._initialize_communication()
            
            # Perform health check
            health = await self.check_health()
            if not health.is_healthy:
                raise RuntimeError(f"Device {self.device_id} failed initial health check")
            
            async with self._state_lock:
                self._state = DeviceState.READY
                self._error_message = None
            
            logger.info(f"Device {self.device_id} initialized successfully")
            
        except Exception as e:
            async with self._state_lock:
                self._state = DeviceState.ERROR
                self._error_message = str(e)
            
            logger.error(f"Failed to initialize device {self.device_id}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the device.
        
        This method should be called to cleanup resources before termination.
        """
        async with self._state_lock:
            if self._state in {DeviceState.SHUTTING_DOWN, DeviceState.SHUTDOWN}:
                logger.warning(f"Device {self.device_id} already shutting down or shutdown")
                return
            
            self._state = DeviceState.SHUTTING_DOWN
        
        try:
            logger.info(f"Shutting down device {self.device_id}...")
            
            # Perform device-specific shutdown
            await self._shutdown_device()
            
            # Cleanup communication
            await self._cleanup_communication()
            
            # Cleanup model resources
            await self._cleanup_model()
            
            async with self._state_lock:
                self._state = DeviceState.SHUTDOWN
            
            logger.info(f"Device {self.device_id} shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown of device {self.device_id}: {e}")
            raise
    
    async def check_health(self) -> DeviceHealth:
        """
        Check the health status of the device.
        
        Returns:
            DeviceHealth object containing health information
        """
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Get GPU utilization (device-specific implementation)
            gpu_util = await self._get_gpu_utilization()
            
            # Calculate uptime
            uptime = time.time() - self._start_time
            
            # Calculate average response time
            avg_response_time = (
                self._metrics.average_processing_time_ms if self._metrics.total_requests > 0 else 0.0
            )
            
            return DeviceHealth(
                is_healthy=self.is_healthy,
                state=self._state,
                uptime_seconds=uptime,
                memory_usage_percent=memory_usage,
                gpu_utilization_percent=gpu_util,
                error_message=self._error_message,
                last_heartbeat=time.time(),
                request_count=self._metrics.total_requests,
                average_response_time_ms=avg_response_time
            )
            
        except Exception as e:
            logger.error(f"Error checking health for device {self.device_id}: {e}")
            return DeviceHealth(
                is_healthy=False,
                state=DeviceState.ERROR,
                uptime_seconds=time.time() - self._start_time,
                memory_usage_percent=0.0,
                gpu_utilization_percent=0.0,
                error_message=str(e),
                last_heartbeat=time.time()
            )
    
    def get_metrics(self) -> DeviceMetrics:
        """Get performance metrics for the device."""
        return self._metrics
    
    def record_request_start(self) -> float:
        """Record the start of a request and return timestamp."""
        start_time = time.time()
        self._metrics.last_request_time = start_time
        return start_time
    
    def record_request_completion(self, start_time: float, success: bool = True) -> None:
        """Record the completion of a request."""
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        self._metrics.total_requests += 1
        self._metrics.total_processing_time_ms += processing_time_ms
        
        if success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1
        
        # Update average processing time
        if self._metrics.total_requests > 0:
            self._metrics.average_processing_time_ms = (
                self._metrics.total_processing_time_ms / self._metrics.total_requests
            )
    
    async def _load_model(self) -> None:
        """Load model components for this device."""
        logger.info(f"Loading model for device {self.device_id}...")
        
        loader = DistributedModelLoader(self.config)
        
        # Load appropriate model shard based on device role
        if self.role == DeviceRole.COORDINATOR:
            self.model, self.tokenizer = loader.load_full_model()
        else:
            self.model, self.tokenizer = loader.load_model_shard(self.device_id)
        
        # Get assigned layers for this device
        assigned_layers = self.config.model.get_device_layers(self.device_id)
        
        # Create layer processor if we have a model and assigned layers
        if self.model and assigned_layers:
            self.layer_processor = LayerProcessor(
                self.model,
                self.device_id,
                assigned_layers
            )
            logger.info(f"Model loaded with layers: {assigned_layers}")
        else:
            logger.warning(f"No model or layers assigned to device {self.device_id}")
    
    async def _cleanup_model(self) -> None:
        """Cleanup model resources."""
        logger.info(f"Cleaning up model resources for device {self.device_id}")
        
        # Clear references to allow garbage collection
        self.layer_processor = None
        self.model = None
        self.tokenizer = None
        
        # Force MLX cleanup if available
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    
    async def _get_gpu_utilization(self) -> float:
        """
        Get GPU utilization percentage.
        
        Returns:
            GPU utilization as percentage (0-100)
        """
        try:
            # For MLX/Metal, we can check if Metal is available
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'is_available'):
                return 50.0 if mx.metal.is_available() else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _initialize_device(self) -> None:
        """Initialize device-specific components."""
        pass
    
    @abstractmethod
    async def _shutdown_device(self) -> None:
        """Shutdown device-specific components."""
        pass
    
    @abstractmethod
    async def _initialize_communication(self) -> None:
        """Initialize communication channels."""
        pass
    
    @abstractmethod
    async def _cleanup_communication(self) -> None:
        """Cleanup communication resources."""
        pass
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request specific to this device type.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            Response data dictionary
        """
        pass
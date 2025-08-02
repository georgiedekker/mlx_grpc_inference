"""
Orchestrator for distributed inference across multiple devices.
Refactored to use device abstraction layer.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

from ..core.config import ClusterConfig, DeviceRole
from ..devices import CoordinatorDevice, DeviceManager
from .request_handler import InferenceRequest, InferenceResponse
from ..monitoring.metrics import MetricsCollector
from ..monitoring.health import HealthMonitor, DistributedHealthChecker

logger = logging.getLogger(__name__)


class DistributedOrchestrator:
    """
    High-level orchestrator for distributed inference using device abstraction.
    
    Responsibilities:
    - Initialize and manage the device abstraction layer
    - Coordinate cluster operations through DeviceManager
    - Provide high-level interface for inference requests
    - Monitor cluster health and performance
    """
    
    def __init__(self, config: ClusterConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Cluster configuration
        """
        self.config = config
        self.device_config = config.get_local_device()
        
        if not self.device_config or self.device_config.role != DeviceRole.COORDINATOR:
            raise ValueError("Orchestrator must run on coordinator device")
        
        # Device abstraction layer
        self.device_manager = DeviceManager(config)
        self.coordinator_device: Optional[CoordinatorDevice] = None
        
        # System state
        self.is_initialized = False
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"Initializing orchestrator on {self.device_config.device_id} with device abstraction layer")
    
    async def initialize(self):
        """Initialize all orchestrator components using device abstraction."""
        if self.is_initialized:
            logger.warning("Orchestrator already initialized")
            return
        
        try:
            logger.info("Starting orchestrator initialization with device abstraction...")
            
            # Initialize the entire cluster through device manager
            await self.device_manager.initialize_cluster()
            
            # Get the coordinator device
            self.coordinator_device = self.device_manager.get_coordinator()
            if not self.coordinator_device:
                raise RuntimeError("Failed to get coordinator device from device manager")
            
            # Verify cluster health
            cluster_health = await self.device_manager.check_cluster_health()
            if not cluster_health.coordinator_healthy:
                raise RuntimeError(f"Cluster not healthy after initialization: {cluster_health.error_message}")
            
            self.is_initialized = True
            logger.info("Orchestrator initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.shutdown()
            raise
    
    
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request through the coordinator device.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coordinator_device:
            raise RuntimeError("Coordinator device not available")
        
        # Convert InferenceRequest to dict format expected by device
        request_data = {
            'request_id': request.request_id,
            'messages': request.messages,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'repetition_penalty': request.repetition_penalty
        }
        
        # Process through coordinator device
        response_data = await self.coordinator_device.process_request(request_data)
        
        # Convert response back to InferenceResponse
        return InferenceResponse(
            request_id=response_data['request_id'],
            content=response_data['content'],
            tokens_generated=response_data['tokens_generated'],
            total_time_ms=response_data['total_time_ms'],
            device_times=response_data['device_times'],
            generation_speed=response_data['tokens_generated'] / (response_data['total_time_ms'] / 1000) if response_data['total_time_ms'] > 0 else 0.0,
            stop_reason='completed'
        )
    
    async def process_streaming_request(self, request: InferenceRequest):
        """
        Process a streaming inference request.
        
        Note: Streaming is not yet implemented in the device abstraction layer.
        This will be added in a future iteration.
        
        Args:
            request: Inference request with stream=True
            
        Yields:
            Streaming tokens and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        # For now, fall back to non-streaming and yield the complete result
        response = await self.process_request(request)
        yield {
            'token': response.content,
            'done': True,
            'metadata': {
                'tokens_generated': response.tokens_generated,
                'total_time_ms': response.total_time_ms,
                'device_times': response.device_times
            }
        }
    
    def create_request(self, 
                      messages: List[Dict[str, str]],
                      max_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 1.0,
                      repetition_penalty: float = 1.1,
                      stream: bool = False,
                      stop_sequences: Optional[List[str]] = None) -> InferenceRequest:
        """
        Create a new inference request.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            stream: Whether to stream response
            stop_sequences: Stop sequences for early termination
            
        Returns:
            Configured inference request
        """
        return InferenceRequest(
            request_id=f"req_{int(time.time() * 1000000)}",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
            stop_sequences=stop_sequences
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status through device abstraction.
        
        Returns:
            System status information
        """
        if not self.is_initialized:
            return {
                'initialized': False,
                'device_id': self.device_config.device_id,
                'status': 'not_initialized'
            }
        
        # Get cluster health from device manager
        cluster_health = await self.device_manager.check_cluster_health()
        
        # Get coordinator device metrics
        coordinator_metrics = None
        if self.coordinator_device:
            coordinator_metrics = self.coordinator_device.get_metrics()
        
        # Get worker status
        workers = self.device_manager.get_workers()
        worker_status = {}
        for worker in workers:
            worker_health = await worker.check_health()
            worker_status[worker.device_id] = {
                'healthy': worker_health.is_healthy,
                'state': worker_health.state.value,
                'request_count': worker_health.request_count,
                'memory_usage_percent': worker_health.memory_usage_percent,
                'gpu_utilization_percent': worker_health.gpu_utilization_percent
            }
        
        return {
            'initialized': True,
            'device_id': self.device_config.device_id,
            'status': 'operational',
            'cluster_health': {
                'state': cluster_health.state.value,
                'total_devices': cluster_health.total_devices,
                'healthy_devices': cluster_health.healthy_devices,
                'coordinator_healthy': cluster_health.coordinator_healthy,
                'worker_health_ratio': cluster_health.worker_health_ratio
            },
            'coordinator_metrics': coordinator_metrics.__dict__ if coordinator_metrics else None,
            'worker_status': worker_status
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check through device abstraction.
        
        Returns:
            Health check results
        """
        try:
            if not self.is_initialized:
                return {
                    'healthy': False,
                    'reason': 'not_initialized',
                    'device_id': self.device_config.device_id
                }
            
            # Check cluster health through device manager
            cluster_health = await self.device_manager.check_cluster_health()
            
            # Check coordinator device health
            coordinator_health = None
            if self.coordinator_device:
                coordinator_health = await self.coordinator_device.check_health()
            
            # Determine overall health
            overall_healthy = (
                cluster_health.coordinator_healthy and
                cluster_health.worker_health_ratio >= 0.5 and
                (coordinator_health.is_healthy if coordinator_health else False)
            )
            
            return {
                'healthy': overall_healthy,
                'device_id': self.device_config.device_id,
                'cluster_health': {
                    'state': cluster_health.state.value,
                    'coordinator_healthy': cluster_health.coordinator_healthy,
                    'worker_health_ratio': cluster_health.worker_health_ratio,
                    'error_message': cluster_health.error_message
                },
                'coordinator_health': {
                    'is_healthy': coordinator_health.is_healthy if coordinator_health else False,
                    'state': coordinator_health.state.value if coordinator_health else 'unknown',
                    'uptime_seconds': coordinator_health.uptime_seconds if coordinator_health else 0
                } if coordinator_health else None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'device_id': self.device_config.device_id
            }
    
    async def shutdown(self):
        """Gracefully shutdown all components through device abstraction."""
        logger.info("Starting orchestrator shutdown...")
        
        self._shutdown_event.set()
        
        # Shutdown cluster through device manager
        try:
            await self.device_manager.shutdown_cluster()
            logger.info("Cluster shutdown completed")
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")
        
        # Clear references
        self.coordinator_device = None
        
        # Reset state
        self.is_initialized = False
        
        logger.info("Orchestrator shutdown completed")
    
    def is_ready(self) -> bool:
        """Check if orchestrator is ready to process requests."""
        return (
            self.is_initialized and
            self.device_manager.is_ready and
            self.coordinator_device is not None and
            self.coordinator_device.is_ready and
            not self._shutdown_event.is_set()
        )
    
    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for orchestrator to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if ready within timeout, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ready():
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get detailed device status information."""
        if not self.is_initialized:
            return {"error": "Orchestrator not initialized"}
        
        # Get coordinator device status
        coordinator_status = None
        if self.coordinator_device:
            coordinator_status = {
                'device_id': self.coordinator_device.device_id,
                'state': self.coordinator_device.state.value,
                'is_ready': self.coordinator_device.is_ready,
                'is_healthy': self.coordinator_device.is_healthy,
                'active_requests': self.coordinator_device.get_active_requests()
            }
        
        # Get worker device status
        workers = self.device_manager.get_workers()
        worker_status = {}
        for worker in workers:
            worker_status[worker.device_id] = {
                'state': worker.state.value,
                'is_ready': worker.is_ready,
                'is_healthy': worker.is_healthy,
                'assigned_layers': worker.get_assigned_layers(),
                'processing_requests': worker.get_processing_requests()
            }
        
        return {
            'coordinator': coordinator_status,
            'workers': worker_status,
            'cluster_state': self.device_manager.cluster_state.value
        }
"""
Worker device implementation for distributed MLX inference.

This module implements the worker device that processes assigned model layers
and responds to processing requests from the coordinator.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import mlx.core as mx

from .base_device import BaseDevice, DeviceState
from ..core.config import ClusterConfig, DeviceConfig, DeviceRole
from ..communication.grpc_server import start_grpc_server
from ..communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array

logger = logging.getLogger(__name__)


@dataclass
class LayerProcessingRequest:
    """Request for processing specific layers."""
    request_id: str
    input_tensor: mx.array
    layer_indices: List[int]
    context: Dict[str, Any]


@dataclass
class LayerProcessingResponse:
    """Response from layer processing."""
    request_id: str
    output_tensor: mx.array
    processing_time_ms: float
    device_id: str


class WorkerDevice(BaseDevice):
    """
    Worker device that processes assigned model layers.
    
    The worker device is responsible for:
    - Loading and managing assigned model layers
    - Processing layer computation requests from coordinator
    - Managing gRPC server for communication
    - Monitoring resource usage and health
    """
    
    def __init__(self, config: ClusterConfig, device_config: DeviceConfig):
        """
        Initialize the worker device.
        
        Args:
            config: Cluster configuration
            device_config: Configuration for this worker device
        """
        if device_config.role != DeviceRole.WORKER:
            raise ValueError(f"Device {device_config.device_id} is not configured as worker")
        
        super().__init__(config, device_config)
        
        # Worker-specific components
        self.grpc_server: Optional[Any] = None
        self.assigned_layers: List[int] = []
        self._processing_requests: Dict[str, float] = {}  # request_id -> start_time
        
        logger.info(f"Initialized worker device: {self.device_id}")
    
    async def _initialize_device(self) -> None:
        """Initialize worker-specific components."""
        logger.info("Initializing worker-specific components...")
        
        # Verify we have the correct role
        if self.role != DeviceRole.WORKER:
            raise RuntimeError("WorkerDevice must have WORKER role")
        
        # Get assigned layers
        self.assigned_layers = self.config.model.get_device_layers(self.device_id)
        
        if not self.assigned_layers:
            logger.warning(f"No layers assigned to worker {self.device_id}")
        else:
            logger.info(f"Worker {self.device_id} assigned layers: {self.assigned_layers}")
        
        # Initialize worker-specific resources
        await self._setup_worker_resources()
        
        logger.info("Worker device components initialized")
    
    async def _shutdown_device(self) -> None:
        """Shutdown worker-specific components."""
        logger.info("Shutting down worker components...")
        
        # Cancel processing requests
        self._processing_requests.clear()
        
        # Cleanup worker resources
        await self._cleanup_worker_resources()
        
        logger.info("Worker device shutdown completed")
    
    async def _initialize_communication(self) -> None:
        """Initialize gRPC server for receiving requests."""
        from ..communication.grpc_server import start_grpc_server
        
        logger.info("Initializing worker communication...")
        
        # Start gRPC server
        logger.info(f"Starting gRPC server on port {self.device_config.grpc_port}")
        self.grpc_server = await start_grpc_server(
            self.config,
            self.device_config,
            self.layer_processor
        )
        
        logger.info("Worker communication initialized")
    
    async def _cleanup_communication(self) -> None:
        """Cleanup communication resources."""
        logger.info("Cleaning up worker communication...")
        
        if self.grpc_server:
            # Stop gRPC server
            await self.grpc_server.stop(grace=5.0)
            self.grpc_server = None
        
        logger.info("Worker communication cleanup completed")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a layer computation request.
        
        Args:
            request_data: Dictionary containing request information
            
        Returns:
            Dictionary containing response data
        """
        # Convert request data to LayerProcessingRequest
        request = LayerProcessingRequest(
            request_id=request_data.get('request_id', ''),
            input_tensor=request_data.get('input_tensor'),
            layer_indices=request_data.get('layer_indices', []),
            context=request_data.get('context', {})
        )
        
        return await self.process_layer_request(request)
    
    async def process_layer_request(self, request: LayerProcessingRequest) -> Dict[str, Any]:
        """
        Process a layer computation request.
        
        Args:
            request: Layer processing request
            
        Returns:
            Dictionary containing response data
        """
        if not self.is_ready:
            raise RuntimeError("Worker device is not ready")
        
        start_time = self.record_request_start()
        processing_start_time = time.time()
        
        # Track processing request
        self._processing_requests[request.request_id] = processing_start_time
        
        try:
            async with self._state_lock:
                self._state = DeviceState.BUSY
            
            logger.debug(f"Processing layers {request.layer_indices} for request {request.request_id}")
            
            # Validate that we can process the requested layers
            if not self._can_process_layers(request.layer_indices):
                raise ValueError(
                    f"Worker {self.device_id} cannot process layers {request.layer_indices}. "
                    f"Assigned layers: {self.assigned_layers}"
                )
            
            # Process the layers
            output_tensor = await self._process_layers(
                request.input_tensor,
                request.layer_indices,
                request.context
            )
            
            processing_time = (time.time() - processing_start_time) * 1000
            
            response_data = {
                'request_id': request.request_id,
                'output_tensor': output_tensor,
                'processing_time_ms': processing_time,
                'device_id': self.device_id
            }
            
            self.record_request_completion(start_time, success=True)
            logger.debug(f"Completed layer processing for request {request.request_id} in {processing_time:.2f}ms")
            
            return response_data
            
        except Exception as e:
            self.record_request_completion(start_time, success=False)
            logger.error(f"Error processing layers for request {request.request_id}: {e}")
            raise
        
        finally:
            # Remove from processing requests
            self._processing_requests.pop(request.request_id, None)
            
            async with self._state_lock:
                self._state = DeviceState.READY
    
    async def _process_layers(self, 
                             input_tensor: mx.array, 
                             layer_indices: List[int],
                             context: Dict[str, Any]) -> mx.array:
        """
        Process the specified layers on the input tensor.
        
        Args:
            input_tensor: Input tensor to process
            layer_indices: Indices of layers to process
            context: Processing context
            
        Returns:
            Output tensor after processing
        """
        if not self.layer_processor:
            raise RuntimeError("Layer processor not initialized")
        
        # Process through the specified layers
        output_tensor = self.layer_processor.process(
            input_tensor,
            layer_indices,
            context
        )
        
        # Ensure computation is complete
        mx.eval(output_tensor)
        
        return output_tensor
    
    def _can_process_layers(self, layer_indices: List[int]) -> bool:
        """
        Check if this worker can process the specified layers.
        
        Args:
            layer_indices: Layer indices to check
            
        Returns:
            True if all layers can be processed, False otherwise
        """
        if not self.assigned_layers:
            return False
        
        # Check if all requested layers are in our assigned layers
        return all(layer_idx in self.assigned_layers for layer_idx in layer_indices)
    
    def get_assigned_layers(self) -> List[int]:
        """Get the list of layers assigned to this worker."""
        return self.assigned_layers.copy()
    
    def get_processing_requests(self) -> Dict[str, float]:
        """Get currently processing requests and their start times."""
        return self._processing_requests.copy()
    
    async def _setup_worker_resources(self) -> None:
        """Setup worker-specific resources."""
        # Initialize any worker-specific resources here
        # For example, layer caching, optimization state, etc.
        
        # Pre-warm the assigned layers if needed
        if self.layer_processor and self.assigned_layers:
            await self._prewarm_layers()
    
    async def _cleanup_worker_resources(self) -> None:
        """Cleanup worker-specific resources."""
        # Cleanup worker-specific resources here
        pass
    
    async def _prewarm_layers(self) -> None:
        """Pre-warm the assigned layers for better performance."""
        try:
            logger.info(f"Pre-warming layers {self.assigned_layers} on worker {self.device_id}")
            
            # Create a dummy input tensor to warm up the layers
            dummy_input = mx.zeros((1, 1, self.model.args.hidden_size if hasattr(self.model, 'args') else 4096))
            
            # Process through assigned layers
            if self.layer_processor:
                _ = self.layer_processor.process(dummy_input, self.assigned_layers, {})
                mx.eval(_)  # Ensure computation is complete
            
            logger.info(f"Layer pre-warming completed for worker {self.device_id}")
            
        except Exception as e:
            logger.warning(f"Failed to pre-warm layers on worker {self.device_id}: {e}")
    
    async def get_layer_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for assigned layers."""
        return {
            'assigned_layers': self.assigned_layers,
            'total_requests': self._metrics.total_requests,
            'successful_requests': self._metrics.successful_requests,
            'failed_requests': self._metrics.failed_requests,
            'average_processing_time_ms': self._metrics.average_processing_time_ms,
            'active_requests': len(self._processing_requests),
            'memory_usage_percent': (await self.check_health()).memory_usage_percent,
            'gpu_utilization_percent': (await self.check_health()).gpu_utilization_percent
        }
    
    def can_accept_request(self) -> bool:
        """Check if the worker can accept a new request."""
        # Simple capacity check - can be enhanced with more sophisticated logic
        max_concurrent_requests = 10  # Configurable
        return (
            self.is_ready and 
            len(self._processing_requests) < max_concurrent_requests
        )
    
    async def estimate_processing_time(self, layer_indices: List[int]) -> float:
        """
        Estimate processing time for given layers.
        
        Args:
            layer_indices: Layer indices to estimate
            
        Returns:
            Estimated processing time in milliseconds
        """
        if not self._can_process_layers(layer_indices):
            return float('inf')
        
        # Simple estimation based on historical averages
        # Can be enhanced with more sophisticated modeling
        base_time_per_layer = self._metrics.average_processing_time_ms / max(len(self.assigned_layers), 1)
        return base_time_per_layer * len(layer_indices)
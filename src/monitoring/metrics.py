"""
Metrics collection and tracking for distributed MLX inference.

This module provides comprehensive metrics collection including:
- Device utilization (CPU, memory, model loading status)
- Inference throughput (requests/second, tokens/second)
- Latency metrics (per-device processing time, end-to-end latency)
- Error rates and failure modes
- Network connectivity between devices
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
import psutil
import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class DeviceMetrics:
    """Metrics for a single device."""
    device_id: str
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    model_loaded: bool = False
    assigned_layers: List[int] = field(default_factory=list)
    is_healthy: bool = True


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    total_tokens: int = 0
    generated_tokens: int = 0
    device_times: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> float:
        """Get tokens per second throughput."""
        if self.duration_ms <= 0:
            return 0.0
        return (self.generated_tokens / self.duration_ms) * 1000


@dataclass
class NetworkMetrics:
    """Network connectivity metrics between devices."""
    source_device: str
    target_device: str
    timestamp: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


class MetricsCollector:
    """Central metrics collector for the distributed system."""
    
    def __init__(self, device_id: str, collection_interval: float = 1.0):
        """
        Initialize metrics collector.
        
        Args:
            device_id: ID of the device this collector runs on
            collection_interval: How often to collect metrics (seconds)
        """
        self.device_id = device_id
        self.collection_interval = collection_interval
        
        # Storage for metrics (thread-safe)
        self._lock = threading.Lock()
        
        # Device metrics (latest only)
        self._device_metrics: Dict[str, DeviceMetrics] = {}
        
        # Inference metrics (rolling window)
        self._inference_metrics: Deque[InferenceMetrics] = deque(maxlen=1000)
        self._active_requests: Dict[str, InferenceMetrics] = {}
        
        # Network metrics (rolling window)
        self._network_metrics: Deque[NetworkMetrics] = deque(maxlen=1000)
        
        # Aggregated statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_processing_time_ms': 0.0,
        }
        
        # Collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Initialized metrics collector for {device_id}")
    
    async def start(self):
        """Start metrics collection."""
        if self._running:
            logger.warning("Metrics collector already running")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop(self):
        """Stop metrics collection."""
        if not self._running:
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                await self._collect_local_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_local_metrics(self):
        """Collect metrics for the local device."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Try to get MLX GPU metrics
            gpu_util = None
            gpu_mem_used = None
            gpu_mem_total = None
            
            try:
                # MLX doesn't provide direct GPU utilization, but we can check memory
                # This is a placeholder - actual implementation would depend on MLX capabilities
                if hasattr(mx, 'metal') and mx.metal.is_available():
                    # MLX on Apple Silicon
                    gpu_util = 0.0  # Placeholder
                    gpu_mem_used = 0.0  # Placeholder
                    gpu_mem_total = memory.total / (1024**3)  # Use system memory as approximation
            except Exception as e:
                logger.debug(f"Could not get GPU metrics: {e}")
            
            # Create device metrics
            metrics = DeviceMetrics(
                device_id=self.device_id,
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                gpu_utilization=gpu_util,
                gpu_memory_used_gb=gpu_mem_used,
                gpu_memory_total_gb=gpu_mem_total,
                model_loaded=False,  # Will be updated by external components
                assigned_layers=[],  # Will be updated by external components
                is_healthy=True  # Will be updated by health checks
            )
            
            with self._lock:
                self._device_metrics[self.device_id] = metrics
                
        except Exception as e:
            logger.error(f"Error collecting local metrics: {e}")
    
    def update_device_status(self, device_id: str, model_loaded: bool, 
                           assigned_layers: List[int], is_healthy: bool):
        """Update device status information."""
        with self._lock:
            if device_id in self._device_metrics:
                metrics = self._device_metrics[device_id]
                metrics.model_loaded = model_loaded
                metrics.assigned_layers = assigned_layers.copy()
                metrics.is_healthy = is_healthy
    
    def start_request(self, request_id: str, total_tokens: int = 0) -> InferenceMetrics:
        """Start tracking an inference request."""
        metrics = InferenceMetrics(
            request_id=request_id,
            start_time=time.time(),
            total_tokens=total_tokens
        )
        
        with self._lock:
            self._active_requests[request_id] = metrics
            self._stats['total_requests'] += 1
        
        logger.debug(f"Started tracking request {request_id}")
        return metrics
    
    def complete_request(self, request_id: str, generated_tokens: int, 
                        device_times: Dict[str, float], error: Optional[str] = None):
        """Complete tracking an inference request."""
        with self._lock:
            if request_id not in self._active_requests:
                logger.warning(f"Request {request_id} not found in active requests")
                return
            
            metrics = self._active_requests.pop(request_id)
            metrics.end_time = time.time()
            metrics.generated_tokens = generated_tokens
            metrics.device_times = device_times.copy()
            metrics.error = error
            
            # Add to historical metrics
            self._inference_metrics.append(metrics)
            
            # Update stats
            if error:
                self._stats['failed_requests'] += 1
            else:
                self._stats['successful_requests'] += 1
                self._stats['total_tokens_generated'] += generated_tokens
            
            self._stats['total_processing_time_ms'] += metrics.duration_ms
        
        logger.debug(f"Completed tracking request {request_id} - "
                    f"{generated_tokens} tokens in {metrics.duration_ms:.1f}ms")
    
    def record_network_metric(self, target_device: str, latency_ms: float, 
                            success: bool, error: Optional[str] = None):
        """Record network connectivity metric."""
        metric = NetworkMetrics(
            source_device=self.device_id,
            target_device=target_device,
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        
        with self._lock:
            self._network_metrics.append(metric)
        
        logger.debug(f"Recorded network metric to {target_device}: "
                    f"{latency_ms:.1f}ms, success={success}")
    
    def get_device_metrics(self, device_id: Optional[str] = None) -> Dict[str, DeviceMetrics]:
        """Get current device metrics."""
        with self._lock:
            if device_id:
                return {device_id: self._device_metrics.get(device_id)} if device_id in self._device_metrics else {}
            return self._device_metrics.copy()
    
    def get_inference_stats(self, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get inference statistics."""
        with self._lock:
            if time_window_seconds is None:
                # Return all-time stats
                recent_metrics = list(self._inference_metrics)
            else:
                # Filter to time window
                cutoff_time = time.time() - time_window_seconds
                recent_metrics = [m for m in self._inference_metrics if m.start_time >= cutoff_time]
            
            if not recent_metrics:
                return {
                    'requests_count': 0,
                    'success_rate': 0.0,
                    'avg_latency_ms': 0.0,
                    'avg_tokens_per_second': 0.0,
                    'total_tokens_generated': 0,
                    'error_rate': 0.0
                }
            
            successful = [m for m in recent_metrics if m.error is None]
            failed = [m for m in recent_metrics if m.error is not None]
            
            total_duration = sum(m.duration_ms for m in recent_metrics)
            total_tokens = sum(m.generated_tokens for m in successful)
            
            return {
                'requests_count': len(recent_metrics),
                'successful_requests': len(successful),
                'failed_requests': len(failed),
                'success_rate': len(successful) / len(recent_metrics) if recent_metrics else 0.0,
                'error_rate': len(failed) / len(recent_metrics) if recent_metrics else 0.0,
                'avg_latency_ms': total_duration / len(recent_metrics) if recent_metrics else 0.0,
                'avg_tokens_per_second': (total_tokens / total_duration * 1000) if total_duration > 0 else 0.0,
                'total_tokens_generated': total_tokens,
                'time_window_seconds': time_window_seconds,
            }
    
    def get_network_stats(self, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get network connectivity statistics."""
        with self._lock:
            if time_window_seconds is None:
                recent_metrics = list(self._network_metrics)
            else:
                cutoff_time = time.time() - time_window_seconds
                recent_metrics = [m for m in self._network_metrics if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {
                    'total_connections': 0,
                    'successful_connections': 0,
                    'avg_latency_ms': 0.0,
                    'success_rate': 0.0
                }
            
            successful = [m for m in recent_metrics if m.success]
            
            return {
                'total_connections': len(recent_metrics),
                'successful_connections': len(successful),
                'success_rate': len(successful) / len(recent_metrics) if recent_metrics else 0.0,
                'avg_latency_ms': sum(m.latency_ms for m in successful) / len(successful) if successful else 0.0,
                'time_window_seconds': time_window_seconds,
            }
    
    def get_throughput_metrics(self, time_window_seconds: float = 60.0) -> Dict[str, float]:
        """Get throughput metrics (requests/second, tokens/second)."""
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds
        
        with self._lock:
            recent_metrics = [m for m in self._inference_metrics 
                            if m.start_time >= cutoff_time and m.error is None]
            
            if not recent_metrics:
                return {
                    'requests_per_second': 0.0,
                    'tokens_per_second': 0.0,
                    'avg_latency_ms': 0.0
                }
            
            total_requests = len(recent_metrics)
            total_tokens = sum(m.generated_tokens for m in recent_metrics)
            total_duration = sum(m.duration_ms for m in recent_metrics)
            
            return {
                'requests_per_second': total_requests / time_window_seconds,
                'tokens_per_second': total_tokens / time_window_seconds,
                'avg_latency_ms': total_duration / total_requests if total_requests > 0 else 0.0,
                'time_window_seconds': time_window_seconds
            }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring systems."""
        with self._lock:
            return {
                'timestamp': time.time(),
                'device_id': self.device_id,
                'device_metrics': {k: {
                    'device_id': v.device_id,
                    'timestamp': v.timestamp,
                    'cpu_percent': v.cpu_percent,
                    'memory_percent': v.memory_percent,
                    'memory_used_gb': v.memory_used_gb,
                    'memory_total_gb': v.memory_total_gb,
                    'gpu_utilization': v.gpu_utilization,
                    'gpu_memory_used_gb': v.gpu_memory_used_gb,
                    'gpu_memory_total_gb': v.gpu_memory_total_gb,
                    'model_loaded': v.model_loaded,
                    'assigned_layers': v.assigned_layers,
                    'is_healthy': v.is_healthy
                } for k, v in self._device_metrics.items()},
                'inference_stats': self.get_inference_stats(300),  # Last 5 minutes
                'network_stats': self.get_network_stats(300),  # Last 5 minutes
                'throughput_metrics': self.get_throughput_metrics(60),  # Last minute
                'all_time_stats': self._stats.copy()
            }
"""
gRPC Health Checking Service implementation.
Provides standard health checks for coordinator and workers.
"""

import time
import logging
from typing import Dict, Set
from concurrent import futures
import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

logger = logging.getLogger(__name__)


class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    Implementation of the gRPC Health Checking Protocol.
    
    This allows coordinators to proactively detect hung or crashed workers.
    """
    
    def __init__(self, service_name: str = "mlx.inference"):
        """
        Initialize health servicer.
        
        Args:
            service_name: Name of this service for health checks
        """
        self.service_name = service_name
        self._status: Dict[str, health_pb2.HealthCheckResponse.ServingStatus] = {
            "": health_pb2.HealthCheckResponse.SERVING,  # Overall status
            service_name: health_pb2.HealthCheckResponse.SERVING
        }
        self._watchers: Dict[str, Set[futures.Future]] = {}
        self._start_time = time.time()
        
    def Check(self, request, context):
        """
        Synchronous health check.
        
        Returns current health status for the requested service.
        """
        service = request.service or ""
        
        if service in self._status:
            return health_pb2.HealthCheckResponse(
                status=self._status[service]
            )
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Service {service} not found")
            return health_pb2.HealthCheckResponse()
    
    def Watch(self, request, context):
        """
        Streaming health check.
        
        Sends updates whenever health status changes.
        """
        service = request.service or ""
        
        # Send initial status
        if service in self._status:
            yield health_pb2.HealthCheckResponse(
                status=self._status[service]
            )
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Service {service} not found")
            return
        
        # Create a future for this watcher
        watcher_future = futures.Future()
        
        # Add to watchers
        if service not in self._watchers:
            self._watchers[service] = set()
        self._watchers[service].add(watcher_future)
        
        try:
            # Wait for status changes
            while not context.is_active():
                try:
                    new_status = watcher_future.result(timeout=30)
                    yield health_pb2.HealthCheckResponse(status=new_status)
                    
                    # Reset future for next update
                    watcher_future = futures.Future()
                    self._watchers[service].add(watcher_future)
                except futures.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield health_pb2.HealthCheckResponse(
                        status=self._status[service]
                    )
        finally:
            # Clean up watcher
            if service in self._watchers:
                self._watchers[service].discard(watcher_future)
    
    def set_status(self, service: str, status: health_pb2.HealthCheckResponse.ServingStatus):
        """
        Update health status for a service.
        
        Args:
            service: Service name (empty string for overall status)
            status: New health status
        """
        old_status = self._status.get(service)
        self._status[service] = status
        
        # Notify watchers if status changed
        if old_status != status and service in self._watchers:
            for watcher in list(self._watchers[service]):
                if not watcher.done():
                    watcher.set_result(status)
            self._watchers[service].clear()
        
        logger.info(f"Health status updated: {service or 'overall'} = {status}")
    
    def set_serving(self, service: str = ""):
        """Mark service as healthy and serving."""
        self.set_status(service, health_pb2.HealthCheckResponse.SERVING)
    
    def set_not_serving(self, service: str = ""):
        """Mark service as unhealthy."""
        self.set_status(service, health_pb2.HealthCheckResponse.NOT_SERVING)
    
    def shutdown(self):
        """Mark all services as not serving during shutdown."""
        for service in list(self._status.keys()):
            self.set_not_serving(service)
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._start_time


class HealthChecker:
    """
    Client-side health checker for monitoring remote services.
    """
    
    def __init__(self, channel: grpc.Channel, service_name: str = ""):
        """
        Initialize health checker.
        
        Args:
            channel: gRPC channel to the service
            service_name: Service name to check (empty for overall)
        """
        self.stub = health_pb2_grpc.HealthStub(channel)
        self.service_name = service_name
    
    def check_health(self, timeout: float = 5.0) -> bool:
        """
        Perform synchronous health check.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            request = health_pb2.HealthCheckRequest(service=self.service_name)
            response = self.stub.Check(request, timeout=timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        except grpc.RpcError as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def watch_health(self, callback, timeout: float = None):
        """
        Watch health status with streaming updates.
        
        Args:
            callback: Function called with (is_healthy: bool) on status changes
            timeout: Overall timeout (None for infinite)
        """
        try:
            request = health_pb2.HealthCheckRequest(service=self.service_name)
            
            for response in self.stub.Watch(request, timeout=timeout):
                is_healthy = response.status == health_pb2.HealthCheckResponse.SERVING
                callback(is_healthy)
                
        except grpc.RpcError as e:
            logger.error(f"Health watch failed: {e}")
            callback(False)


def add_health_service_to_server(
    server: grpc.Server,
    health_servicer: Optional[HealthServicer] = None
) -> HealthServicer:
    """
    Add health checking service to a gRPC server.
    
    Args:
        server: gRPC server instance
        health_servicer: Optional custom health servicer
        
    Returns:
        The health servicer instance
    """
    if health_servicer is None:
        health_servicer = HealthServicer()
    
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    return health_servicer


# Example usage in worker/coordinator
def setup_health_monitoring(server: grpc.Server, device_id: str) -> HealthServicer:
    """
    Set up health monitoring for a device.
    
    Args:
        server: gRPC server instance
        device_id: Device identifier
        
    Returns:
        Configured health servicer
    """
    health_servicer = HealthServicer(f"mlx.inference.{device_id}")
    add_health_service_to_server(server, health_servicer)
    
    # Mark as serving
    health_servicer.set_serving()
    health_servicer.set_serving(f"mlx.inference.{device_id}")
    
    logger.info(f"Health monitoring enabled for {device_id}")
    
    return health_servicer
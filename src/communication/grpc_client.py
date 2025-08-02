"""
gRPC client for inter-device communication.
"""

import grpc
import time
import logging
from typing import Dict, List, Optional, Any
import mlx.core as mx
from dataclasses import dataclass

from ..core.config import DeviceConfig
from .tensor_utils import serialize_mlx_array, deserialize_mlx_array
from .dns_resolver import get_global_resolver, EnhancedDNSResolver

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing layers on a remote device."""

    output_tensor: mx.array
    processing_time_ms: float
    device_id: str


class GRPCInferenceClient:
    """Client for communicating with remote inference workers."""

    def __init__(
        self,
        target_device: DeviceConfig,
        timeout: float = 60.0,
        dns_resolver: Optional[EnhancedDNSResolver] = None,
        metrics_collector=None,
    ):
        """
        Initialize client for a specific target device.

        Args:
            target_device: Configuration of the target device
            timeout: Request timeout in seconds
            dns_resolver: DNS resolver instance (uses global if None)
            metrics_collector: Optional metrics collector for network monitoring
        """
        self.target_device = target_device
        self.timeout = timeout
        self.dns_resolver = dns_resolver or get_global_resolver()
        self.metrics_collector = metrics_collector
        self.channel = None
        self.stub = None
        self._current_address = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3

        # Register for network change notifications
        self.dns_resolver.add_network_change_callback(self._on_network_change)

        self._connect()

    def _connect(self):
        """Establish connection to the target device with enhanced DNS resolution."""
        # Import generated code
        from . import inference_pb2_grpc

        # Close existing connection if any
        if self.channel:
            try:
                self.channel.close()
            except Exception as e:
                logger.debug(f"Error closing existing channel: {e}")

        # Create channel with increased message size limits and DNS options
        options = [
            ("grpc.max_send_message_length", 512 * 1024 * 1024),  # 512MB
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),  # 512MB
            ("grpc.dns_enable_srv_queries", False),  # Disable SRV queries
            ("grpc.enable_retries", False),  # Disable retries for faster failure
            ("grpc.keepalive_time_ms", 60000),  # Keepalive every 60s instead of 30s
            ("grpc.keepalive_timeout_ms", 20000),  # More generous timeout
            ("grpc.keepalive_permit_without_calls", False),  # Only ping when there's activity
            ("grpc.http2.min_time_between_pings_ms", 60000),  # Minimum 60s between pings
        ]

        # Use enhanced DNS resolution
        result = self.dns_resolver.resolve_hostname_detailed(self.target_device.hostname)

        if result.ip_address:
            address = f"{result.ip_address}:{self.target_device.grpc_port}"
            self._current_address = address

            logger.info(
                f"Resolved {self.target_device.hostname} -> {result.ip_address} "
                f"using {result.strategy.value if result.strategy else 'unknown'} "
                f"in {result.resolution_time_ms:.1f}ms"
            )
        else:
            # Fall back to original hostname
            address = f"{self.target_device.hostname}:{self.target_device.grpc_port}"
            self._current_address = address
            logger.warning(
                f"Could not resolve {self.target_device.hostname}, using original hostname. Error: {result.error}"
            )

        try:
            self.channel = grpc.insecure_channel(address, options=options)
            self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

            # Test connection with a quick health check
            try:
                health_response = self.health_check()
                if health_response.get("healthy"):
                    logger.info(
                        f"Successfully connected to {self.target_device.device_id} at {address}"
                    )
                    self._connection_attempts = 0
                else:
                    logger.warning(
                        f"Connected to {self.target_device.device_id} at {address} but health check failed"
                    )
            except Exception as e:
                logger.warning(
                    f"Connected to {self.target_device.device_id} at {address} but health check failed: {e}"
                )

        except Exception as e:
            logger.error(f"Failed to connect to {self.target_device.device_id} at {address}: {e}")
            self._connection_attempts += 1
            raise

    def process_layers(
        self,
        input_tensor: mx.array,
        layer_indices: List[int],
        request_id: str,
        context: Optional[Dict[str, str]] = None,
        compress: bool = False,
    ) -> ProcessingResult:
        """
        Process layers on the remote device.

        Args:
            input_tensor: Input tensor
            layer_indices: Indices of layers to process
            request_id: Unique request identifier
            context: Optional context information
            compress: Whether to compress tensor data

        Returns:
            ProcessingResult with output tensor and metadata
        """
        from . import inference_pb2

        try:
            # Serialize input tensor
            tensor_data, metadata = serialize_mlx_array(input_tensor, compress=compress)

            # Create request
            request = inference_pb2.LayerRequest(
                request_id=request_id,
                input_tensor=tensor_data,
                layer_indices=layer_indices,
                metadata=inference_pb2.TensorMetadata(
                    shape=metadata["shape"],
                    dtype=metadata["dtype"],
                    compressed=metadata["compressed"],
                ),
                context=context or {},
            )

            # Make RPC call
            start_time = time.time()
            response = self.stub.ProcessLayers(request, timeout=self.timeout)
            rpc_time = (time.time() - start_time) * 1000

            # Deserialize output
            output_metadata = {
                "shape": list(response.metadata.shape),
                "dtype": response.metadata.dtype,
                "compressed": response.metadata.compressed,
            }
            output_tensor = deserialize_mlx_array(response.output_tensor, output_metadata)

            logger.info(
                f"Received response from {response.device_id} in {rpc_time:.1f}ms "
                f"(processing: {response.processing_time_ms:.1f}ms)"
            )

            return ProcessingResult(
                output_tensor=output_tensor,
                processing_time_ms=response.processing_time_ms,
                device_id=response.device_id,
            )

        except grpc.RpcError as e:
            logger.error(
                f"RPC failed to {self.target_device.device_id}: {e.code()} - {e.details()}"
            )

            # Try to reconnect on certain types of failures
            if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                if self._connection_attempts < self._max_connection_attempts:
                    logger.info(
                        f"Attempting to reconnect to {self.target_device.device_id} (attempt {self._connection_attempts + 1})"
                    )
                    try:
                        # Invalidate DNS cache for this hostname and reconnect
                        self.dns_resolver.invalidate_hostname(self.target_device.hostname)
                        self._connect()

                        # Retry the request once after reconnection
                        response = self.stub.ProcessLayers(request, timeout=self.timeout)
                        rpc_time = (time.time() - start_time) * 1000

                        # Deserialize output
                        output_metadata = {
                            "shape": list(response.metadata.shape),
                            "dtype": response.metadata.dtype,
                            "compressed": response.metadata.compressed,
                        }
                        output_tensor = deserialize_mlx_array(
                            response.output_tensor, output_metadata
                        )

                        logger.info(
                            f"Received response from {response.device_id} after reconnect in {rpc_time:.1f}ms "
                            f"(processing: {response.processing_time_ms:.1f}ms)"
                        )

                        return ProcessingResult(
                            output_tensor=output_tensor,
                            processing_time_ms=response.processing_time_ms,
                            device_id=response.device_id,
                        )
                    except Exception as retry_error:
                        logger.error(
                            f"Retry failed for {self.target_device.device_id}: {retry_error}"
                        )

            raise
        except Exception as e:
            logger.error(f"Error processing layers on {self.target_device.device_id}: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Check health of the remote device."""
        from . import inference_pb2

        start_time = time.time()
        success = False
        error = None

        try:
            request = inference_pb2.Empty()
            response = self.stub.HealthCheck(request, timeout=5.0)

            success = response.healthy
            latency_ms = (time.time() - start_time) * 1000

            # Record network metric if metrics collector is available
            if self.metrics_collector:
                self.metrics_collector.record_network_metric(
                    self.target_device.device_id, latency_ms, success, error=None
                )

            return {
                "healthy": response.healthy,
                "device_id": response.device_id,
                "timestamp": response.timestamp,
                "details": dict(response.details),
                "latency_ms": latency_ms,
            }
        except grpc.RpcError as e:
            error = str(e)
            latency_ms = (time.time() - start_time) * 1000

            # Record network metric failure if metrics collector is available
            if self.metrics_collector:
                self.metrics_collector.record_network_metric(
                    self.target_device.device_id, latency_ms, False, error=error
                )

            logger.error(f"Health check failed for {self.target_device.device_id}: {e}")
            return {
                "healthy": False,
                "device_id": self.target_device.device_id,
                "error": error,
                "latency_ms": latency_ms,
            }

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the remote device."""
        from . import inference_pb2

        try:
            request = inference_pb2.Empty()
            response = self.stub.GetDeviceInfo(request, timeout=5.0)

            return {
                "device_id": response.device_id,
                "hostname": response.hostname,
                "rank": response.rank,
                "role": response.role,
                "assigned_layers": list(response.assigned_layers),
                "capabilities": dict(response.capabilities),
                "gpu_utilization": response.gpu_utilization,
                "memory_usage_gb": response.memory_usage_gb,
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to get device info for {self.target_device.device_id}: {e}")
            raise

    def _on_network_change(self):
        """Handle network change events."""
        logger.info(
            f"Network change detected, invalidating DNS cache for {self.target_device.hostname}"
        )
        self.dns_resolver.invalidate_hostname(self.target_device.hostname)

        # Optionally reconnect proactively
        try:
            self._connect()
        except Exception as e:
            logger.warning(f"Proactive reconnection failed for {self.target_device.device_id}: {e}")

    def reconnect(self):
        """Manually trigger reconnection."""
        logger.info(f"Manual reconnection requested for {self.target_device.device_id}")
        try:
            self.dns_resolver.invalidate_hostname(self.target_device.hostname)
            self._connect()
        except Exception as e:
            logger.error(f"Manual reconnection failed for {self.target_device.device_id}: {e}")
            raise

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current connection."""
        return {
            "device_id": self.target_device.device_id,
            "hostname": self.target_device.hostname,
            "current_address": self._current_address,
            "connection_attempts": self._connection_attempts,
            "channel_state": str(self.channel.get_state()) if self.channel else "None",
        }

    def close(self):
        """Close the connection."""
        # Remove network change callback
        self.dns_resolver.remove_network_change_callback(self._on_network_change)

        if self.channel:
            self.channel.close()
            logger.info(f"Closed connection to {self.target_device.device_id}")


class ConnectionPool:
    """Pool of gRPC connections to remote devices with enhanced DNS resolution."""

    def __init__(
        self,
        config,
        local_device_id: str,
        dns_resolver: Optional[EnhancedDNSResolver] = None,
        metrics_collector=None,
    ):
        """
        Initialize connection pool.

        Args:
            config: Cluster configuration
            local_device_id: ID of the local device
            dns_resolver: DNS resolver instance (uses global if None)
            metrics_collector: Optional metrics collector for network monitoring
        """
        self.config = config
        self.local_device_id = local_device_id
        self.dns_resolver = dns_resolver or get_global_resolver()
        self.metrics_collector = metrics_collector
        self.clients: Dict[str, GRPCInferenceClient] = {}

        # Start DNS monitoring if not already active
        if not self.dns_resolver._monitoring_active:
            self.dns_resolver.start_monitoring()

        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize connections to all remote devices."""
        for device in self.config.devices:
            if device.device_id != self.local_device_id:
                try:
                    client = GRPCInferenceClient(
                        device,
                        timeout=self.config.performance.request_timeout_seconds,
                        dns_resolver=self.dns_resolver,
                        metrics_collector=self.metrics_collector,
                    )
                    self.clients[device.device_id] = client
                    logger.info(f"Initialized connection to {device.device_id}")
                except Exception as e:
                    logger.error(f"Failed to connect to {device.device_id}: {e}")

    def get_client(self, device_id: str) -> Optional[GRPCInferenceClient]:
        """Get client for a specific device."""
        return self.clients.get(device_id)

    def get_next_device_client(self, current_device_id: str) -> Optional[GRPCInferenceClient]:
        """Get client for the next device in the pipeline."""
        current_device = self.config.get_device(current_device_id)
        if not current_device:
            return None

        # Find next device by rank
        next_rank = current_device.rank + 1
        next_device = self.config.get_device_by_rank(next_rank)

        if next_device:
            return self.get_client(next_device.device_id)
        return None

    def reconnect_all(self):
        """Reconnect all clients."""
        for device_id, client in self.clients.items():
            try:
                client.reconnect()
                logger.info(f"Reconnected to {device_id}")
            except Exception as e:
                logger.error(f"Failed to reconnect to {device_id}: {e}")

    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get connection status for all clients."""
        status = {}
        for device_id, client in self.clients.items():
            try:
                status[device_id] = client.get_connection_info()
                # Add health check
                health = client.health_check()
                status[device_id]["healthy"] = health.get("healthy", False)
            except Exception as e:
                status[device_id] = {"device_id": device_id, "error": str(e), "healthy": False}
        return status

    def get_dns_stats(self) -> Dict[str, Any]:
        """Get DNS resolver statistics."""
        return self.dns_resolver.get_cache_stats()

    def close_all(self):
        """Close all connections."""
        for client in self.clients.values():
            client.close()
        self.clients.clear()

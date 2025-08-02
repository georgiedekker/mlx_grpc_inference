"""
gRPC client for coordinating distributed MLX inference.

This module provides the client-side implementation for managing
distributed inference across multiple devices.
"""

import grpc
import mlx.core as mx
import numpy as np
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, Future

# Import generated protobuf modules
try:
    import distributed_inference_pb2 as pb2
    import distributed_inference_pb2_grpc as pb2_grpc
except ImportError:
    logger.warning("Protocol buffer modules not found. Run generate_proto.sh first.")
    pb2 = None
    pb2_grpc = None

from grpc_server import TensorSerializer
from device_capabilities import DeviceProfile
from sharding_strategy import ShardingPlan, ShardAssignment

logger = logging.getLogger(__name__)


@dataclass
class DeviceConnection:
    """Represents a connection to a remote device."""
    device_id: str
    hostname: str
    port: int
    channel: grpc.Channel
    stub: pb2_grpc.DistributedInferenceStub
    assignment: Optional[ShardAssignment] = None
    healthy: bool = True
    last_health_check: float = 0.0


class DistributedInferenceClient:
    """Client for coordinating distributed inference across devices."""
    
    def __init__(self, timeout: float = 30.0, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self.connections: Dict[str, DeviceConnection] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._health_check_interval = 10.0
        self._health_check_thread = None
        self._stop_health_checks = threading.Event()
        
        logger.info("Initialized distributed inference client")
    
    def connect_device(self, device_id: str, hostname: str, port: int) -> DeviceConnection:
        """Connect to a remote device."""
        logger.info(f"Connecting to device {device_id} at {hostname}:{port}")
        
        # Create channel with options
        options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),  # 1GB
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),  # 1GB
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
        ]
        
        channel = grpc.insecure_channel(f'{hostname}:{port}', options=options)
        stub = pb2_grpc.DistributedInferenceStub(channel)
        
        # Test connection
        try:
            request = pb2.HealthCheckRequest(include_stats=False)
            response = stub.HealthCheck(request, timeout=5.0)
            if not response.healthy:
                raise Exception(f"Device unhealthy: {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to {device_id}: {e}")
            raise
        
        connection = DeviceConnection(
            device_id=device_id,
            hostname=hostname,
            port=port,
            channel=channel,
            stub=stub,
            healthy=True,
            last_health_check=time.time()
        )
        
        self.connections[device_id] = connection
        logger.info(f"Successfully connected to {device_id}")
        
        return connection
    
    def get_device_capabilities(self, device_id: str) -> pb2.DeviceCapabilities:
        """Get capabilities of a connected device."""
        if device_id not in self.connections:
            raise ValueError(f"Device {device_id} not connected")
        
        conn = self.connections[device_id]
        request = pb2.GetCapabilitiesRequest()
        
        try:
            response = conn.stub.GetCapabilities(request, timeout=self.timeout)
            return response.capabilities
        except Exception as e:
            logger.error(f"Failed to get capabilities from {device_id}: {e}")
            raise
    
    def initialize_shards(self, model_name: str, model_provider: str,
                         sharding_plan: ShardingPlan) -> Dict[str, bool]:
        """Initialize model shards on all devices according to the sharding plan."""
        logger.info(f"Initializing shards for model {model_name}")
        
        # Create futures for parallel initialization
        futures: Dict[str, Future] = {}
        
        for assignment in sharding_plan.assignments:
            device_id = assignment.device_id
            
            if device_id not in self.connections:
                logger.error(f"Device {device_id} not connected")
                continue
            
            conn = self.connections[device_id]
            conn.assignment = assignment
            
            # Create shard info
            shard_info = pb2.ShardInfo(
                device_id=device_id,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                layer_names=[f"layer_{i}" for i in range(assignment.start_layer, assignment.end_layer)],
                shard_size_bytes=int(assignment.estimated_memory_gb * 1024**3)
            )
            
            # Create request
            request = pb2.InitializeShardRequest(
                model_name=model_name,
                model_provider=model_provider,
                shard_info=shard_info,
                load_from_cache=True
            )
            
            # Submit initialization task
            future = self.executor.submit(self._initialize_shard, conn, request)
            futures[device_id] = future
        
        # Wait for all initializations to complete
        results = {}
        for device_id, future in futures.items():
            try:
                success = future.result(timeout=self.timeout * 2)  # Allow extra time for loading
                results[device_id] = success
            except Exception as e:
                logger.error(f"Failed to initialize shard on {device_id}: {e}")
                results[device_id] = False
        
        # Check if all succeeded
        if all(results.values()):
            logger.info("All shards initialized successfully")
        else:
            failed = [d for d, success in results.items() if not success]
            logger.error(f"Failed to initialize shards on devices: {failed}")
        
        return results
    
    def _initialize_shard(self, conn: DeviceConnection, 
                         request: pb2.InitializeShardRequest) -> bool:
        """Initialize shard on a single device."""
        try:
            response = conn.stub.InitializeShard(request, timeout=self.timeout * 2)
            if response.success:
                logger.info(f"Initialized shard on {conn.device_id} in {response.load_time_ms}ms")
                return True
            else:
                logger.error(f"Failed to initialize shard on {conn.device_id}: {response.message}")
                return False
        except Exception as e:
            logger.error(f"Error initializing shard on {conn.device_id}: {e}")
            return False
    
    def forward_pipeline(self, input_ids: mx.array, 
                        max_tokens: int = 100,
                        temperature: float = 0.7) -> List[int]:
        """Run forward pass through the distributed pipeline.
        
        This implements pipeline parallelism where each device processes
        its assigned layers sequentially.
        """
        if not self.connections:
            raise ValueError("No devices connected")
        
        # Sort connections by layer order
        sorted_connections = sorted(
            self.connections.values(),
            key=lambda c: c.assignment.start_layer if c.assignment else float('inf')
        )
        
        # Verify we have complete pipeline
        if not all(c.assignment for c in sorted_connections):
            raise ValueError("Not all devices have shard assignments")
        
        tokens = []
        cache_dict = {}
        
        # Generate tokens
        for token_idx in range(max_tokens):
            # Start with input embedding (first device)
            current_tensor = input_ids if token_idx == 0 else mx.array([tokens[-1]])
            
            # Process through pipeline
            for conn in sorted_connections:
                request = pb2.ForwardRequest(
                    request_id=f"req_{token_idx}_{conn.device_id}",
                    input_tensor=TensorSerializer.tensor_to_proto(current_tensor),
                    return_cache=True
                )
                
                # Add cache for this device if available
                device_cache_key = f"{conn.device_id}_cache"
                if device_cache_key in cache_dict:
                    for i, cache_tensor in enumerate(cache_dict[device_cache_key]):
                        request.cache[str(i)] = TensorSerializer.tensor_to_proto(cache_tensor)
                
                try:
                    # Forward pass on device
                    response = conn.stub.Forward(request, timeout=self.timeout)
                    
                    # Update current tensor
                    current_tensor = TensorSerializer.proto_to_tensor(response.output_tensor)
                    
                    # Update cache
                    if response.cache:
                        device_cache = []
                        for i in sorted(response.cache.keys()):
                            device_cache.append(TensorSerializer.proto_to_tensor(response.cache[i]))
                        cache_dict[device_cache_key] = device_cache
                    
                except Exception as e:
                    logger.error(f"Forward pass failed on {conn.device_id}: {e}")
                    raise
            
            # Sample next token (current_tensor should be logits from last device)
            if temperature == 0:
                next_token = mx.argmax(current_tensor[:, -1, :], axis=-1).item()
            else:
                logits = current_tensor[:, -1, :] / temperature
                next_token = mx.random.categorical(logits).item()
            
            tokens.append(next_token)
            
            # Check for EOS token (model-specific, using common value)
            if next_token == 2:  # Common EOS token ID
                break
        
        return tokens
    
    def start_health_monitoring(self):
        """Start background health monitoring of connected devices."""
        if self._health_check_thread and self._health_check_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self._stop_health_checks.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Started health monitoring")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self._stop_health_checks.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _health_check_loop(self):
        """Background loop for health checks."""
        while not self._stop_health_checks.is_set():
            for conn in list(self.connections.values()):
                if time.time() - conn.last_health_check >= self._health_check_interval:
                    self._check_device_health(conn)
            
            # Sleep with interruptible wait
            self._stop_health_checks.wait(timeout=1.0)
    
    def _check_device_health(self, conn: DeviceConnection):
        """Check health of a single device."""
        try:
            request = pb2.HealthCheckRequest(include_stats=True)
            response = conn.stub.HealthCheck(request, timeout=5.0)
            
            conn.healthy = response.healthy
            conn.last_health_check = time.time()
            
            if not response.healthy:
                logger.warning(f"Device {conn.device_id} unhealthy: {response.status}")
            
        except Exception as e:
            logger.error(f"Health check failed for {conn.device_id}: {e}")
            conn.healthy = False
    
    def shutdown_device(self, device_id: str, force: bool = False, 
                       grace_period: int = 5) -> bool:
        """Shutdown a connected device."""
        if device_id not in self.connections:
            logger.warning(f"Device {device_id} not connected")
            return False
        
        conn = self.connections[device_id]
        request = pb2.ShutdownRequest(
            force=force,
            grace_period_seconds=grace_period
        )
        
        try:
            response = conn.stub.Shutdown(request, timeout=self.timeout)
            if response.success:
                logger.info(f"Shutdown initiated for {device_id}: {response.message}")
                # Remove connection
                del self.connections[device_id]
                return True
            else:
                logger.error(f"Failed to shutdown {device_id}: {response.message}")
                return False
        except Exception as e:
            logger.error(f"Error shutting down {device_id}: {e}")
            return False
    
    def close(self):
        """Close all connections and clean up resources."""
        logger.info("Closing distributed inference client")
        
        # Stop health monitoring
        self.stop_health_monitoring()
        
        # Close all connections
        for device_id in list(self.connections.keys()):
            try:
                conn = self.connections[device_id]
                conn.channel.close()
                del self.connections[device_id]
            except Exception as e:
                logger.error(f"Error closing connection to {device_id}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Client closed")


class DistributedInferenceOrchestrator:
    """High-level orchestrator for distributed inference."""
    
    def __init__(self, client: DistributedInferenceClient):
        self.client = client
        self.model_name = None
        self.model_provider = None
        self.sharding_plan = None
        self.tokenizer = None
    
    def setup_cluster(self, device_configs: List[Dict[str, Any]],
                     model_name: str, model_provider: str,
                     sharding_plan: ShardingPlan):
        """Set up the distributed cluster."""
        logger.info("Setting up distributed cluster")
        
        # Connect to all devices
        for config in device_configs:
            try:
                self.client.connect_device(
                    device_id=config['device_id'],
                    hostname=config['hostname'],
                    port=config['port']
                )
            except Exception as e:
                logger.error(f"Failed to connect to {config['device_id']}: {e}")
                raise
        
        # Initialize shards
        results = self.client.initialize_shards(model_name, model_provider, sharding_plan)
        
        if not all(results.values()):
            raise RuntimeError("Failed to initialize all shards")
        
        self.model_name = model_name
        self.model_provider = model_provider
        self.sharding_plan = sharding_plan
        
        # Start health monitoring
        self.client.start_health_monitoring()
        
        logger.info("Cluster setup complete")
    
    def generate(self, prompt: str, max_tokens: int = 100,
                temperature: float = 0.7) -> str:
        """Generate text using distributed inference."""
        if not self.tokenizer:
            # Load tokenizer (would be from model wrapper in production)
            from mlx_lm import load
            _, self.tokenizer = load(self.model_name)
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = mx.array(input_ids)
        
        # Run distributed inference
        start_time = time.time()
        output_tokens = self.client.forward_pipeline(
            input_tensor, max_tokens, temperature
        )
        inference_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(output_tokens)
        
        logger.info(f"Generated {len(output_tokens)} tokens in {inference_time:.2f}s "
                   f"({len(output_tokens)/inference_time:.1f} tokens/s)")
        
        return response
    
    def shutdown(self):
        """Shutdown the cluster."""
        logger.info("Shutting down cluster")
        
        # Shutdown all devices
        for device_id in list(self.client.connections.keys()):
            self.client.shutdown_device(device_id, grace_period=5)
        
        # Close client
        self.client.close()


if __name__ == "__main__":
    # Test client
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = DistributedInferenceClient()
    
    try:
        # Connect to local test server
        conn = client.connect_device("test_device", "localhost", 50051)
        
        # Get capabilities
        caps = client.get_device_capabilities("test_device")
        print(f"Device capabilities: {caps}")
        
    finally:
        client.close()
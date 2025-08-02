"""
Coordinator device implementation for distributed MLX inference.

This module implements the coordinator device that manages distributed inference
orchestration, worker coordination, and response generation.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from .base_device import BaseDevice, DeviceState
from ..core.config import ClusterConfig, DeviceConfig, DeviceRole
from ..communication.connection_pool import ConnectionPool
from ..communication.grpc_client import ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for distributed inference."""
    request_id: str
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    repetition_penalty: float = 1.1


@dataclass
class InferenceResponse:
    """Response from distributed inference."""
    request_id: str
    content: str
    tokens_generated: int
    total_time_ms: float
    device_times: Dict[str, float]


class CoordinatorDevice(BaseDevice):
    """
    Coordinator device that orchestrates distributed inference across workers.
    
    The coordinator device is responsible for:
    - Managing connections to worker devices
    - Orchestrating distributed forward passes
    - Generating responses through iterative processing
    - Handling request routing and load balancing
    """
    
    def __init__(self, config: ClusterConfig, device_config: DeviceConfig):
        """
        Initialize the coordinator device.
        
        Args:
            config: Cluster configuration
            device_config: Configuration for this coordinator device
        """
        if device_config.role != DeviceRole.COORDINATOR:
            raise ValueError(f"Device {device_config.device_id} is not configured as coordinator")
        
        super().__init__(config, device_config)
        
        # Coordinator-specific components
        self.connection_pool: Optional[ConnectionPool] = None
        self._active_requests: Dict[str, float] = {}  # request_id -> start_time
        
        logger.info(f"Initialized coordinator device: {self.device_id}")
    
    async def _initialize_device(self) -> None:
        """Initialize coordinator-specific components."""
        logger.info("Initializing coordinator-specific components...")
        
        # Verify we have the correct role
        if self.role != DeviceRole.COORDINATOR:
            raise RuntimeError("CoordinatorDevice must have COORDINATOR role")
        
        # Initialize any coordinator-specific resources
        await self._setup_coordinator_resources()
        
        logger.info("Coordinator device components initialized")
    
    async def _shutdown_device(self) -> None:
        """Shutdown coordinator-specific components."""
        logger.info("Shutting down coordinator components...")
        
        # Cancel active requests
        self._active_requests.clear()
        
        # Cleanup coordinator resources
        await self._cleanup_coordinator_resources()
        
        logger.info("Coordinator device shutdown completed")
    
    async def _initialize_communication(self) -> None:
        """Initialize communication with worker devices."""
        logger.info("Initializing coordinator communication...")
        
        # Create connection pool for worker communication
        self.connection_pool = ConnectionPool(self.config, self.device_id)
        
        # Verify all workers are healthy
        await self._verify_workers()
        
        logger.info("Coordinator communication initialized")
    
    async def _cleanup_communication(self) -> None:
        """Cleanup communication resources."""
        logger.info("Cleaning up coordinator communication...")
        
        if self.connection_pool:
            # Close all connections
            await self.connection_pool.close_all()
            self.connection_pool = None
        
        logger.info("Coordinator communication cleanup completed")
    
    async def _verify_workers(self) -> None:
        """Verify all worker devices are healthy and reachable."""
        workers = self.config.get_workers()
        
        if not workers:
            logger.warning("No worker devices configured")
            return
        
        logger.info(f"Verifying {len(workers)} worker devices...")
        
        for worker in workers:
            try:
                # Use async get_connection instead of legacy get_client
                connection = await self.connection_pool.get_connection(worker.device_id)
                if not connection:
                    raise RuntimeError(f"No connection to worker {worker.device_id}")
                
                try:
                    health = connection.client.health_check()
                    if not health.get('healthy'):
                        raise RuntimeError(f"Worker {worker.device_id} is not healthy")
                    
                    logger.info(f"Worker {worker.device_id} is healthy")
                finally:
                    # Release the connection back to the pool
                    self.connection_pool.release_connection(worker.device_id, connection)
                
            except Exception as e:
                logger.error(f"Worker verification failed for {worker.device_id}: {e}")
                raise
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an inference request using distributed pipeline.
        
        Args:
            request_data: Dictionary containing request information
            
        Returns:
            Dictionary containing response data
        """
        # Convert request data to InferenceRequest
        request = InferenceRequest(
            request_id=request_data.get('request_id', str(uuid.uuid4())),
            messages=request_data.get('messages', []),
            max_tokens=request_data.get('max_tokens', 512),
            temperature=request_data.get('temperature', 0.7),
            top_p=request_data.get('top_p', 1.0),
            repetition_penalty=request_data.get('repetition_penalty', 1.1)
        )
        
        return await self.process_inference_request(request)
    
    async def process_inference_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """
        Process an inference request across distributed devices.
        
        Args:
            request: Inference request object
            
        Returns:
            Dictionary containing response data
        """
        if not self.is_ready:
            raise RuntimeError("Coordinator device is not ready")
        
        start_time = self.record_request_start()
        request_start_time = time.time()
        device_times = {}
        
        # Track active request
        self._active_requests[request.request_id] = request_start_time
        
        try:
            async with self._state_lock:
                self._state = DeviceState.BUSY
            
            logger.info(f"Processing inference request {request.request_id}")
            
            # Format messages for the model
            prompt = self._format_messages(request.messages)
            
            # Tokenize input
            input_ids = mx.array(self.tokenizer.encode(prompt))
            
            # Process through distributed pipeline
            output = await self._distributed_forward(
                input_ids, 
                request.request_id,
                device_times
            )
            
            # Generate response
            generated_text = await self._generate_response(
                output,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.repetition_penalty
            )
            
            # Count generated tokens
            tokens_generated = len(self.tokenizer.encode(generated_text)) if generated_text else 0
            
            total_time = (time.time() - request_start_time) * 1000
            
            response_data = {
                'request_id': request.request_id,
                'content': generated_text,
                'tokens_generated': tokens_generated,
                'total_time_ms': total_time,
                'device_times': device_times
            }
            
            self.record_request_completion(start_time, success=True)
            logger.info(f"Completed inference request {request.request_id} in {total_time:.2f}ms")
            
            return response_data
            
        except Exception as e:
            self.record_request_completion(start_time, success=False)
            logger.error(f"Error processing request {request.request_id}: {e}")
            raise
        
        finally:
            # Remove from active requests
            self._active_requests.pop(request.request_id, None)
            
            async with self._state_lock:
                self._state = DeviceState.READY
    
    async def _distributed_forward(self, 
                                  input_ids: mx.array, 
                                  request_id: str,
                                  device_times: Dict[str, float]) -> mx.array:
        """
        Perform distributed forward pass across devices.
        
        Args:
            input_ids: Input token IDs
            request_id: Request identifier
            device_times: Dict to store timing information
            
        Returns:
            Final hidden states
        """
        # Process embeddings on coordinator
        start = time.time()
        hidden_states = self.layer_processor.process_embedding(input_ids)
        mx.eval(hidden_states)
        device_times['embedding'] = (time.time() - start) * 1000
        
        # Process coordinator's layers
        start = time.time()
        coordinator_layers = self.config.model.get_device_layers(self.device_id)
        if coordinator_layers:
            hidden_states = self.layer_processor.process(
                hidden_states,
                coordinator_layers,
                {}
            )
            mx.eval(hidden_states)
        device_times[self.device_id] = (time.time() - start) * 1000
        
        # Process through worker devices in order
        current_device_id = self.device_id
        
        while True:
            # Get next device
            next_client = self.connection_pool.get_next_device_client(current_device_id)
            if not next_client:
                break
            
            next_device = self.config.get_device_by_rank(
                self.config.get_device(current_device_id).rank + 1
            )
            
            # Get layers for next device
            next_layers = self.config.model.get_device_layers(next_device.device_id)
            if not next_layers:
                logger.warning(f"No layers assigned to {next_device.device_id}")
                break
            
            # Send to next device
            logger.debug(f"Sending to {next_device.device_id} for layers {next_layers}")
            
            start = time.time()
            result = next_client.process_layers(
                hidden_states,
                next_layers,
                request_id,
                {}
            )
            device_times[next_device.device_id] = result.processing_time_ms
            
            hidden_states = result.output_tensor
            current_device_id = next_device.device_id
        
        return hidden_states
    
    async def _generate_response(self,
                                initial_hidden_states: mx.array,
                                max_tokens: int,
                                temperature: float,
                                top_p: float,
                                repetition_penalty: float) -> str:
        """Generate text response iteratively through distributed pipeline."""
        # Create sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        
        # Generate tokens iteratively
        generated_ids = []
        current_hidden_states = initial_hidden_states
        
        for i in range(max_tokens):
            # Process through output layers to get logits
            logits = self.layer_processor.process_output(current_hidden_states)
            
            # Sample next token
            next_token = sampler(logits[:, -1:, :])  # Keep batch dimension
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            # Check for EOS token
            if token_id == self.tokenizer.eos_token_id:
                logger.debug(f"EOS token reached after {i+1} tokens")
                break
            
            # For next iteration, we need to run the new token through the distributed pipeline
            # Convert token to input tensor
            new_input = mx.array([[token_id]])  # Shape: [1, 1]
            
            # Run through distributed pipeline to get hidden states for next iteration
            device_times = {}  # We don't track these in generation loop
            current_hidden_states = await self._distributed_forward(
                new_input,
                f"generation_step_{i}",
                device_times
            )
            
            logger.debug(f"Generated token {i+1}/{max_tokens}: {token_id}")
        
        # Decode all generated tokens
        if generated_ids:
            generated_text = self.tokenizer.decode(generated_ids)
            logger.info(f"Generated {len(generated_ids)} tokens")
            return generated_text
        else:
            logger.warning("No tokens generated")
            return ""
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        # Simple formatting - can be customized per model
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "
        return prompt
    
    async def _setup_coordinator_resources(self) -> None:
        """Setup coordinator-specific resources."""
        # Initialize any coordinator-specific resources here
        # For example, request queues, load balancing state, etc.
        pass
    
    async def _cleanup_coordinator_resources(self) -> None:
        """Cleanup coordinator-specific resources."""
        # Cleanup coordinator-specific resources here
        pass
    
    def get_active_requests(self) -> Dict[str, float]:
        """Get currently active requests and their start times."""
        return self._active_requests.copy()
    
    def get_worker_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all worker devices."""
        if not self.connection_pool:
            return {}
        
        worker_health = {}
        workers = self.config.get_workers()
        
        for worker in workers:
            try:
                client = self.connection_pool.get_client(worker.device_id)
                if client:
                    health = client.health_check()
                    worker_health[worker.device_id] = health
                else:
                    worker_health[worker.device_id] = {
                        'healthy': False,
                        'error': 'No connection'
                    }
            except Exception as e:
                worker_health[worker.device_id] = {
                    'healthy': False,
                    'error': str(e)
                }
        
        return worker_health
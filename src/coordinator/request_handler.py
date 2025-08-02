"""
Request handler for processing distributed inference requests.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import mlx.core as mx

from ..core.config import ClusterConfig
from ..model.inference import LayerProcessor
from ..communication.connection_pool import ConnectionPool
from .inference_pipeline import DistributedInferencePipeline
from .generation_engine import DistributedGenerationEngine, GenerationConfig, GenerationResult
from .device_coordinator import DeviceCoordinator

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
    stream: bool = False
    stop_sequences: Optional[List[str]] = None


@dataclass
class InferenceResponse:
    """Response from distributed inference."""
    request_id: str
    content: str
    tokens_generated: int
    total_time_ms: float
    device_times: Dict[str, float]
    generation_speed: float
    stop_reason: str


class RequestHandler:
    """Handles distributed inference requests with clean separation of concerns."""
    
    def __init__(self, 
                 config: ClusterConfig,
                 layer_processor: LayerProcessor,
                 connection_pool: ConnectionPool,
                 tokenizer,
                 device_coordinator: Optional[DeviceCoordinator] = None):
        """
        Initialize request handler.
        
        Args:
            config: Cluster configuration
            layer_processor: Local layer processor
            connection_pool: Pool of gRPC connections
            tokenizer: Model tokenizer
            device_coordinator: Optional device coordinator for health monitoring
        """
        self.config = config
        self.layer_processor = layer_processor
        self.connection_pool = connection_pool
        self.tokenizer = tokenizer
        self.device_config = config.get_local_device()
        
        # Initialize specialized components
        self.inference_pipeline = DistributedInferencePipeline(
            config, layer_processor, connection_pool
        )
        self.generation_engine = DistributedGenerationEngine(
            config, layer_processor, self.inference_pipeline, tokenizer
        )
        self.device_coordinator = device_coordinator
        
        logger.info(f"RequestHandler initialized for {self.device_config.device_id}")
    
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request across distributed devices.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        start_time = time.time()
        
        try:
            # Validate cluster health if coordinator is available
            if self.device_coordinator:
                await self._validate_cluster_health()
            
            # Validate request
            if not self.validate_request(request):
                raise ValueError(f"Invalid request: {request.request_id}")
            
            # Format messages and tokenize
            prompt = self._format_messages(request.messages)
            input_ids = mx.array(self.tokenizer.encode(prompt))
            
            # Process through distributed pipeline to get initial hidden states
            initial_hidden_states, device_times = await self.inference_pipeline.forward_pass(
                input_ids, 
                request.request_id
            )
            
            # Create generation configuration
            generation_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stop_sequences=request.stop_sequences,
                stream=request.stream
            )
            
            # Generate response using generation engine
            generation_result = await self.generation_engine.generate(
                initial_hidden_states,
                generation_config,
                request.request_id
            )
            
            total_time = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                content=generation_result.generated_text,
                tokens_generated=len(generation_result.token_ids),
                total_time_ms=total_time,
                device_times=device_times,
                generation_speed=generation_result.tokens_per_second,
                stop_reason=generation_result.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            raise
    
    async def process_streaming_request(self, request: InferenceRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a streaming inference request.
        
        Args:
            request: Inference request with stream=True
            
        Yields:
            Streaming tokens and metadata
        """
        if not request.stream:
            raise ValueError("Streaming requests must have stream=True")
        
        try:
            # Validate cluster health
            if self.device_coordinator:
                await self._validate_cluster_health()
            
            # Validate request
            if not self.validate_request(request):
                raise ValueError(f"Invalid request: {request.request_id}")
            
            # Format and tokenize
            prompt = self._format_messages(request.messages)
            input_ids = mx.array(self.tokenizer.encode(prompt))
            
            # Get initial hidden states
            initial_hidden_states, device_times = await self.inference_pipeline.forward_pass(
                input_ids, request.request_id
            )
            
            # Create generation config
            generation_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stop_sequences=request.stop_sequences,
                stream=True
            )
            
            # Stream generation
            async for token_data in self.generation_engine.generate_streaming(
                initial_hidden_states, generation_config, request.request_id
            ):
                yield {
                    'request_id': request.request_id,
                    'token': token_data['token'],
                    'token_id': token_data['token_id'],
                    'index': token_data['index'],
                    'partial_text': token_data['partial_text'],
                    'device_times': device_times if token_data['index'] == 0 else {}
                }
                
        except Exception as e:
            logger.error(f"Error in streaming request {request.request_id}: {e}")
            yield {
                'request_id': request.request_id,
                'error': str(e),
                'final': True
            }
    
    async def _validate_cluster_health(self):
        """Validate that the cluster is healthy for processing requests."""
        if not self.device_coordinator:
            return
        
        is_valid, issues = self.device_coordinator.validate_pipeline_health()
        if not is_valid:
            raise RuntimeError(f"Cluster health issues: {'; '.join(issues)}")
    
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
    
    def create_request(self, 
                      messages: List[Dict[str, str]],
                      max_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 1.0,
                      repetition_penalty: float = 1.1,
                      stream: bool = False,
                      stop_sequences: Optional[List[str]] = None) -> InferenceRequest:
        """Create a new inference request with generated ID."""
        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
            stop_sequences=stop_sequences
        )
    
    def validate_request(self, request: InferenceRequest) -> bool:
        """Validate an inference request."""
        if not request.request_id:
            return False
        if not request.messages:
            return False
        if request.max_tokens <= 0:
            return False
        if not (0.0 <= request.temperature <= 2.0):
            return False
        if not (0.0 <= request.top_p <= 1.0):
            return False
        if request.repetition_penalty < 1.0:
            return False
        return True
    
    def get_handler_status(self) -> Dict[str, Any]:
        """Get status information about the request handler."""
        pipeline_info = self.inference_pipeline.get_pipeline_info()
        
        status = {
            'device_id': self.device_config.device_id,
            'pipeline_info': pipeline_info,
            'components': {
                'inference_pipeline': True,
                'generation_engine': True,
                'device_coordinator': self.device_coordinator is not None
            }
        }
        
        if self.device_coordinator:
            status['cluster_status'] = self.device_coordinator.get_cluster_status()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        try:
            # Basic component health
            health_status = {
                'healthy': True,
                'device_id': self.device_config.device_id,
                'components': {
                    'inference_pipeline': True,
                    'generation_engine': True,
                    'connection_pool': self.connection_pool.get_connection_count() > 0
                }
            }
            
            # Check cluster health if coordinator available
            if self.device_coordinator:
                cluster_status = self.device_coordinator.get_cluster_status()
                health_status['cluster_health'] = cluster_status
                health_status['healthy'] = cluster_status['cluster_operational']
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'device_id': self.device_config.device_id
            }
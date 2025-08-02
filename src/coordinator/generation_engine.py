"""
Text generation engine for distributed inference with iterative token generation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from ..core.config import ClusterConfig
from ..model.inference import LayerProcessor
from .inference_pipeline import DistributedInferencePipeline

logger = logging.getLogger(__name__)


class GenerationConfig:
    """Configuration for text generation."""
    
    def __init__(self,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.1,
                 stop_sequences: Optional[List[str]] = None,
                 stream: bool = False):
        """
        Initialize generation configuration.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            stop_sequences: Sequences that trigger early stopping
            stream: Whether to stream tokens as they're generated
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stop_sequences = stop_sequences or []
        self.stream = stream
    
    def validate(self) -> bool:
        """Validate generation configuration."""
        if self.max_tokens <= 0:
            return False
        if not (0.0 <= self.temperature <= 2.0):
            return False
        if not (0.0 <= self.top_p <= 1.0):
            return False
        if self.repetition_penalty < 1.0:
            return False
        return True


class GenerationResult:
    """Result from text generation."""
    
    def __init__(self,
                 generated_text: str,
                 token_ids: List[int],
                 generation_time_ms: float,
                 tokens_per_second: float,
                 stop_reason: str):
        """
        Initialize generation result.
        
        Args:
            generated_text: The generated text
            token_ids: List of generated token IDs
            generation_time_ms: Total generation time in milliseconds
            tokens_per_second: Generation speed
            stop_reason: Reason generation stopped ('max_tokens', 'eos', 'stop_sequence')
        """
        self.generated_text = generated_text
        self.token_ids = token_ids
        self.generation_time_ms = generation_time_ms
        self.tokens_per_second = tokens_per_second
        self.stop_reason = stop_reason


class DistributedGenerationEngine:
    """Engine for distributed text generation with iterative token sampling."""
    
    def __init__(self,
                 config: ClusterConfig,
                 layer_processor: LayerProcessor,
                 inference_pipeline: DistributedInferencePipeline,
                 tokenizer):
        """
        Initialize generation engine.
        
        Args:
            config: Cluster configuration
            layer_processor: Local layer processor for output layers
            inference_pipeline: Distributed inference pipeline
            tokenizer: Model tokenizer
        """
        self.config = config
        self.layer_processor = layer_processor
        self.inference_pipeline = inference_pipeline
        self.tokenizer = tokenizer
        
        logger.info("DistributedGenerationEngine initialized")
    
    async def generate(self,
                      initial_hidden_states: mx.array,
                      generation_config: GenerationConfig,
                      request_id: str,
                      token_callback: Optional[Callable[[str, int], None]] = None) -> GenerationResult:
        """
        Generate text iteratively through distributed pipeline.
        
        Args:
            initial_hidden_states: Initial hidden states from prompt processing
            generation_config: Generation configuration
            request_id: Request identifier for tracking
            token_callback: Optional callback for streaming tokens
            
        Returns:
            Generation result with text, timing, and metadata
        """
        if not generation_config.validate():
            raise ValueError("Invalid generation configuration")
        
        start_time = time.time()
        
        # Create sampler
        sampler = make_sampler(
            temp=generation_config.temperature,
            top_p=generation_config.top_p
        )
        
        # Initialize generation state
        generated_ids = []
        current_hidden_states = initial_hidden_states
        stop_reason = "max_tokens"
        
        logger.debug(f"Starting generation for request {request_id}")
        
        # Generate tokens iteratively
        for token_idx in range(generation_config.max_tokens):
            try:
                # Generate next token
                next_token_id, current_hidden_states = await self._generate_next_token(
                    current_hidden_states,
                    sampler,
                    request_id,
                    token_idx
                )
                
                generated_ids.append(next_token_id)
                
                # Check for stopping conditions
                stop_reason, should_stop = self._check_stopping_conditions(
                    next_token_id,
                    generated_ids,
                    generation_config
                )
                
                # Call token callback for streaming
                if token_callback:
                    token_text = self.tokenizer.decode([next_token_id])
                    token_callback(token_text, token_idx)
                
                if should_stop:
                    logger.debug(f"Generation stopped: {stop_reason} after {token_idx + 1} tokens")
                    break
                    
                logger.debug(f"Generated token {token_idx + 1}/{generation_config.max_tokens}: {next_token_id}")
                
            except Exception as e:
                logger.error(f"Error generating token {token_idx}: {e}")
                stop_reason = "error"
                break
        
        # Calculate timing metrics
        generation_time = (time.time() - start_time) * 1000
        tokens_per_second = len(generated_ids) / (generation_time / 1000) if generation_time > 0 else 0
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids) if generated_ids else ""
        
        logger.info(f"Generated {len(generated_ids)} tokens in {generation_time:.1f}ms ({tokens_per_second:.1f} tokens/s)")
        
        return GenerationResult(
            generated_text=generated_text,
            token_ids=generated_ids,
            generation_time_ms=generation_time,
            tokens_per_second=tokens_per_second,
            stop_reason=stop_reason
        )
    
    async def _generate_next_token(self,
                                  hidden_states: mx.array,
                                  sampler: Callable,
                                  request_id: str,
                                  token_idx: int) -> tuple[int, mx.array]:
        """
        Generate the next token in the sequence.
        
        Args:
            hidden_states: Current hidden states
            sampler: Token sampling function
            request_id: Request identifier
            token_idx: Current token index
            
        Returns:
            Tuple of (next_token_id, new_hidden_states)
        """
        # Process through output layers to get logits
        logits = self.layer_processor.process_output(hidden_states)
        
        # Sample next token from logits
        next_token = sampler(logits[:, -1:, :])  # Keep batch dimension
        next_token_id = next_token.item()
        
        # For next iteration, process the new token through the distributed pipeline
        if token_idx < 511:  # Prevent infinite generation
            # Convert token to input tensor
            new_input = mx.array([[next_token_id]])  # Shape: [1, 1]
            
            # Run through distributed pipeline for next iteration
            new_hidden_states, _ = await self.inference_pipeline.forward_pass(
                new_input,
                f"{request_id}_gen_{token_idx}"
            )
        else:
            new_hidden_states = hidden_states  # Use current states to prevent errors
        
        return next_token_id, new_hidden_states
    
    def _check_stopping_conditions(self,
                                  token_id: int,
                                  generated_ids: List[int],
                                  config: GenerationConfig) -> tuple[str, bool]:
        """
        Check if generation should stop.
        
        Args:
            token_id: Latest generated token ID
            generated_ids: All generated token IDs so far
            config: Generation configuration
            
        Returns:
            Tuple of (stop_reason, should_stop)
        """
        # Check for EOS token
        if token_id == self.tokenizer.eos_token_id:
            return "eos", True
        
        # Check for stop sequences
        if config.stop_sequences:
            generated_text = self.tokenizer.decode(generated_ids)
            for stop_seq in config.stop_sequences:
                if stop_seq in generated_text:
                    return "stop_sequence", True
        
        # Check max tokens
        if len(generated_ids) >= config.max_tokens:
            return "max_tokens", True
        
        return "continuing", False
    
    async def generate_streaming(self,
                               initial_hidden_states: mx.array,
                               generation_config: GenerationConfig,
                               request_id: str):
        """
        Generate text with streaming support (async generator).
        
        Args:
            initial_hidden_states: Initial hidden states from prompt processing
            generation_config: Generation configuration with stream=True
            request_id: Request identifier
            
        Yields:
            Individual tokens as they're generated
        """
        if not generation_config.stream:
            raise ValueError("Streaming generation requires stream=True in config")
        
        # Create sampler
        sampler = make_sampler(
            temp=generation_config.temperature,
            top_p=generation_config.top_p
        )
        
        # Initialize state
        generated_ids = []
        current_hidden_states = initial_hidden_states
        
        # Generate tokens iteratively
        for token_idx in range(generation_config.max_tokens):
            try:
                # Generate next token
                next_token_id, current_hidden_states = await self._generate_next_token(
                    current_hidden_states,
                    sampler,
                    request_id,
                    token_idx
                )
                
                generated_ids.append(next_token_id)
                
                # Yield token
                token_text = self.tokenizer.decode([next_token_id])
                yield {
                    'token': token_text,
                    'token_id': next_token_id,
                    'index': token_idx,
                    'partial_text': self.tokenizer.decode(generated_ids)
                }
                
                # Check stopping conditions
                _, should_stop = self._check_stopping_conditions(
                    next_token_id,
                    generated_ids,
                    generation_config
                )
                
                if should_stop:
                    break
                    
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                break
    
    def estimate_generation_time(self, 
                               token_count: int, 
                               batch_size: int = 1) -> float:
        """
        Estimate generation time based on token count and system performance.
        
        Args:
            token_count: Number of tokens to generate
            batch_size: Batch size for generation
            
        Returns:
            Estimated time in seconds
        """
        # This is a simplified estimation - real implementation would use
        # performance metrics from previous generations
        base_latency_ms = 50  # Base latency per token
        pipeline_overhead_ms = 10  # Overhead per pipeline pass
        
        estimated_ms = token_count * (base_latency_ms + pipeline_overhead_ms)
        return estimated_ms / 1000.0
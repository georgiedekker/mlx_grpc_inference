#!/usr/bin/env python3
"""
Batch processing for multiple concurrent requests in MLX inference.
Efficiently handles multiple prompts in parallel.
"""
import asyncio
import time
import mlx.core as mx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Single request in a batch."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    timestamp: float
    future: asyncio.Future
    
    
@dataclass
class BatchResult:
    """Result for a single request in batch."""
    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time: float


class BatchProcessor:
    """
    Handles batching of multiple inference requests for efficient processing.
    Accumulates requests and processes them together when batch is ready.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 4,
        max_wait_time: float = 0.1,
        max_sequence_length: int = 2048
    ):
        """
        Initialize batch processor.
        
        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance
            max_batch_size: Maximum requests to batch together
            max_wait_time: Maximum time to wait for batch to fill (seconds)
            max_sequence_length: Maximum sequence length for padding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_sequence_length = max_sequence_length
        
        # Request queue
        self.request_queue: deque[BatchRequest] = deque()
        self._queue_lock = asyncio.Lock()
        
        # Processing state
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_requests_processed = 0
        self.total_batches_processed = 0
        self.average_batch_size = 0.0
        
    async def start(self):
        """Start the batch processor."""
        if not self._processing:
            self._processing = True
            self._processor_task = asyncio.create_task(self._process_loop())
            logger.info(f"Batch processor started (max_batch={self.max_batch_size}, max_wait={self.max_wait_time}s)")
    
    async def stop(self):
        """Stop the batch processor."""
        self._processing = False
        if self._processor_task:
            await self._processor_task
        logger.info("Batch processor stopped")
    
    async def submit_request(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> BatchResult:
        """
        Submit a request for batch processing.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            BatchResult with generated text and metrics
        """
        request_id = str(uuid.uuid4())[:8]
        future = asyncio.Future()
        
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timestamp=time.time(),
            future=future
        )
        
        async with self._queue_lock:
            self.request_queue.append(request)
            logger.debug(f"Request {request_id} added to queue (queue_size={len(self.request_queue)})")
        
        # Wait for result
        result = await future
        return result
    
    async def _process_loop(self):
        """Main processing loop for batching requests."""
        while self._processing:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if batch:
                    # Process batch
                    await self._process_batch(batch)
                else:
                    # No requests, wait a bit
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """
        Collect requests into a batch.
        
        Returns batch when:
        - Max batch size is reached
        - Max wait time is exceeded
        - Queue is empty after processing
        """
        batch = []
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            async with self._queue_lock:
                if self.request_queue:
                    batch.append(self.request_queue.popleft())
                else:
                    # No more requests
                    break
            
            # Check if we've waited too long
            if batch and (time.time() - start_time) > self.max_wait_time:
                break
            
            # Small delay to allow more requests to arrive
            if len(batch) < self.max_batch_size:
                await asyncio.sleep(0.001)
        
        if batch:
            logger.info(f"Collected batch of {len(batch)} requests")
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """
        Process a batch of requests together.
        
        Args:
            batch: List of BatchRequest objects
        """
        start_time = time.time()
        batch_size = len(batch)
        
        try:
            # Tokenize all prompts
            all_prompts = [req.prompt for req in batch]
            tokenized_prompts = [self.tokenizer.encode(p) for p in all_prompts]
            prompt_lengths = [len(tokens) for tokens in tokenized_prompts]
            
            # Pad to same length for batching
            max_prompt_length = max(prompt_lengths)
            padded_prompts = []
            attention_masks = []
            
            for tokens, length in zip(tokenized_prompts, prompt_lengths):
                # Pad with tokenizer's pad token (usually 0)
                padding_length = max_prompt_length - length
                padded_tokens = tokens + [0] * padding_length
                padded_prompts.append(padded_tokens)
                
                # Create attention mask (1 for real tokens, 0 for padding)
                mask = [1] * length + [0] * padding_length
                attention_masks.append(mask)
            
            # Convert to MLX arrays
            input_ids = mx.array(padded_prompts)
            attention_mask = mx.array(attention_masks)
            
            logger.info(f"Processing batch: {batch_size} requests, max_length={max_prompt_length}")
            
            # Generate for all prompts in batch
            # Note: This is simplified - actual batch generation would need more work
            results = await self._batch_generate(
                input_ids,
                attention_mask,
                batch,
                prompt_lengths
            )
            
            # Update statistics
            self.total_requests_processed += batch_size
            self.total_batches_processed += 1
            self.average_batch_size = (
                (self.average_batch_size * (self.total_batches_processed - 1) + batch_size)
                / self.total_batches_processed
            )
            
            batch_time = time.time() - start_time
            logger.info(
                f"Batch processed: {batch_size} requests in {batch_time:.2f}s "
                f"({batch_size/batch_time:.1f} req/s)"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set error on all futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _batch_generate(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        requests: List[BatchRequest],
        prompt_lengths: List[int]
    ):
        """
        Generate text for a batch of inputs.
        
        This is a simplified version - real batch generation would need:
        - Proper handling of different sequence lengths
        - Dynamic batching during generation
        - Efficient KV cache management for batch
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        
        batch_size = len(requests)
        
        # For now, process each request individually
        # (True batch processing would require more complex implementation)
        for i, req in enumerate(requests):
            try:
                start_time = time.time()
                
                # Create sampler for this request
                sampler = make_sampler(temp=req.temperature)
                
                # Generate text
                result = generate(
                    self.model,
                    self.tokenizer,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    sampler=sampler,
                    verbose=False
                )
                
                generation_time = time.time() - start_time
                
                # Extract generated text (remove prompt)
                if result.startswith(req.prompt):
                    generated_text = result[len(req.prompt):]
                else:
                    generated_text = result
                
                # Create result
                completion_tokens = len(self.tokenizer.encode(generated_text))
                batch_result = BatchResult(
                    request_id=req.request_id,
                    generated_text=generated_text,
                    prompt_tokens=prompt_lengths[i],
                    completion_tokens=completion_tokens,
                    generation_time=generation_time
                )
                
                # Set result on future
                req.future.set_result(batch_result)
                
            except Exception as e:
                logger.error(f"Error generating for request {req.request_id}: {e}")
                req.future.set_exception(e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            "total_requests": self.total_requests_processed,
            "total_batches": self.total_batches_processed,
            "average_batch_size": round(self.average_batch_size, 2),
            "queue_size": len(self.request_queue),
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time
        }


class DynamicBatchProcessor(BatchProcessor):
    """
    Advanced batch processor with dynamic batching.
    Adjusts batch size and timing based on load.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Dynamic adjustment parameters
        self.min_batch_size = 1
        self.target_latency = 100  # Target latency in ms
        self.load_history = deque(maxlen=100)
        
    async def _adjust_parameters(self):
        """Dynamically adjust batching parameters based on load."""
        if len(self.load_history) < 10:
            return
        
        # Calculate average load
        avg_queue_size = np.mean([h['queue_size'] for h in self.load_history])
        avg_latency = np.mean([h['latency'] for h in self.load_history])
        
        # Adjust batch size based on queue size
        if avg_queue_size > self.max_batch_size * 2:
            # High load - increase batch size
            self.max_batch_size = min(self.max_batch_size + 1, 8)
            logger.info(f"Increased batch size to {self.max_batch_size} due to high load")
        elif avg_queue_size < self.max_batch_size / 2 and self.max_batch_size > 2:
            # Low load - decrease batch size for lower latency
            self.max_batch_size = max(self.max_batch_size - 1, 2)
            logger.info(f"Decreased batch size to {self.max_batch_size} due to low load")
        
        # Adjust wait time based on latency
        if avg_latency > self.target_latency * 1.5:
            # Latency too high - reduce wait time
            self.max_wait_time = max(self.max_wait_time * 0.9, 0.01)
            logger.info(f"Reduced wait time to {self.max_wait_time:.3f}s due to high latency")
        elif avg_latency < self.target_latency * 0.5:
            # Latency very low - can afford to wait longer for better batching
            self.max_wait_time = min(self.max_wait_time * 1.1, 0.5)
            logger.info(f"Increased wait time to {self.max_wait_time:.3f}s due to low latency")
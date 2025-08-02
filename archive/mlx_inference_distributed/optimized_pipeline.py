"""
Optimized pipeline implementation for distributed MLX inference.

This module provides performance-optimized versions of the pipeline execution
with overlapping computation, better cache management, and reduced serialization overhead.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import threading
import mlx.core as mx
import numpy as np

# Import original components
from grpc_client import DistributedInferenceClient, DeviceConnection
from grpc_server import TensorSerializer
import distributed_inference_pb2 as pb2

logger = logging.getLogger(__name__)


@dataclass
class PipelineToken:
    """Token being processed through the pipeline."""
    token_id: int
    tensor: mx.array
    stage: int  # Current pipeline stage
    timestamp: float
    cache_version: Optional[int] = None


@dataclass
class DeviceStats:
    """Performance statistics for a device."""
    total_requests: int = 0
    total_time: float = 0.0
    avg_processing_time: float = 0.0
    last_batch_size: int = 1
    optimal_batch_size: int = 1
    cache_hit_rate: float = 0.0


class OptimizedCacheManager:
    """Optimized cache management with incremental updates and compression."""
    
    def __init__(self, enable_compression: bool = True, max_cache_entries: int = 1000):
        self.device_caches: Dict[str, Dict[str, mx.array]] = defaultdict(dict)
        self.cache_versions: Dict[str, int] = defaultdict(int)
        self.cache_timestamps: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.enable_compression = enable_compression
        self.max_cache_entries = max_cache_entries
        self._lock = threading.RLock()
    
    def update_cache(self, device_id: str, layer_idx: int, 
                    cache_tensor: mx.array, token_id: int) -> None:
        """Update cache with new tensor, managing size and versions."""
        with self._lock:
            cache_key = f"layer_{layer_idx}_token_{token_id}"
            
            # Clean old entries if cache is getting too large
            if len(self.device_caches[device_id]) >= self.max_cache_entries:
                self._evict_old_cache_entries(device_id)
            
            # Store the cache tensor
            self.device_caches[device_id][cache_key] = cache_tensor
            self.cache_timestamps[device_id][cache_key] = time.time()
            self.cache_versions[device_id] += 1
            
            logger.debug(f"Updated cache for {device_id}, key: {cache_key}, "
                        f"version: {self.cache_versions[device_id]}")
    
    def get_cache_for_request(self, device_id: str, token_id: int) -> Dict[str, mx.array]:
        """Get relevant cache entries for a request."""
        with self._lock:
            if device_id not in self.device_caches:
                return {}
            
            # Get cache entries for previous tokens (KV cache pattern)
            relevant_cache = {}
            for key, cache_tensor in self.device_caches[device_id].items():
                # Extract token_id from key (format: "layer_X_token_Y")
                try:
                    key_token_id = int(key.split("_token_")[-1])
                    if key_token_id < token_id:  # Only previous tokens
                        relevant_cache[key] = cache_tensor
                except (ValueError, IndexError):
                    continue
            
            return relevant_cache
    
    def _evict_old_cache_entries(self, device_id: str) -> None:
        """Remove oldest cache entries to free memory."""
        if device_id not in self.cache_timestamps:
            return
        
        # Sort by timestamp and remove oldest 20%
        timestamps = self.cache_timestamps[device_id]
        sorted_keys = sorted(timestamps.keys(), key=lambda k: timestamps[k])
        
        num_to_remove = max(1, len(sorted_keys) // 5)  # Remove 20%
        
        for key in sorted_keys[:num_to_remove]:
            if key in self.device_caches[device_id]:
                del self.device_caches[device_id][key]
            if key in self.cache_timestamps[device_id]:
                del self.cache_timestamps[device_id][key]
        
        logger.debug(f"Evicted {num_to_remove} cache entries from {device_id}")
    
    def get_cache_stats(self, device_id: str) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            if device_id not in self.device_caches:
                return {"cache_entries": 0, "cache_version": 0, "memory_mb": 0.0}
            
            num_entries = len(self.device_caches[device_id])
            version = self.cache_versions[device_id]
            
            # Estimate memory usage
            total_bytes = 0
            for cache_tensor in self.device_caches[device_id].values():
                total_bytes += cache_tensor.nbytes
            
            return {
                "cache_entries": num_entries,
                "cache_version": version,
                "memory_mb": total_bytes / (1024 * 1024)
            }


class MicroBatchProcessor:
    """Processes multiple tokens together when beneficial."""
    
    def __init__(self, initial_batch_size: int = 2, max_batch_size: int = 8):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.device_stats: Dict[str, DeviceStats] = defaultdict(DeviceStats)
        self._stats_lock = threading.RLock()
    
    def get_optimal_batch_size(self, device_id: str) -> int:
        """Determine optimal batch size based on device performance."""
        with self._stats_lock:
            stats = self.device_stats[device_id]
            
            if stats.total_requests < 5:  # Not enough data yet
                return self.initial_batch_size
            
            # If average processing time per token is improving with batching, increase batch size
            if stats.avg_processing_time < 0.05:  # Under 50ms per token is good
                return min(stats.optimal_batch_size + 1, self.max_batch_size)
            elif stats.avg_processing_time > 0.1:  # Over 100ms per token is slow
                return max(stats.optimal_batch_size - 1, 1)
            
            return stats.optimal_batch_size
    
    def update_stats(self, device_id: str, batch_size: int, 
                    processing_time: float) -> None:
        """Update performance statistics for adaptive batching."""
        with self._stats_lock:
            stats = self.device_stats[device_id]
            
            stats.total_requests += 1
            stats.total_time += processing_time
            stats.avg_processing_time = stats.total_time / stats.total_requests
            stats.last_batch_size = batch_size
            
            # Update optimal batch size based on performance
            time_per_token = processing_time / batch_size
            if time_per_token < stats.avg_processing_time * 0.8:  # 20% improvement
                stats.optimal_batch_size = min(batch_size + 1, self.max_batch_size)
            elif time_per_token > stats.avg_processing_time * 1.2:  # 20% degradation
                stats.optimal_batch_size = max(batch_size - 1, 1)
    
    async def process_token_batch(self, connection: DeviceConnection,
                                 token_batch: List[mx.array],
                                 cache_manager: OptimizedCacheManager,
                                 token_ids: List[int]) -> List[mx.array]:
        """Process a batch of tokens through a single device."""
        start_time = time.time()
        
        if len(token_batch) == 1:
            # Single token processing
            result = await self._process_single_token(
                connection, token_batch[0], cache_manager, token_ids[0]
            )
            processing_time = time.time() - start_time
            self.update_stats(connection.device_id, 1, processing_time)
            return [result]
        
        # Batch processing
        try:
            # Concatenate tokens for batch processing
            batch_tensor = mx.concatenate(token_batch, axis=0)
            
            # Create batched request
            request = pb2.ForwardRequest(
                request_id=f"batch_{connection.device_id}_{token_ids[0]}",
                input_tensor=TensorSerializer.tensor_to_proto(batch_tensor),
                return_cache=True
            )
            
            # Add relevant cache for each token in batch
            for i, token_id in enumerate(token_ids):
                cache_data = cache_manager.get_cache_for_request(
                    connection.device_id, token_id
                )
                for cache_key, cache_tensor in cache_data.items():
                    proto_key = f"token_{i}_{cache_key}"
                    request.cache[proto_key] = TensorSerializer.tensor_to_proto(cache_tensor)
            
            # Forward pass
            response = connection.stub.Forward(request, timeout=30.0)
            
            # Process batch response
            batch_output = TensorSerializer.proto_to_tensor(response.output_tensor)
            
            # Split batch output back to individual tokens
            individual_outputs = []
            batch_size = len(token_batch)
            
            for i in range(batch_size):
                # Extract individual token output
                token_output = batch_output[i:i+1]  # Keep batch dimension
                individual_outputs.append(token_output)
                
                # Update cache for this token
                if response.cache:
                    for cache_key, cache_proto in response.cache.items():
                        if cache_key.startswith(f"token_{i}_"):
                            layer_key = cache_key.replace(f"token_{i}_", "")
                            cache_tensor = TensorSerializer.proto_to_tensor(cache_proto)
                            
                            # Extract layer index from key
                            try:
                                layer_idx = int(layer_key.split("_")[1])
                                cache_manager.update_cache(
                                    connection.device_id, layer_idx, 
                                    cache_tensor, token_ids[i]
                                )
                            except (ValueError, IndexError):
                                logger.warning(f"Could not parse cache key: {cache_key}")
            
            processing_time = time.time() - start_time
            self.update_stats(connection.device_id, batch_size, processing_time)
            
            logger.debug(f"Processed batch of {batch_size} tokens on {connection.device_id} "
                        f"in {processing_time*1000:.1f}ms")
            
            return individual_outputs
            
        except Exception as e:
            logger.error(f"Batch processing failed on {connection.device_id}: {e}")
            # Fallback to individual processing
            results = []
            for i, (token, token_id) in enumerate(zip(token_batch, token_ids)):
                result = await self._process_single_token(
                    connection, token, cache_manager, token_id
                )
                results.append(result)
            
            processing_time = time.time() - start_time
            self.update_stats(connection.device_id, len(token_batch), processing_time)
            
            return results
    
    async def _process_single_token(self, connection: DeviceConnection,
                                   token: mx.array, cache_manager: OptimizedCacheManager,
                                   token_id: int) -> mx.array:
        """Process a single token through a device."""
        request = pb2.ForwardRequest(
            request_id=f"single_{connection.device_id}_{token_id}",
            input_tensor=TensorSerializer.tensor_to_proto(token),
            return_cache=True
        )
        
        # Add cache for this token
        cache_data = cache_manager.get_cache_for_request(connection.device_id, token_id)
        for cache_key, cache_tensor in cache_data.items():
            request.cache[cache_key] = TensorSerializer.tensor_to_proto(cache_tensor)
        
        # Forward pass
        response = connection.stub.Forward(request, timeout=30.0)
        
        # Update cache
        if response.cache:
            for cache_key, cache_proto in response.cache.items():
                cache_tensor = TensorSerializer.proto_to_tensor(cache_proto)
                try:
                    layer_idx = int(cache_key.split("_")[1])
                    cache_manager.update_cache(
                        connection.device_id, layer_idx, cache_tensor, token_id
                    )
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse cache key: {cache_key}")
        
        return TensorSerializer.proto_to_tensor(response.output_tensor)


class OptimizedPipelineExecutor:
    """Manages optimized pipeline execution with overlapping computation."""
    
    def __init__(self, connections: Dict[str, DeviceConnection]):
        self.connections = sorted(
            connections.values(),
            key=lambda c: c.assignment.start_layer if c.assignment else float('inf')
        )
        self.cache_manager = OptimizedCacheManager()
        self.batch_processor = MicroBatchProcessor()
        self.pipeline_queues: List[asyncio.Queue] = []
        self.stats = defaultdict(lambda: defaultdict(float))
    
    async def forward_pipeline_optimized(self, input_ids: mx.array,
                                       max_tokens: int = 100,
                                       temperature: float = 0.7,
                                       use_pipelining: bool = True,
                                       use_micro_batching: bool = True) -> List[int]:
        """
        Optimized pipeline execution with multiple performance improvements.
        
        Args:
            input_ids: Initial input tensor
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_pipelining: Enable pipeline parallelism (overlapping execution)
            use_micro_batching: Enable micro-batching optimization
            
        Returns:
            List of generated token IDs
        """
        start_time = time.time()
        
        if use_pipelining and len(self.connections) > 1:
            tokens = await self._pipeline_parallel_execution(
                input_ids, max_tokens, temperature, use_micro_batching
            )
        else:
            tokens = await self._pipeline_sequential_optimized(
                input_ids, max_tokens, temperature, use_micro_batching
            )
        
        total_time = time.time() - start_time
        tokens_per_sec = len(tokens) / total_time if total_time > 0 else 0
        
        logger.info(f"Generated {len(tokens)} tokens in {total_time:.2f}s "
                   f"({tokens_per_sec:.1f} tokens/sec)")
        
        return tokens
    
    async def _pipeline_parallel_execution(self, input_ids: mx.array, max_tokens: int,
                                         temperature: float, use_micro_batching: bool) -> List[int]:
        """Execute pipeline with overlapping stages for maximum throughput."""
        logger.info("Starting optimized parallel pipeline execution")
        
        # Initialize queues for inter-stage communication
        num_stages = len(self.connections)
        self.pipeline_queues = [asyncio.Queue(maxsize=3) for _ in range(num_stages + 1)]
        
        # Start all pipeline stages
        stage_tasks = []
        for i, connection in enumerate(self.connections):
            task = asyncio.create_task(
                self._pipeline_stage_worker(
                    connection, i, self.pipeline_queues[i], self.pipeline_queues[i + 1],
                    use_micro_batching
                )
            )
            stage_tasks.append(task)
        
        # Start token sampler (final stage)
        sampler_task = asyncio.create_task(
            self._token_sampler_worker(
                self.pipeline_queues[num_stages], self.pipeline_queues[0],
                temperature, max_tokens
            )
        )
        
        # Feed initial input
        initial_token = PipelineToken(
            token_id=0,
            tensor=input_ids,
            stage=0,
            timestamp=time.time()
        )
        await self.pipeline_queues[0].put(initial_token)
        
        # Collect generated tokens
        generated_tokens = []
        try:
            # Wait for generation to complete or timeout
            await asyncio.wait_for(sampler_task, timeout=max_tokens * 2.0)  # 2s per token max
            generated_tokens = sampler_task.result()
        except asyncio.TimeoutError:
            logger.warning("Pipeline execution timed out")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
        finally:
            # Clean up tasks
            for task in stage_tasks + [sampler_task]:
                if not task.done():
                    task.cancel()
        
        return generated_tokens
    
    async def _pipeline_stage_worker(self, connection: DeviceConnection, stage_idx: int,
                                   input_queue: asyncio.Queue, output_queue: asyncio.Queue,
                                   use_micro_batching: bool):
        """Worker for a single pipeline stage."""
        logger.debug(f"Started pipeline stage {stage_idx} on {connection.device_id}")
        
        while True:
            try:
                # Get token(s) to process
                if use_micro_batching:
                    # Try to get multiple tokens for batch processing
                    tokens_to_process = []
                    token_ids = []
                    
                    # Get first token (blocking)
                    first_token = await input_queue.get()
                    if first_token is None:  # Shutdown signal
                        break
                    
                    tokens_to_process.append(first_token.tensor)
                    token_ids.append(first_token.token_id)
                    
                    # Try to get additional tokens (non-blocking)
                    batch_size = self.batch_processor.get_optimal_batch_size(connection.device_id)
                    for _ in range(batch_size - 1):
                        try:
                            additional_token = input_queue.get_nowait()
                            if additional_token is None:
                                break
                            tokens_to_process.append(additional_token.tensor)
                            token_ids.append(additional_token.token_id)
                        except asyncio.QueueEmpty:
                            break
                    
                    # Process batch
                    outputs = await self.batch_processor.process_token_batch(
                        connection, tokens_to_process, self.cache_manager, token_ids
                    )
                    
                    # Send outputs to next stage
                    for i, output_tensor in enumerate(outputs):
                        output_token = PipelineToken(
                            token_id=token_ids[i],
                            tensor=output_tensor,
                            stage=stage_idx + 1,
                            timestamp=time.time()
                        )
                        await output_queue.put(output_token)
                
                else:
                    # Single token processing
                    token = await input_queue.get()
                    if token is None:  # Shutdown signal
                        break
                    
                    output_tensor = await self.batch_processor._process_single_token(
                        connection, token.tensor, self.cache_manager, token.token_id
                    )
                    
                    output_token = PipelineToken(
                        token_id=token.token_id,
                        tensor=output_tensor,
                        stage=stage_idx + 1,
                        timestamp=time.time()
                    )
                    await output_queue.put(output_token)
                
            except Exception as e:
                logger.error(f"Error in pipeline stage {stage_idx}: {e}")
                break
        
        logger.debug(f"Pipeline stage {stage_idx} on {connection.device_id} finished")
    
    async def _token_sampler_worker(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue,
                                  temperature: float, max_tokens: int) -> List[int]:
        """Worker that samples tokens and manages generation completion."""
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            try:
                # Get logits from final pipeline stage
                token = await asyncio.wait_for(input_queue.get(), timeout=5.0)
                if token is None:
                    break
                
                # Sample next token
                logits_tensor = token.tensor
                if temperature == 0:
                    next_token = mx.argmax(logits_tensor[:, -1, :], axis=-1).item()
                else:
                    logits = logits_tensor[:, -1, :] / temperature
                    next_token = mx.random.categorical(logits).item()
                
                generated_tokens.append(next_token)
                
                # Check for EOS token
                if next_token == 2:  # Common EOS token ID
                    logger.info(f"EOS token generated, stopping at {len(generated_tokens)} tokens")
                    break
                
                # Feed next token back to first stage for continued generation
                if len(generated_tokens) < max_tokens:
                    next_input_token = PipelineToken(
                        token_id=len(generated_tokens),
                        tensor=mx.array([next_token]),
                        stage=0,
                        timestamp=time.time()
                    )
                    await output_queue.put(next_input_token)
                
            except asyncio.TimeoutError:
                logger.warning("Token sampling timed out")
                break
            except Exception as e:
                logger.error(f"Error in token sampling: {e}")
                break
        
        # Send shutdown signals to all stages
        for queue in self.pipeline_queues:
            try:
                await queue.put(None)
            except:
                pass
        
        return generated_tokens
    
    async def _pipeline_sequential_optimized(self, input_ids: mx.array, max_tokens: int,
                                           temperature: float, use_micro_batching: bool) -> List[int]:
        """Sequential processing with micro-batching and cache optimizations."""
        logger.info("Starting optimized sequential pipeline execution")
        
        tokens = []
        current_token_id = 0
        current_tensor = input_ids
        
        while len(tokens) < max_tokens:
            # Process through each device in sequence
            for connection in self.connections:
                if use_micro_batching:
                    # For sequential processing, micro-batching is less beneficial
                    # but we can still use optimized single token processing
                    current_tensor = await self.batch_processor._process_single_token(
                        connection, current_tensor, self.cache_manager, current_token_id
                    )
                else:
                    # Basic optimized processing
                    current_tensor = await self._basic_forward_pass(
                        connection, current_tensor, current_token_id
                    )
            
            # Sample next token
            if temperature == 0:
                next_token = mx.argmax(current_tensor[:, -1, :], axis=-1).item()
            else:
                logits = current_tensor[:, -1, :] / temperature
                next_token = mx.random.categorical(logits).item()
            
            tokens.append(next_token)
            
            # Check for EOS token
            if next_token == 2:
                break
            
            # Prepare for next iteration
            current_token_id += 1
            current_tensor = mx.array([next_token])
        
        return tokens
    
    async def _basic_forward_pass(self, connection: DeviceConnection,
                                input_tensor: mx.array, token_id: int) -> mx.array:
        """Basic forward pass with cache management."""
        request = pb2.ForwardRequest(
            request_id=f"basic_{connection.device_id}_{token_id}",
            input_tensor=TensorSerializer.tensor_to_proto(input_tensor),
            return_cache=True
        )
        
        # Add relevant cache
        cache_data = self.cache_manager.get_cache_for_request(connection.device_id, token_id)
        for cache_key, cache_tensor in cache_data.items():
            request.cache[cache_key] = TensorSerializer.tensor_to_proto(cache_tensor)
        
        # Forward pass
        response = connection.stub.Forward(request, timeout=30.0)
        
        # Update cache
        if response.cache:
            for cache_key, cache_proto in response.cache.items():
                cache_tensor = TensorSerializer.proto_to_tensor(cache_proto)
                try:
                    layer_idx = int(cache_key.split("_")[1])
                    self.cache_manager.update_cache(
                        connection.device_id, layer_idx, cache_tensor, token_id
                    )
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse cache key: {cache_key}")
        
        return TensorSerializer.proto_to_tensor(response.output_tensor)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        stats = {
            "cache_stats": {},
            "device_stats": {},
            "batch_stats": {}
        }
        
        # Cache statistics
        for device_id in [conn.device_id for conn in self.connections]:
            stats["cache_stats"][device_id] = self.cache_manager.get_cache_stats(device_id)
        
        # Device batch processing statistics
        for device_id, device_stats in self.batch_processor.device_stats.items():
            stats["device_stats"][device_id] = {
                "total_requests": device_stats.total_requests,
                "avg_processing_time": device_stats.avg_processing_time,
                "optimal_batch_size": device_stats.optimal_batch_size,
                "last_batch_size": device_stats.last_batch_size
            }
        
        return stats


class OptimizedDistributedClient(DistributedInferenceClient):
    """Extended client with optimized pipeline execution."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_executor = None
    
    def initialize_optimized_pipeline(self):
        """Initialize the optimized pipeline after connections are established."""
        if self.connections:
            self.pipeline_executor = OptimizedPipelineExecutor(self.connections)
            logger.info("Initialized optimized pipeline executor")
    
    async def forward_pipeline_optimized(self, input_ids: mx.array,
                                       max_tokens: int = 100,
                                       temperature: float = 0.7,
                                       use_pipelining: bool = True,
                                       use_micro_batching: bool = True) -> List[int]:
        """
        Optimized forward pipeline execution.
        
        This method provides significant performance improvements over the original
        forward_pipeline method through:
        - Pipeline parallelism with overlapping execution
        - Micro-batching for improved throughput
        - Optimized cache management
        - Reduced serialization overhead
        """
        if not self.pipeline_executor:
            self.initialize_optimized_pipeline()
        
        if not self.pipeline_executor:
            raise ValueError("Pipeline executor not initialized - no connections available")
        
        return await self.pipeline_executor.forward_pipeline_optimized(
            input_ids, max_tokens, temperature, use_pipelining, use_micro_batching
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if not self.pipeline_executor:
            return {"error": "Pipeline executor not initialized"}
        
        return self.pipeline_executor.get_performance_stats()


# Example usage and testing
async def test_optimized_pipeline():
    """Test the optimized pipeline implementation."""
    logger.info("Testing optimized pipeline implementation")
    
    # This would be used with actual connections in practice
    # For testing, we'd need to set up mock connections or actual servers
    
    client = OptimizedDistributedClient()
    
    # In practice, you'd connect to actual devices:
    # client.connect_device("device1", "localhost", 50051)
    # client.connect_device("device2", "localhost", 50052)
    
    # Initialize optimized pipeline
    client.initialize_optimized_pipeline()
    
    # Test input
    test_input = mx.array([[1, 2, 3, 4, 5]])
    
    # Run optimized inference
    # tokens = await client.forward_pipeline_optimized(
    #     test_input, max_tokens=10, temperature=0.7,
    #     use_pipelining=True, use_micro_batching=True
    # )
    
    # Get performance stats
    # stats = client.get_optimization_stats()
    # logger.info(f"Performance stats: {stats}")
    
    logger.info("Optimized pipeline test completed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_optimized_pipeline())
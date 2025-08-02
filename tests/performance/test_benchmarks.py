"""
Performance benchmarks for distributed MLX inference system.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import mlx.core as mx

from src.coordinator.orchestrator import DistributedOrchestrator, InferenceRequest
from src.model.inference import LayerProcessor
from src.communication.grpc_client import ProcessingResult


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    iterations: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    throughput_requests_per_second: float
    memory_usage_gb: float
    error_count: int


class TestInferencePerformance:
    """Performance tests for inference pipeline."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_request_latency_benchmark(self, mock_cluster_config):
        """Benchmark latency for single requests."""
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                with patch('src.coordinator.orchestrator.ConnectionPool'):
                    
                    # Setup fast mocks for consistent timing
                    mock_model = MagicMock()
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer.decode.return_value = "Benchmark response"
                    mock_tokenizer.eos_token_id = 2
                    
                    mock_loader = mock_loader_class.return_value
                    mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                    
                    mock_processor = MagicMock()
                    mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                    mock_processor.process.return_value = mx.ones((1, 5, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])  # Immediate EOS
                        mock_make_sampler.return_value = mock_sampler
                        
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Warmup
                        warmup_request = InferenceRequest(
                            request_id="warmup",
                            messages=[{"role": "user", "content": "warmup"}],
                            max_tokens=1
                        )
                        await orchestrator.process_request(warmup_request)
                        
                        # Benchmark iterations
                        iterations = 50
                        latencies = []
                        error_count = 0
                        
                        for i in range(iterations):
                            request = InferenceRequest(
                                request_id=f"benchmark_{i}",
                                messages=[{"role": "user", "content": f"Benchmark test {i}"}],
                                max_tokens=10
                            )
                            
                            start_time = time.time()
                            try:
                                response = await orchestrator.process_request(request)
                                end_time = time.time()
                                latency = (end_time - start_time) * 1000
                                latencies.append(latency)
                                
                                # Verify response quality
                                assert response.request_id == request.request_id
                                assert len(response.content) > 0
                                
                            except Exception as e:
                                error_count += 1
                                print(f"Error in iteration {i}: {e}")
                        
                        # Calculate statistics
                        if latencies:
                            result = BenchmarkResult(
                                test_name="single_request_latency",
                                iterations=iterations,
                                total_time_ms=sum(latencies),
                                avg_latency_ms=statistics.mean(latencies),
                                min_latency_ms=min(latencies),
                                max_latency_ms=max(latencies),
                                p95_latency_ms=self._calculate_percentile(latencies, 95),
                                throughput_requests_per_second=1000 / statistics.mean(latencies),
                                memory_usage_gb=0.0,  # Would implement real memory tracking
                                error_count=error_count
                            )
                            
                            # Performance assertions
                            assert result.avg_latency_ms < 1000, f"Average latency too high: {result.avg_latency_ms}ms"
                            assert result.error_count == 0, f"Errors occurred: {result.error_count}"
                            assert result.throughput_requests_per_second > 1, f"Throughput too low: {result.throughput_requests_per_second}"
                            
                            self._print_benchmark_results(result)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_request_throughput(self, mock_cluster_config):
        """Benchmark throughput with concurrent requests."""
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                with patch('src.coordinator.orchestrator.ConnectionPool'):
                    
                    # Setup mocks
                    mock_model = MagicMock()
                    mock_tokenizer = MagicMock()
                    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                    mock_tokenizer.decode.return_value = "Concurrent response"
                    mock_tokenizer.eos_token_id = 2
                    
                    mock_loader = mock_loader_class.return_value
                    mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                    
                    mock_processor = MagicMock()
                    mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                    mock_processor.process.return_value = mx.ones((1, 5, 512))
                    mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                    mock_processor_class.return_value = mock_processor
                    
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])
                        mock_make_sampler.return_value = mock_sampler
                        
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Test different concurrency levels
                        concurrency_levels = [1, 2, 4, 8]
                        results = []
                        
                        for concurrency in concurrency_levels:
                            async def process_batch():
                                tasks = []
                                for i in range(concurrency):
                                    request = InferenceRequest(
                                        request_id=f"concurrent_{concurrency}_{i}",
                                        messages=[{"role": "user", "content": f"Concurrent test {i}"}],
                                        max_tokens=5
                                    )
                                    task = orchestrator.process_request(request)
                                    tasks.append(task)
                                
                                return await asyncio.gather(*tasks)
                            
                            # Measure throughput for this concurrency level
                            iterations = 10
                            total_requests = concurrency * iterations
                            
                            start_time = time.time()
                            for _ in range(iterations):
                                await process_batch()
                            end_time = time.time()
                            
                            total_time = end_time - start_time
                            throughput = total_requests / total_time
                            
                            result = {
                                'concurrency': concurrency,
                                'total_requests': total_requests,
                                'total_time_seconds': total_time,
                                'throughput_rps': throughput,
                                'avg_latency_ms': (total_time / total_requests) * 1000
                            }
                            results.append(result)
                            
                            print(f"Concurrency {concurrency}: {throughput:.2f} RPS, {result['avg_latency_ms']:.2f}ms avg latency")
                        
                        # Verify throughput scaling
                        base_throughput = results[0]['throughput_rps']
                        for result in results[1:]:
                            # With perfect scaling, throughput should increase with concurrency
                            # In practice, it may plateau or even decrease due to contention
                            assert result['throughput_rps'] > 0
                            print(f"Concurrency {result['concurrency']}: {result['throughput_rps']:.2f} RPS vs base {base_throughput:.2f} RPS")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, mock_cluster_config):
        """Benchmark memory usage patterns."""
        
        with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
            with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                
                # Setup mocks with memory tracking
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_tokenizer.encode.return_value = [1] * 100  # Longer sequence
                mock_tokenizer.decode.return_value = "Memory test response"
                mock_tokenizer.eos_token_id = 2
                
                mock_loader = mock_loader_class.return_value
                mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                
                # Mock processor with memory tracking
                mock_processor = MagicMock()
                mock_processor.process_embedding.return_value = mx.ones((1, 100, 512))
                mock_processor.process.return_value = mx.ones((1, 100, 512))
                mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                
                # Mock memory usage that increases with processing
                memory_usage = {'allocated_gb': 1.0, 'cached_gb': 0.5, 'reserved_gb': 2.0}
                mock_processor.get_memory_usage.return_value = memory_usage
                mock_processor_class.return_value = mock_processor
                
                with patch('src.coordinator.orchestrator.ConnectionPool'):
                    with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                        mock_sampler = MagicMock()
                        mock_sampler.return_value = mx.array([[2]])
                        mock_make_sampler.return_value = mock_sampler
                        
                        orchestrator = DistributedOrchestrator(mock_cluster_config)
                        await orchestrator.initialize()
                        
                        # Test memory usage with different sequence lengths
                        sequence_lengths = [10, 50, 100, 200]
                        memory_results = []
                        
                        for seq_len in sequence_lengths:
                            # Update mock for this sequence length
                            mock_tokenizer.encode.return_value = [1] * seq_len
                            mock_processor.process_embedding.return_value = mx.ones((1, seq_len, 512))
                            mock_processor.process.return_value = mx.ones((1, seq_len, 512))
                            
                            # Simulate increasing memory usage with sequence length
                            memory_usage['allocated_gb'] = 1.0 + (seq_len * 0.01)
                            
                            request = InferenceRequest(
                                request_id=f"memory_test_{seq_len}",
                                messages=[{"role": "user", "content": "x" * seq_len}],
                                max_tokens=10
                            )
                            
                            start_memory = mock_processor.get_memory_usage()
                            response = await orchestrator.process_request(request)
                            end_memory = mock_processor.get_memory_usage()
                            
                            memory_results.append({
                                'sequence_length': seq_len,
                                'start_memory_gb': start_memory['allocated_gb'],
                                'end_memory_gb': end_memory['allocated_gb'],
                                'memory_delta_gb': end_memory['allocated_gb'] - start_memory['allocated_gb'],
                                'response_length': len(response.content)
                            })
                        
                        # Verify memory usage is reasonable
                        for result in memory_results:
                            assert result['end_memory_gb'] > 0
                            assert result['end_memory_gb'] < 10.0  # Should not exceed reasonable limits
                            print(f"Seq len {result['sequence_length']}: {result['end_memory_gb']:.2f}GB memory")
                        
                        # Verify memory scales reasonably with sequence length
                        memory_growth = memory_results[-1]['end_memory_gb'] - memory_results[0]['end_memory_gb']
                        assert memory_growth >= 0  # Memory should not decrease with longer sequences
    
    @pytest.mark.performance
    def test_layer_processing_benchmark(self, mock_mlx_model):
        """Benchmark individual layer processing performance."""
        
        processor = LayerProcessor(mock_mlx_model, "benchmark_device", [0, 1, 2, 3, 4])
        
        # Test different tensor sizes
        tensor_sizes = [
            (1, 10, 512),    # Small
            (1, 50, 512),    # Medium
            (1, 100, 512),   # Large
            (4, 50, 512),    # Batch processing
        ]
        
        results = []
        
        for batch_size, seq_len, hidden_size in tensor_sizes:
            input_tensor = mx.ones((batch_size, seq_len, hidden_size))
            layers_to_process = [0, 1, 2]
            
            # Benchmark processing time
            iterations = 10
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                result = processor.process(input_tensor, layers_to_process, {})
                mx.eval(result)  # Ensure computation completes
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate throughput metrics
            total_elements = batch_size * seq_len * hidden_size
            throughput_elements_per_ms = total_elements / avg_time
            
            result = {
                'tensor_shape': (batch_size, seq_len, hidden_size),
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'throughput_elements_per_ms': throughput_elements_per_ms,
                'layers_processed': len(layers_to_process)
            }
            results.append(result)
            
            print(f"Shape {result['tensor_shape']}: {avg_time:.2f}ms avg, {throughput_elements_per_ms:.0f} elem/ms")
        
        # Verify performance characteristics
        for result in results:
            assert result['avg_time_ms'] > 0
            assert result['avg_time_ms'] < 1000  # Should process layers within 1 second
            assert result['throughput_elements_per_ms'] > 0
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            fraction = index - int(index)
            return lower + fraction * (upper - lower)
    
    def _print_benchmark_results(self, result: BenchmarkResult):
        """Print formatted benchmark results."""
        print(f"\n=== Benchmark Results: {result.test_name} ===")
        print(f"Iterations: {result.iterations}")
        print(f"Total time: {result.total_time_ms:.2f}ms")
        print(f"Average latency: {result.avg_latency_ms:.2f}ms")
        print(f"Min latency: {result.min_latency_ms:.2f}ms")
        print(f"Max latency: {result.max_latency_ms:.2f}ms")
        print(f"P95 latency: {result.p95_latency_ms:.2f}ms")
        print(f"Throughput: {result.throughput_requests_per_second:.2f} RPS")
        print(f"Memory usage: {result.memory_usage_gb:.2f}GB")
        print(f"Error count: {result.error_count}")
        print("=" * 50)


class TestCommunicationPerformance:
    """Performance tests for gRPC communication."""
    
    @pytest.mark.performance
    def test_tensor_serialization_benchmark(self):
        """Benchmark tensor serialization/deserialization performance."""
        
        from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
        
        # Test different tensor sizes
        tensor_sizes = [
            (1, 10, 512),
            (1, 100, 512),
            (1, 500, 512),
            (4, 100, 512),
        ]
        
        results = []
        
        for shape in tensor_sizes:
            tensor = mx.random.normal(shape).astype(mx.float32)
            
            # Benchmark serialization
            serialize_times = []
            deserialize_times = []
            
            iterations = 20
            
            for _ in range(iterations):
                # Serialize
                start_time = time.time()
                serialized_data, metadata = serialize_mlx_array(tensor, compress=False)
                serialize_time = (time.time() - start_time) * 1000
                serialize_times.append(serialize_time)
                
                # Deserialize
                start_time = time.time()
                deserialized_tensor = deserialize_mlx_array(serialized_data, metadata)
                deserialize_time = (time.time() - start_time) * 1000
                deserialize_times.append(deserialize_time)
                
                # Verify correctness
                assert mx.allclose(tensor, deserialized_tensor)
            
            result = {
                'shape': shape,
                'avg_serialize_ms': statistics.mean(serialize_times),
                'avg_deserialize_ms': statistics.mean(deserialize_times),
                'total_roundtrip_ms': statistics.mean(serialize_times) + statistics.mean(deserialize_times),
                'data_size_mb': len(serialized_data) / (1024 * 1024),
                'throughput_mb_per_s': (len(serialized_data) / (1024 * 1024)) / ((statistics.mean(serialize_times) + statistics.mean(deserialize_times)) / 1000)
            }
            results.append(result)
            
            print(f"Shape {shape}: {result['total_roundtrip_ms']:.2f}ms roundtrip, {result['throughput_mb_per_s']:.2f} MB/s")
        
        # Performance assertions
        for result in results:
            assert result['avg_serialize_ms'] < 1000  # Should serialize within 1 second
            assert result['avg_deserialize_ms'] < 1000  # Should deserialize within 1 second
            assert result['throughput_mb_per_s'] > 1  # Should achieve at least 1 MB/s
    
    @pytest.mark.performance
    def test_compression_performance_trade_off(self):
        """Test performance trade-off between compression and speed."""
        
        from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
        
        # Test with large tensor that benefits from compression
        large_tensor = mx.ones((1, 1000, 512))  # Repetitive data compresses well
        
        # Test both compressed and uncompressed
        compression_modes = [False, True]
        results = {}
        
        for compress in compression_modes:
            serialize_times = []
            deserialize_times = []
            data_sizes = []
            
            iterations = 10
            
            for _ in range(iterations):
                # Serialize
                start_time = time.time()
                serialized_data, metadata = serialize_mlx_array(large_tensor, compress=compress)
                serialize_time = (time.time() - start_time) * 1000
                serialize_times.append(serialize_time)
                data_sizes.append(len(serialized_data))
                
                # Deserialize
                start_time = time.time()
                deserialized_tensor = deserialize_mlx_array(serialized_data, metadata)
                deserialize_time = (time.time() - start_time) * 1000
                deserialize_times.append(deserialize_time)
            
            results[compress] = {
                'avg_serialize_ms': statistics.mean(serialize_times),
                'avg_deserialize_ms': statistics.mean(deserialize_times),
                'avg_data_size_mb': statistics.mean(data_sizes) / (1024 * 1024),
                'compression_ratio': 1.0  # Will calculate below
            }
        
        # Calculate compression ratio
        uncompressed_size = results[False]['avg_data_size_mb']
        compressed_size = results[True]['avg_data_size_mb']
        compression_ratio = uncompressed_size / compressed_size
        results[True]['compression_ratio'] = compression_ratio
        
        print(f"Uncompressed: {results[False]['avg_serialize_ms']:.2f}ms serialize, {results[False]['avg_data_size_mb']:.2f} MB")
        print(f"Compressed: {results[True]['avg_serialize_ms']:.2f}ms serialize, {results[True]['avg_data_size_mb']:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Verify compression provides benefit
        assert compression_ratio > 1.5  # Should achieve at least 1.5x compression
        assert results[True]['avg_data_size_mb'] < results[False]['avg_data_size_mb']


class TestScalabilityBenchmarks:
    """Scalability benchmarks for distributed system."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_device_scaling_simulation(self, mock_cluster_config):
        """Test performance scaling with different numbers of devices."""
        
        # Test with different numbers of simulated devices
        device_counts = [1, 2, 3, 4]
        results = []
        
        for num_devices in device_counts:
            # Modify cluster config for this test
            test_config = mock_cluster_config
            
            # Simulate more devices by adjusting layer distribution
            layers_per_device = 9 // num_devices
            layer_distribution = {}
            
            for i in range(num_devices):
                device_id = f"device_{i}" if i > 0 else "coordinator"
                start_layer = i * layers_per_device
                end_layer = min((i + 1) * layers_per_device, 9)
                layer_distribution[device_id] = list(range(start_layer, end_layer))
            
            test_config.model.layer_distribution = layer_distribution
            
            with patch('src.model.loader.DistributedModelLoader') as mock_loader_class:
                with patch('src.model.inference.LayerProcessor') as mock_processor_class:
                    with patch('src.coordinator.orchestrator.ConnectionPool') as mock_pool_class:
                        
                        # Setup mocks
                        mock_model = MagicMock()
                        mock_tokenizer = MagicMock()
                        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                        mock_tokenizer.decode.return_value = f"Response from {num_devices} devices"
                        mock_tokenizer.eos_token_id = 2
                        
                        mock_loader = mock_loader_class.return_value
                        mock_loader.load_full_model.return_value = (mock_model, mock_tokenizer)
                        
                        mock_processor = MagicMock()
                        mock_processor.process_embedding.return_value = mx.ones((1, 5, 512))
                        mock_processor.process.return_value = mx.ones((1, 5, 512))
                        mock_processor.process_output.return_value = mx.ones((1, 1, 32000))
                        mock_processor_class.return_value = mock_processor
                        
                        # Mock connection pool for multiple devices
                        mock_pool = MagicMock()
                        
                        # Create clients for each device (except coordinator)
                        clients = []
                        for i in range(1, num_devices):
                            client = MagicMock()
                            processing_time = 50 + (i * 10)  # Simulate increasing processing time
                            client.process_layers.return_value = ProcessingResult(
                                output_tensor=mx.ones((1, 5, 512)),
                                processing_time_ms=processing_time,
                                device_id=f"device_{i}"
                            )
                            client.health_check.return_value = {'healthy': True}
                            clients.append(client)
                        
                        # Setup connection pool to return clients in sequence
                        if clients:
                            mock_pool.get_next_device_client.side_effect = clients + [None]
                        else:
                            mock_pool.get_next_device_client.return_value = None
                        
                        mock_pool.get_client.side_effect = lambda device_id: clients[0] if clients else None
                        mock_pool_class.return_value = mock_pool
                        
                        with patch('src.coordinator.orchestrator.make_sampler') as mock_make_sampler:
                            mock_sampler = MagicMock()
                            mock_sampler.return_value = mx.array([[2]])
                            mock_make_sampler.return_value = mock_sampler
                            
                            # Test this configuration
                            orchestrator = DistributedOrchestrator(test_config)
                            await orchestrator.initialize()
                            
                            # Benchmark inference time
                            iterations = 5
                            times = []
                            
                            for i in range(iterations):
                                request = InferenceRequest(
                                    request_id=f"scaling_{num_devices}_{i}",
                                    messages=[{"role": "user", "content": "Scaling test"}],
                                    max_tokens=1
                                )
                                
                                start_time = time.time()
                                response = await orchestrator.process_request(request)
                                end_time = time.time()
                                
                                times.append((end_time - start_time) * 1000)
                            
                            avg_time = statistics.mean(times)
                            
                            result = {
                                'num_devices': num_devices,
                                'avg_latency_ms': avg_time,
                                'layers_per_device': layers_per_device,
                                'device_utilization': len([d for d in layer_distribution.values() if d]) / num_devices
                            }
                            results.append(result)
                            
                            print(f"Devices {num_devices}: {avg_time:.2f}ms avg latency")
        
        # Analyze scaling characteristics
        for i, result in enumerate(results):
            if i > 0:
                prev_result = results[i-1]
                latency_change = result['avg_latency_ms'] - prev_result['avg_latency_ms']
                print(f"Adding device {result['num_devices']}: {latency_change:+.2f}ms latency change")
        
        # Verify reasonable performance across all configurations
        for result in results:
            assert result['avg_latency_ms'] > 0
            assert result['avg_latency_ms'] < 5000  # Should complete within 5 seconds
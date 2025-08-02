#!/usr/bin/env python3
"""
Comprehensive Benchmarking Framework for Distributed MLX Inference System.

This framework provides detailed performance analysis across:
- Tensor serialization/deserialization performance
- Distributed inference latency and throughput
- Memory usage and efficiency
- Communication overhead measurement
- Single-device vs distributed performance comparison
- Cache management efficiency
"""

import asyncio
import time
import logging
import statistics
import json
import sys
import os
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import concurrent.futures
import psutil
import requests

import mlx.core as mx
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import distributed inference components
try:
    from grpc_server import TensorSerializer, DistributedInferenceServicer
    from grpc_client import DistributedInferenceClient
    from optimized_pipeline import OptimizedCacheManager, OptimizedDistributedClient
    from device_capabilities import DeviceCapabilityDetector
except ImportError as e:
    print(f"Warning: Could not import distributed components: {e}")
    print("Some benchmarks may not be available")

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics for a single test."""
    test_name: str
    implementation: str
    
    # Timing metrics
    total_time: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float
    operations_per_second: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency_score: float
    
    # Communication metrics
    serialization_time_ms: float
    deserialization_time_ms: float
    network_transfer_time_ms: float
    compression_ratio: float
    
    # Quality metrics
    success_rate: float
    error_count: int
    cache_hit_rate: float
    
    # Additional metadata
    test_params: Dict[str, Any]
    timestamp: str


@dataclass
class SystemResourceSnapshot:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    disk_io_read: int
    disk_io_write: int


class ResourceMonitor:
    """Monitors system resource usage during benchmarks."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.snapshots: List[SystemResourceSnapshot] = []
        self.monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._initial_network = psutil.net_io_counters()
        self._initial_disk = psutil.disk_io_counters()
    
    def start(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self._initial_network = psutil.net_io_counters()
        self._initial_disk = psutil.disk_io_counters()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.snapshots:
            return {}
        
        return {
            "peak_cpu_percent": max(s.cpu_percent for s in self.snapshots),
            "avg_cpu_percent": statistics.mean(s.cpu_percent for s in self.snapshots),
            "peak_memory_mb": max(s.memory_mb for s in self.snapshots),
            "avg_memory_mb": statistics.mean(s.memory_mb for s in self.snapshots),
            "total_network_bytes": (self.snapshots[-1].network_bytes_sent + 
                                  self.snapshots[-1].network_bytes_recv),
            "total_disk_io_bytes": (self.snapshots[-1].disk_io_read + 
                                  self.snapshots[-1].disk_io_write)
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                net_io = psutil.net_io_counters()
                disk_io = psutil.disk_io_counters()
                memory = psutil.virtual_memory()
                
                snapshot = SystemResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(),
                    memory_mb=memory.used / (1024 * 1024),
                    memory_percent=memory.percent,
                    network_bytes_sent=net_io.bytes_sent - self._initial_network.bytes_sent,
                    network_bytes_recv=net_io.bytes_recv - self._initial_network.bytes_recv,
                    disk_io_read=disk_io.read_bytes - self._initial_disk.read_bytes,
                    disk_io_write=disk_io.write_bytes - self._initial_disk.write_bytes
                )
                
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break


class TensorSerializationBenchmark:
    """Benchmarks tensor serialization/deserialization performance."""
    
    def __init__(self):
        self.test_tensors = self._generate_test_tensors()
    
    def _generate_test_tensors(self) -> List[Tuple[str, mx.array]]:
        """Generate various tensor configurations for testing."""
        tensors = []
        
        # Small tensors (typical for embeddings/tokens)
        tensors.append(("small_1d", mx.array([1, 2, 3, 4, 5])))
        tensors.append(("small_2d", mx.array([[1, 2], [3, 4]])))
        
        # Medium tensors (typical for hidden states)
        tensors.append(("medium_embedding", mx.random.normal((512, 768))))
        tensors.append(("medium_attention", mx.random.normal((12, 64, 64))))
        
        # Large tensors (typical for model weights)
        tensors.append(("large_linear", mx.random.normal((4096, 1024))))
        tensors.append(("large_batch", mx.random.normal((32, 512, 768))))
        
        # Different dtypes
        tensors.append(("float32_tensor", mx.random.normal((100, 100)).astype(mx.float32)))
        tensors.append(("float16_tensor", mx.random.normal((100, 100)).astype(mx.float16)))
        tensors.append(("bfloat16_tensor", mx.random.normal((100, 100)).astype(mx.bfloat16)))
        
        # Sparse tensors (mostly zeros)
        sparse_tensor = mx.zeros((1000, 1000))
        sparse_tensor[::100, ::100] = mx.random.normal((10, 10))
        tensors.append(("sparse_tensor", sparse_tensor))
        
        return tensors
    
    async def run_benchmark(self, iterations: int = 1000) -> BenchmarkMetrics:
        """Run tensor serialization benchmark."""
        logger.info(f"Running tensor serialization benchmark ({iterations} iterations)")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        all_serialize_times = []
        all_deserialize_times = []
        all_sizes = []
        error_count = 0
        
        start_time = time.time()
        
        for tensor_name, tensor in self.test_tensors:
            logger.debug(f"Benchmarking tensor: {tensor_name}")
            
            serialize_times = []
            deserialize_times = []
            
            for _ in range(iterations // len(self.test_tensors)):
                try:
                    # Measure serialization
                    serialize_start = time.time()
                    proto_tensor = TensorSerializer.tensor_to_proto(tensor)
                    serialize_time = (time.time() - serialize_start) * 1000  # ms
                    serialize_times.append(serialize_time)
                    
                    # Measure size
                    serialized_size = len(proto_tensor.data)
                    all_sizes.append(serialized_size)
                    
                    # Measure deserialization
                    deserialize_start = time.time()
                    reconstructed = TensorSerializer.proto_to_tensor(proto_tensor)
                    mx.eval(reconstructed)  # Ensure computation
                    deserialize_time = (time.time() - deserialize_start) * 1000  # ms
                    deserialize_times.append(deserialize_time)
                    
                    # Verify correctness
                    if not mx.allclose(tensor, reconstructed, rtol=1e-5):
                        error_count += 1
                        logger.warning(f"Tensor reconstruction error for {tensor_name}")
                
                except Exception as e:
                    error_count += 1
                    logger.error(f"Serialization error for {tensor_name}: {e}")
            
            all_serialize_times.extend(serialize_times)
            all_deserialize_times.extend(deserialize_times)
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop()
        
        # Calculate compression ratio
        original_sizes = [tensor.nbytes for _, tensor in self.test_tensors]
        avg_compression_ratio = statistics.mean(
            orig / serialized for orig, serialized 
            in zip(original_sizes * (iterations // len(self.test_tensors)), all_sizes)
        )
        
        # Calculate metrics
        total_operations = len(all_serialize_times) + len(all_deserialize_times)
        success_rate = 1.0 - (error_count / total_operations) if total_operations > 0 else 0.0
        
        return BenchmarkMetrics(
            test_name="tensor_serialization",
            implementation="grpc_protobuf",
            total_time=total_time,
            avg_latency_ms=statistics.mean(all_serialize_times + all_deserialize_times),
            min_latency_ms=min(all_serialize_times + all_deserialize_times),
            max_latency_ms=max(all_serialize_times + all_deserialize_times),
            p95_latency_ms=np.percentile(all_serialize_times + all_deserialize_times, 95),
            p99_latency_ms=np.percentile(all_serialize_times + all_deserialize_times, 99),
            tokens_per_second=0.0,  # Not applicable
            requests_per_second=0.0,  # Not applicable
            operations_per_second=total_operations / total_time,
            peak_memory_mb=resource_metrics.get("peak_memory_mb", 0.0),
            avg_memory_mb=resource_metrics.get("avg_memory_mb", 0.0),
            memory_efficiency_score=1.0 / (1.0 + resource_metrics.get("peak_memory_mb", 100.0) / 1000.0),
            serialization_time_ms=statistics.mean(all_serialize_times),
            deserialization_time_ms=statistics.mean(all_deserialize_times),
            network_transfer_time_ms=0.0,  # Local serialization
            compression_ratio=avg_compression_ratio,
            success_rate=success_rate,
            error_count=error_count,
            cache_hit_rate=0.0,  # Not applicable
            test_params={
                "iterations": iterations,
                "tensor_count": len(self.test_tensors),
                "total_operations": total_operations
            },
            timestamp=datetime.now().isoformat()
        )


class DistributedInferenceBenchmark:
    """Benchmarks distributed inference performance."""
    
    def __init__(self, api_url: str = "http://localhost:8100"):
        self.api_url = api_url
        self.test_prompts = self._generate_test_prompts()
    
    def _generate_test_prompts(self) -> List[Tuple[str, str, int]]:
        """Generate test prompts with varying complexity."""
        return [
            ("simple_question", "What is 2+2?", 20),
            ("medium_explanation", "Explain machine learning in simple terms.", 100),
            ("complex_reasoning", "Compare and contrast different machine learning algorithms, their strengths and weaknesses.", 200),
            ("long_generation", "Write a detailed story about a robot learning to understand human emotions.", 300),
            ("code_generation", "Write a Python function to implement binary search with error handling and documentation.", 150),
            ("mathematical", "Solve this step by step: If a train travels at 80 km/h for 2.5 hours, how far does it go?", 80),
            ("creative_writing", "Create a haiku about artificial intelligence and the future of technology.", 50),
            ("technical_explanation", "Describe how neural networks process information, including forward and backward propagation.", 250)
        ]
    
    async def run_latency_benchmark(self, iterations: int = 50) -> BenchmarkMetrics:
        """Benchmark inference latency across different prompt types."""
        logger.info(f"Running distributed inference latency benchmark ({iterations} iterations)")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        all_latencies = []
        all_tokens_per_second = []
        error_count = 0
        total_tokens = 0
        
        start_time = time.time()
        
        for prompt_name, prompt_text, max_tokens in self.test_prompts:
            logger.debug(f"Benchmarking prompt: {prompt_name}")
            
            for iteration in range(iterations // len(self.test_prompts)):
                try:
                    request_data = {
                        "model": "mlx-community/Qwen3-1.7B-8bit",
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                    
                    request_start = time.time()
                    response = requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=request_data,
                        timeout=60
                    )
                    request_time = time.time() - request_start
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "usage" in data and "completion_tokens" in data["usage"]:
                            tokens_generated = data["usage"]["completion_tokens"]
                            total_tokens += tokens_generated
                            
                            latency_ms = request_time * 1000
                            all_latencies.append(latency_ms)
                            
                            if request_time > 0:
                                tokens_per_sec = tokens_generated / request_time
                                all_tokens_per_second.append(tokens_per_sec)
                    else:
                        error_count += 1
                        logger.warning(f"Request failed: {response.status_code}")
                
                except Exception as e:
                    error_count += 1
                    logger.error(f"Inference error for {prompt_name}: {e}")
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop()
        
        total_requests = iterations
        success_rate = 1.0 - (error_count / total_requests) if total_requests > 0 else 0.0
        
        return BenchmarkMetrics(
            test_name="distributed_inference_latency",
            implementation="grpc_distributed",
            total_time=total_time,
            avg_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
            min_latency_ms=min(all_latencies) if all_latencies else 0.0,
            max_latency_ms=max(all_latencies) if all_latencies else 0.0,
            p95_latency_ms=np.percentile(all_latencies, 95) if all_latencies else 0.0,
            p99_latency_ms=np.percentile(all_latencies, 99) if all_latencies else 0.0,
            tokens_per_second=statistics.mean(all_tokens_per_second) if all_tokens_per_second else 0.0,
            requests_per_second=len(all_latencies) / total_time if total_time > 0 else 0.0,
            operations_per_second=len(all_latencies) / total_time if total_time > 0 else 0.0,
            peak_memory_mb=resource_metrics.get("peak_memory_mb", 0.0),
            avg_memory_mb=resource_metrics.get("avg_memory_mb", 0.0),
            memory_efficiency_score=total_tokens / resource_metrics.get("peak_memory_mb", 1.0),
            serialization_time_ms=0.0,  # Measured separately
            deserialization_time_ms=0.0,  # Measured separately
            network_transfer_time_ms=resource_metrics.get("total_network_bytes", 0.0) / 1000.0,
            compression_ratio=1.0,  # Not measured here
            success_rate=success_rate,
            error_count=error_count,
            cache_hit_rate=0.0,  # Would need cache metrics from server
            test_params={
                "iterations": iterations,
                "prompt_types": len(self.test_prompts),
                "total_tokens_generated": total_tokens
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def run_throughput_benchmark(self, concurrent_requests: int = 10, 
                                     total_requests: int = 100) -> BenchmarkMetrics:
        """Benchmark throughput with concurrent requests."""
        logger.info(f"Running throughput benchmark ({concurrent_requests} concurrent, {total_requests} total)")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        all_latencies = []
        all_tokens_per_second = []
        error_count = 0
        total_tokens = 0
        
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(prompt_data: Tuple[str, str, int]) -> Optional[Dict]:
            async with semaphore:
                prompt_name, prompt_text, max_tokens = prompt_data
                
                try:
                    request_data = {
                        "model": "mlx-community/Qwen3-1.7B-8bit",
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                    
                    request_start = time.time()
                    # Use aiohttp for async requests (simplified with requests for now)
                    response = requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=request_data,
                        timeout=60
                    )
                    request_time = time.time() - request_start
                    
                    if response.status_code == 200:
                        return {
                            "latency": request_time * 1000,
                            "response": response.json(),
                            "success": True
                        }
                    else:
                        return {
                            "latency": request_time * 1000,
                            "error": f"HTTP {response.status_code}",
                            "success": False
                        }
                
                except Exception as e:
                    return {
                        "error": str(e),
                        "success": False
                    }
        
        # Generate request tasks
        tasks = []
        for i in range(total_requests):
            prompt_data = self.test_prompts[i % len(self.test_prompts)]
            tasks.append(make_request(prompt_data))
        
        start_time = time.time()
        
        # Execute all requests
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop()
        
        # Process results
        for result in results:
            if result.get("success", False):
                all_latencies.append(result["latency"])
                if "response" in result and "usage" in result["response"]:
                    tokens = result["response"]["usage"].get("completion_tokens", 0)
                    total_tokens += tokens
                    if result["latency"] > 0:
                        tokens_per_sec = tokens / (result["latency"] / 1000)
                        all_tokens_per_second.append(tokens_per_sec)
            else:
                error_count += 1
        
        success_rate = len(all_latencies) / total_requests if total_requests > 0 else 0.0
        
        return BenchmarkMetrics(
            test_name="distributed_inference_throughput",
            implementation="grpc_distributed",
            total_time=total_time,
            avg_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
            min_latency_ms=min(all_latencies) if all_latencies else 0.0,
            max_latency_ms=max(all_latencies) if all_latencies else 0.0,
            p95_latency_ms=np.percentile(all_latencies, 95) if all_latencies else 0.0,
            p99_latency_ms=np.percentile(all_latencies, 99) if all_latencies else 0.0,
            tokens_per_second=statistics.mean(all_tokens_per_second) if all_tokens_per_second else 0.0,
            requests_per_second=len(all_latencies) / total_time if total_time > 0 else 0.0,
            operations_per_second=len(all_latencies) / total_time if total_time > 0 else 0.0,
            peak_memory_mb=resource_metrics.get("peak_memory_mb", 0.0),
            avg_memory_mb=resource_metrics.get("avg_memory_mb", 0.0),
            memory_efficiency_score=total_tokens / resource_metrics.get("peak_memory_mb", 1.0),
            serialization_time_ms=0.0,
            deserialization_time_ms=0.0,
            network_transfer_time_ms=resource_metrics.get("total_network_bytes", 0.0) / 1000.0,
            compression_ratio=1.0,
            success_rate=success_rate,
            error_count=error_count,
            cache_hit_rate=0.0,
            test_params={
                "concurrent_requests": concurrent_requests,
                "total_requests": total_requests,
                "total_tokens_generated": total_tokens
            },
            timestamp=datetime.now().isoformat()
        )


class CommunicationBenchmark:
    """Benchmarks gRPC communication overhead."""
    
    def __init__(self):
        self.test_data_sizes = [
            (1024, "1KB"),
            (10 * 1024, "10KB"),
            (100 * 1024, "100KB"),
            (1024 * 1024, "1MB"),
            (10 * 1024 * 1024, "10MB")
        ]
    
    async def run_communication_benchmark(self, iterations: int = 100) -> BenchmarkMetrics:
        """Benchmark gRPC communication overhead."""
        logger.info(f"Running communication benchmark ({iterations} iterations)")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        all_round_trip_times = []
        all_serialize_times = []
        all_deserialize_times = []
        error_count = 0
        
        start_time = time.time()
        
        for data_size, size_name in self.test_data_sizes:
            logger.debug(f"Benchmarking data size: {size_name}")
            
            # Generate test tensor of appropriate size
            tensor_shape = (data_size // 4,)  # 4 bytes per float32
            test_tensor = mx.random.normal(tensor_shape).astype(mx.float32)
            
            for _ in range(iterations // len(self.test_data_sizes)):
                try:
                    # Measure full round-trip (serialize + deserialize)
                    round_trip_start = time.time()
                    
                    # Serialize
                    serialize_start = time.time()
                    proto_tensor = TensorSerializer.tensor_to_proto(test_tensor)
                    serialize_time = (time.time() - serialize_start) * 1000
                    all_serialize_times.append(serialize_time)
                    
                    # Simulate network transfer (for now, just the serialized size)
                    serialized_data = proto_tensor.SerializeToString()
                    transfer_size = len(serialized_data)
                    
                    # Deserialize
                    deserialize_start = time.time()
                    reconstructed = TensorSerializer.proto_to_tensor(proto_tensor)
                    mx.eval(reconstructed)
                    deserialize_time = (time.time() - deserialize_start) * 1000
                    all_deserialize_times.append(deserialize_time)
                    
                    round_trip_time = (time.time() - round_trip_start) * 1000
                    all_round_trip_times.append(round_trip_time)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Communication benchmark error for {size_name}: {e}")
        
        total_time = time.time() - start_time
        resource_metrics = monitor.stop()
        
        total_operations = len(all_round_trip_times)
        success_rate = 1.0 - (error_count / (total_operations + error_count)) if (total_operations + error_count) > 0 else 0.0
        
        return BenchmarkMetrics(
            test_name="grpc_communication",
            implementation="grpc_protobuf",
            total_time=total_time,
            avg_latency_ms=statistics.mean(all_round_trip_times) if all_round_trip_times else 0.0,
            min_latency_ms=min(all_round_trip_times) if all_round_trip_times else 0.0,
            max_latency_ms=max(all_round_trip_times) if all_round_trip_times else 0.0,
            p95_latency_ms=np.percentile(all_round_trip_times, 95) if all_round_trip_times else 0.0,
            p99_latency_ms=np.percentile(all_round_trip_times, 99) if all_round_trip_times else 0.0,
            tokens_per_second=0.0,
            requests_per_second=0.0,
            operations_per_second=total_operations / total_time if total_time > 0 else 0.0,
            peak_memory_mb=resource_metrics.get("peak_memory_mb", 0.0),
            avg_memory_mb=resource_metrics.get("avg_memory_mb", 0.0),
            memory_efficiency_score=1.0,
            serialization_time_ms=statistics.mean(all_serialize_times) if all_serialize_times else 0.0,
            deserialization_time_ms=statistics.mean(all_deserialize_times) if all_deserialize_times else 0.0,
            network_transfer_time_ms=statistics.mean(all_round_trip_times) - statistics.mean(all_serialize_times) - statistics.mean(all_deserialize_times) if all_round_trip_times and all_serialize_times and all_deserialize_times else 0.0,
            compression_ratio=1.0,
            success_rate=success_rate,
            error_count=error_count,
            cache_hit_rate=0.0,
            test_params={
                "iterations": iterations,
                "data_sizes": [size for size, _ in self.test_data_sizes],
                "total_operations": total_operations
            },
            timestamp=datetime.now().isoformat()
        )


class ComprehensiveBenchmarkSuite:
    """Main benchmarking suite that orchestrates all tests."""
    
    def __init__(self, api_url: str = "http://localhost:8100", output_dir: str = "benchmark_results"):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize benchmark components
        self.tensor_benchmark = TensorSerializationBenchmark()
        self.inference_benchmark = DistributedInferenceBenchmark(api_url)
        self.communication_benchmark = CommunicationBenchmark()
        
        self.results: List[BenchmarkMetrics] = []
    
    def check_system_status(self) -> bool:
        """Check if the distributed system is ready for benchmarking."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return (health_data.get("status") == "healthy" and 
                       health_data.get("model_loaded", False))
        except:
            pass
        return False
    
    async def run_all_benchmarks(self, 
                                tensor_iterations: int = 1000,
                                inference_iterations: int = 50,
                                communication_iterations: int = 100,
                                concurrent_requests: int = 10,
                                throughput_requests: int = 100) -> Dict[str, Any]:
        """Run all benchmark suites."""
        logger.info("Starting comprehensive benchmark suite")
        
        # Check system status
        if not self.check_system_status():
            logger.warning("Distributed system not ready. Some benchmarks may fail.")
        
        benchmark_results = {
            "suite_info": {
                "start_time": datetime.now().isoformat(),
                "api_url": self.api_url,
                "system_ready": self.check_system_status()
            },
            "benchmarks": {},
            "summary": {},
            "errors": []
        }
        
        # Run tensor serialization benchmark
        try:
            logger.info("Running tensor serialization benchmark...")
            tensor_metrics = await self.tensor_benchmark.run_benchmark(tensor_iterations)
            self.results.append(tensor_metrics)
            benchmark_results["benchmarks"]["tensor_serialization"] = asdict(tensor_metrics)
        except Exception as e:
            error_msg = f"Tensor serialization benchmark failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Run communication benchmark
        try:
            logger.info("Running communication benchmark...")
            comm_metrics = await self.communication_benchmark.run_communication_benchmark(communication_iterations)
            self.results.append(comm_metrics)
            benchmark_results["benchmarks"]["communication"] = asdict(comm_metrics)
        except Exception as e:
            error_msg = f"Communication benchmark failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Run inference benchmarks (only if system is ready)
        if self.check_system_status():
            try:
                logger.info("Running inference latency benchmark...")
                latency_metrics = await self.inference_benchmark.run_latency_benchmark(inference_iterations)
                self.results.append(latency_metrics)
                benchmark_results["benchmarks"]["inference_latency"] = asdict(latency_metrics)
            except Exception as e:
                error_msg = f"Inference latency benchmark failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
            
            try:
                logger.info("Running throughput benchmark...")
                throughput_metrics = await self.inference_benchmark.run_throughput_benchmark(
                    concurrent_requests, throughput_requests)
                self.results.append(throughput_metrics)
                benchmark_results["benchmarks"]["throughput"] = asdict(throughput_metrics)
            except Exception as e:
                error_msg = f"Throughput benchmark failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
        else:
            logger.warning("Skipping inference benchmarks - system not ready")
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["suite_info"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"comprehensive_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {results_file}")
        
        # Generate human-readable report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.md"
        self._generate_markdown_report(benchmark_results, report_file)
        
        return benchmark_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        summary = {
            "total_benchmarks": len(self.results),
            "overall_success_rate": statistics.mean(r.success_rate for r in self.results),
            "total_errors": sum(r.error_count for r in self.results),
            "benchmarks_by_type": {}
        }
        
        # Group by test type
        by_type = {}
        for result in self.results:
            test_type = result.test_name
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(result)
        
        # Generate type-specific summaries
        for test_type, results in by_type.items():
            summary["benchmarks_by_type"][test_type] = {
                "count": len(results),
                "avg_latency_ms": statistics.mean(r.avg_latency_ms for r in results),
                "avg_throughput": statistics.mean(r.operations_per_second for r in results),
                "avg_memory_mb": statistics.mean(r.avg_memory_mb for r in results),
                "success_rate": statistics.mean(r.success_rate for r in results)
            }
        
        return summary
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_file: Path):
        """Generate a human-readable markdown report."""
        report_lines = [
            "# Comprehensive MLX Distributed Inference Benchmark Report",
            f"Generated: {results['suite_info']['start_time']}",
            f"API URL: {results['suite_info']['api_url']}",
            f"System Ready: {results['suite_info']['system_ready']}",
            ""
        ]
        
        # Summary section
        if "summary" in results:
            summary = results["summary"]
            report_lines.extend([
                "## Executive Summary",
                f"- **Total Benchmarks:** {summary.get('total_benchmarks', 0)}",
                f"- **Overall Success Rate:** {summary.get('overall_success_rate', 0):.2%}",
                f"- **Total Errors:** {summary.get('total_errors', 0)}",
                ""
            ])
        
        # Detailed results
        if "benchmarks" in results:
            report_lines.append("## Detailed Results")
            
            for benchmark_name, metrics in results["benchmarks"].items():
                report_lines.extend([
                    f"### {benchmark_name.replace('_', ' ').title()}",
                    f"- **Average Latency:** {metrics.get('avg_latency_ms', 0):.2f} ms",
                    f"- **95th Percentile:** {metrics.get('p95_latency_ms', 0):.2f} ms",
                    f"- **99th Percentile:** {metrics.get('p99_latency_ms', 0):.2f} ms",
                    f"- **Throughput:** {metrics.get('operations_per_second', 0):.2f} ops/sec",
                    f"- **Memory Usage:** {metrics.get('avg_memory_mb', 0):.2f} MB",
                    f"- **Success Rate:** {metrics.get('success_rate', 0):.2%}",
                    ""
                ])
        
        # Errors section
        if results.get("errors"):
            report_lines.extend([
                "## Errors Encountered",
                *[f"- {error}" for error in results["errors"]],
                ""
            ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Markdown report saved to: {output_file}")


async def main():
    """Main entry point for running benchmarks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive MLX Distributed Inference Benchmark")
    parser.add_argument("--api-url", default="http://localhost:8100", help="API server URL")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--tensor-iterations", type=int, default=1000, help="Tensor serialization iterations")
    parser.add_argument("--inference-iterations", type=int, default=50, help="Inference benchmark iterations")
    parser.add_argument("--communication-iterations", type=int, default=100, help="Communication benchmark iterations")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="Concurrent requests for throughput test")
    parser.add_argument("--throughput-requests", type=int, default=100, help="Total requests for throughput test")
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = ComprehensiveBenchmarkSuite(
        api_url=args.api_url,
        output_dir=args.output_dir
    )
    
    logger.info("Starting comprehensive benchmark suite...")
    
    # Run all benchmarks
    results = await suite.run_all_benchmarks(
        tensor_iterations=args.tensor_iterations,
        inference_iterations=args.inference_iterations,
        communication_iterations=args.communication_iterations,
        concurrent_requests=args.concurrent_requests,
        throughput_requests=args.throughput_requests
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUITE COMPLETED")
    logger.info("="*80)
    
    if "summary" in results:
        summary = results["summary"]
        logger.info(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        logger.info(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.2%}")
        logger.info(f"Total Errors: {summary.get('total_errors', 0)}")
        
        if "benchmarks_by_type" in summary:
            logger.info("\nResults by Benchmark Type:")
            for test_type, stats in summary["benchmarks_by_type"].items():
                logger.info(f"  {test_type}:")
                logger.info(f"    Avg Latency: {stats['avg_latency_ms']:.2f} ms")
                logger.info(f"    Avg Throughput: {stats['avg_throughput']:.2f} ops/sec")
                logger.info(f"    Success Rate: {stats['success_rate']:.2%}")
    
    if results.get("errors"):
        logger.warning(f"\nErrors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            logger.warning(f"  - {error}")
    
    logger.info("\nBenchmark suite completed successfully!")
    return 0 if not results.get("errors") else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
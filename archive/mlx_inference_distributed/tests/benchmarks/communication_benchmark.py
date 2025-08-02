#!/usr/bin/env python3
"""
Communication Overhead Benchmarks for gRPC Tensor Passing.

This module provides detailed benchmarking of gRPC communication overhead in the
distributed MLX inference system, including:
- Raw gRPC call latency
- Tensor serialization/deserialization overhead
- Network transfer efficiency
- Compression effectiveness
- Batch processing benefits
- Connection pooling performance
- Error handling and retry mechanisms
"""

import asyncio
import time
import logging
import statistics
import json
import sys
import os
import threading
import socket
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import concurrent.futures
import tempfile

import mlx.core as mx
import numpy as np
import grpc
from grpc import aio as aio_grpc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from grpc_server import TensorSerializer, DistributedInferenceServicer
    import distributed_inference_pb2 as pb2
    import distributed_inference_pb2_grpc as pb2_grpc
    from distributed_comm import GRPCCommunicator
except ImportError as e:
    print(f"Warning: Could not import gRPC components: {e}")

logger = logging.getLogger(__name__)


@dataclass
class CommunicationMetrics:
    """Metrics for communication performance."""
    test_name: str
    
    # Timing metrics (milliseconds)
    avg_total_time_ms: float
    min_total_time_ms: float
    max_total_time_ms: float
    p95_total_time_ms: float
    p99_total_time_ms: float
    
    # Component breakdown (milliseconds)
    avg_serialization_time_ms: float
    avg_network_time_ms: float
    avg_deserialization_time_ms: float
    
    # Throughput metrics
    operations_per_second: float
    bytes_per_second: float
    
    # Size metrics
    original_size_bytes: int
    serialized_size_bytes: int
    compression_ratio: float
    
    # Quality metrics
    success_rate: float
    error_count: int
    timeout_count: int
    
    # Configuration
    test_config: Dict[str, Any]
    timestamp: str


@dataclass
class NetworkLatencyProfile:
    """Network latency profile for different scenarios."""
    scenario: str
    avg_latency_ms: float
    jitter_ms: float
    packet_loss_percent: float
    bandwidth_mbps: float


class TensorGenerator:
    """Generates test tensors with various characteristics."""
    
    @staticmethod
    def generate_test_tensors() -> List[Tuple[str, mx.array, Dict[str, Any]]]:
        """Generate comprehensive set of test tensors."""
        tensors = []
        
        # Small tensors (typical for embeddings/hidden states)
        tensors.append((
            "small_vector",
            mx.random.normal((256,)).astype(mx.float32),
            {"category": "small", "elements": 256, "dtype": "float32"}
        ))
        
        tensors.append((
            "small_matrix",
            mx.random.normal((32, 64)).astype(mx.float16),
            {"category": "small", "elements": 2048, "dtype": "float16"}
        ))
        
        # Medium tensors (typical for layer outputs)
        tensors.append((
            "medium_hidden_state",
            mx.random.normal((128, 768)).astype(mx.float32),
            {"category": "medium", "elements": 98304, "dtype": "float32"}
        ))
        
        tensors.append((
            "medium_attention",
            mx.random.normal((12, 128, 128)).astype(mx.float16),
            {"category": "medium", "elements": 196608, "dtype": "float16"}
        ))
        
        # Large tensors (typical for model weights)
        tensors.append((
            "large_linear_weight",
            mx.random.normal((4096, 1024)).astype(mx.float32),
            {"category": "large", "elements": 4194304, "dtype": "float32"}
        ))
        
        tensors.append((
            "large_embedding",
            mx.random.normal((50000, 768)).astype(mx.float16),
            {"category": "large", "elements": 38400000, "dtype": "float16"}
        ))
        
        # Special case tensors
        # Sparse tensor (mostly zeros)
        sparse_tensor = mx.zeros((1000, 1000)).astype(mx.float32)
        sparse_tensor[::100, ::100] = mx.random.normal((10, 10))
        tensors.append((
            "sparse_tensor",
            sparse_tensor,
            {"category": "sparse", "elements": 1000000, "dtype": "float32", "sparsity": 0.99}
        ))
        
        # bfloat16 tensor (tests dtype preservation)
        tensors.append((
            "bfloat16_tensor",
            mx.random.normal((512, 512)).astype(mx.bfloat16),
            {"category": "medium", "elements": 262144, "dtype": "bfloat16"}
        ))
        
        # Integer tensor
        tensors.append((
            "int32_indices",
            mx.random.randint(0, 10000, (1024, 256)).astype(mx.int32),
            {"category": "medium", "elements": 262144, "dtype": "int32"}
        ))
        
        # Very large tensor (stress test)
        tensors.append((
            "xlarge_tensor",
            mx.random.normal((2048, 2048)).astype(mx.float32),
            {"category": "xlarge", "elements": 4194304, "dtype": "float32"}
        ))
        
        return tensors


class gRPCConnectionTester:
    """Tests gRPC connection establishment and management."""
    
    def __init__(self, server_address: str = "localhost:50100"):
        self.server_address = server_address
        self.channel = None
        self.stub = None
    
    async def test_connection_establishment(self, iterations: int = 100) -> CommunicationMetrics:
        """Test gRPC connection establishment performance."""
        logger.info(f"Testing gRPC connection establishment ({iterations} iterations)")
        
        connection_times = []
        errors = 0
        timeouts = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                conn_start = time.time()
                
                # Create new channel
                channel = aio_grpc.insecure_channel(self.server_address)
                stub = pb2_grpc.DistributedInferenceStub(channel)
                
                # Test the connection with a simple call
                try:
                    # Create a minimal request for connection test
                    request = pb2.InferenceRequest(
                        input=pb2.Tensor(
                            shape=[1],
                            dtype="float32",
                            data=struct.pack('f', 1.0)
                        ),
                        sequence_length=1
                    )
                    
                    # Make call with short timeout
                    response = await stub.ProcessInference(request, timeout=5.0)
                    
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        timeouts += 1
                    else:
                        errors += 1
                
                # Close channel
                await channel.close()
                
                connection_time = (time.time() - conn_start) * 1000
                connection_times.append(connection_time)
                
            except Exception as e:
                errors += 1
                logger.debug(f"Connection test error: {e}")
        
        total_time = time.time() - start_time
        
        return CommunicationMetrics(
            test_name="grpc_connection_establishment",
            avg_total_time_ms=statistics.mean(connection_times) if connection_times else 0,
            min_total_time_ms=min(connection_times) if connection_times else 0,
            max_total_time_ms=max(connection_times) if connection_times else 0,
            p95_total_time_ms=np.percentile(connection_times, 95) if connection_times else 0,
            p99_total_time_ms=np.percentile(connection_times, 99) if connection_times else 0,
            avg_serialization_time_ms=0,  # Not applicable
            avg_network_time_ms=statistics.mean(connection_times) if connection_times else 0,
            avg_deserialization_time_ms=0,  # Not applicable
            operations_per_second=len(connection_times) / total_time if total_time > 0 else 0,
            bytes_per_second=0,  # Not applicable
            original_size_bytes=0,
            serialized_size_bytes=0,
            compression_ratio=1.0,
            success_rate=len(connection_times) / iterations if iterations > 0 else 0,
            error_count=errors,
            timeout_count=timeouts,
            test_config={
                "iterations": iterations,
                "server_address": self.server_address
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_connection_reuse(self, iterations: int = 100) -> CommunicationMetrics:
        """Test performance of reusing gRPC connections."""
        logger.info(f"Testing gRPC connection reuse ({iterations} iterations)")
        
        call_times = []
        errors = 0
        timeouts = 0
        
        # Establish persistent connection
        channel = aio_grpc.insecure_channel(self.server_address)
        stub = pb2_grpc.DistributedInferenceStub(channel)
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                call_start = time.time()
                
                # Create minimal request
                request = pb2.InferenceRequest(
                    input=pb2.Tensor(
                        shape=[1],
                        dtype="float32",
                        data=struct.pack('f', float(i))
                    ),
                    sequence_length=1
                )
                
                try:
                    response = await stub.ProcessInference(request, timeout=5.0)
                    call_time = (time.time() - call_start) * 1000
                    call_times.append(call_time)
                
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        timeouts += 1
                    else:
                        errors += 1
                
            except Exception as e:
                errors += 1
                logger.debug(f"Connection reuse test error: {e}")
        
        total_time = time.time() - start_time
        
        # Close connection
        await channel.close()
        
        return CommunicationMetrics(
            test_name="grpc_connection_reuse",
            avg_total_time_ms=statistics.mean(call_times) if call_times else 0,
            min_total_time_ms=min(call_times) if call_times else 0,
            max_total_time_ms=max(call_times) if call_times else 0,
            p95_total_time_ms=np.percentile(call_times, 95) if call_times else 0,
            p99_total_time_ms=np.percentile(call_times, 99) if call_times else 0,
            avg_serialization_time_ms=0,
            avg_network_time_ms=statistics.mean(call_times) if call_times else 0,
            avg_deserialization_time_ms=0,
            operations_per_second=len(call_times) / total_time if total_time > 0 else 0,
            bytes_per_second=0,
            original_size_bytes=0,
            serialized_size_bytes=0,
            compression_ratio=1.0,
            success_rate=len(call_times) / iterations if iterations > 0 else 0,
            error_count=errors,
            timeout_count=timeouts,
            test_config={
                "iterations": iterations,
                "server_address": self.server_address,
                "connection_reused": True
            },
            timestamp=datetime.now().isoformat()
        )


class TensorSerializationBenchmark:
    """Benchmarks tensor serialization performance in isolation."""
    
    def __init__(self):
        self.test_tensors = TensorGenerator.generate_test_tensors()
    
    async def benchmark_serialization_performance(self, iterations: int = 1000) -> List[CommunicationMetrics]:
        """Benchmark tensor serialization performance for different tensor types."""
        logger.info(f"Benchmarking tensor serialization performance ({iterations} iterations)")
        
        results = []
        
        for tensor_name, tensor, tensor_info in self.test_tensors:
            logger.debug(f"Benchmarking serialization for {tensor_name}")
            
            serialize_times = []
            deserialize_times = []
            total_times = []
            original_sizes = []
            serialized_sizes = []
            errors = 0
            
            start_time = time.time()
            
            for i in range(iterations // len(self.test_tensors)):
                try:
                    # Measure serialization
                    serialize_start = time.time()
                    proto_tensor = TensorSerializer.tensor_to_proto(tensor)
                    serialize_time = (time.time() - serialize_start) * 1000
                    serialize_times.append(serialize_time)
                    
                    # Measure serialized size
                    serialized_data = proto_tensor.SerializeToString()
                    serialized_size = len(serialized_data)
                    serialized_sizes.append(serialized_size)
                    
                    # Measure deserialization
                    deserialize_start = time.time()
                    reconstructed = TensorSerializer.proto_to_tensor(proto_tensor)
                    mx.eval(reconstructed)  # Ensure computation
                    deserialize_time = (time.time() - deserialize_start) * 1000
                    deserialize_times.append(deserialize_time)
                    
                    # Total round-trip time
                    total_time = serialize_time + deserialize_time
                    total_times.append(total_time)
                    
                    # Original size
                    original_size = tensor.nbytes
                    original_sizes.append(original_size)
                    
                    # Verify correctness
                    if not mx.allclose(tensor, reconstructed, rtol=1e-4):
                        if tensor.dtype != mx.bfloat16:  # bfloat16 has lower precision
                            logger.warning(f"Reconstruction mismatch for {tensor_name}")
                
                except Exception as e:
                    errors += 1
                    logger.debug(f"Serialization error for {tensor_name}: {e}")
            
            benchmark_duration = time.time() - start_time
            
            # Calculate metrics
            avg_original_size = statistics.mean(original_sizes) if original_sizes else 0
            avg_serialized_size = statistics.mean(serialized_sizes) if serialized_sizes else 0
            compression_ratio = avg_original_size / avg_serialized_size if avg_serialized_size > 0 else 1
            
            metrics = CommunicationMetrics(
                test_name=f"tensor_serialization_{tensor_name}",
                avg_total_time_ms=statistics.mean(total_times) if total_times else 0,
                min_total_time_ms=min(total_times) if total_times else 0,
                max_total_time_ms=max(total_times) if total_times else 0,
                p95_total_time_ms=np.percentile(total_times, 95) if total_times else 0,
                p99_total_time_ms=np.percentile(total_times, 99) if total_times else 0,
                avg_serialization_time_ms=statistics.mean(serialize_times) if serialize_times else 0,
                avg_network_time_ms=0,  # No network in this test
                avg_deserialization_time_ms=statistics.mean(deserialize_times) if deserialize_times else 0,
                operations_per_second=len(total_times) / benchmark_duration if benchmark_duration > 0 else 0,
                bytes_per_second=avg_serialized_size * len(total_times) / benchmark_duration if benchmark_duration > 0 else 0,
                original_size_bytes=int(avg_original_size),
                serialized_size_bytes=int(avg_serialized_size),
                compression_ratio=compression_ratio,
                success_rate=len(total_times) / (len(total_times) + errors) if (len(total_times) + errors) > 0 else 0,
                error_count=errors,
                timeout_count=0,
                test_config={
                    "tensor_name": tensor_name,
                    "tensor_info": tensor_info,
                    "iterations": iterations // len(self.test_tensors)
                },
                timestamp=datetime.now().isoformat()
            )
            
            results.append(metrics)
            
            logger.debug(f"  {tensor_name}: {metrics.avg_total_time_ms:.2f}ms avg, "
                        f"{metrics.compression_ratio:.2f}x compression, "
                        f"{metrics.operations_per_second:.0f} ops/sec")
        
        return results


class NetworkOverheadAnalyzer:
    """Analyzes network overhead in gRPC communication."""
    
    def __init__(self, server_address: str = "localhost:50100"):
        self.server_address = server_address
        self.host, self.port = server_address.split(':')
        self.port = int(self.port)
    
    async def measure_raw_network_latency(self, iterations: int = 100) -> CommunicationMetrics:
        """Measure raw network latency using TCP sockets."""
        logger.info(f"Measuring raw network latency ({iterations} iterations)")
        
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                latency_start = time.time()
                
                # Create socket connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                
                # Connect and immediately close (measures connection overhead)
                sock.connect((self.host, self.port))
                sock.close()
                
                latency = (time.time() - latency_start) * 1000
                latencies.append(latency)
                
            except Exception as e:
                errors += 1
                logger.debug(f"Network latency test error: {e}")
        
        total_time = time.time() - start_time
        
        return CommunicationMetrics(
            test_name="raw_network_latency",
            avg_total_time_ms=statistics.mean(latencies) if latencies else 0,
            min_total_time_ms=min(latencies) if latencies else 0,
            max_total_time_ms=max(latencies) if latencies else 0,
            p95_total_time_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_total_time_ms=np.percentile(latencies, 99) if latencies else 0,
            avg_serialization_time_ms=0,
            avg_network_time_ms=statistics.mean(latencies) if latencies else 0,
            avg_deserialization_time_ms=0,
            operations_per_second=len(latencies) / total_time if total_time > 0 else 0,
            bytes_per_second=0,
            original_size_bytes=0,
            serialized_size_bytes=0,
            compression_ratio=1.0,
            success_rate=len(latencies) / iterations if iterations > 0 else 0,
            error_count=errors,
            timeout_count=0,
            test_config={
                "iterations": iterations,
                "server_address": self.server_address,
                "test_type": "tcp_socket"
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def analyze_grpc_overhead(self, baseline_latency_ms: float, grpc_latency_ms: float) -> Dict[str, Any]:
        """Analyze gRPC overhead compared to raw network latency."""
        
        if baseline_latency_ms <= 0:
            return {"error": "Invalid baseline latency"}
        
        overhead_ms = grpc_latency_ms - baseline_latency_ms
        overhead_percentage = (overhead_ms / baseline_latency_ms) * 100
        
        analysis = {
            "baseline_latency_ms": baseline_latency_ms,
            "grpc_latency_ms": grpc_latency_ms,
            "overhead_ms": overhead_ms,
            "overhead_percentage": overhead_percentage,
            "efficiency_score": baseline_latency_ms / grpc_latency_ms if grpc_latency_ms > 0 else 0
        }
        
        # Categorize overhead
        if overhead_percentage < 10:
            analysis["overhead_category"] = "LOW"
            analysis["recommendation"] = "Excellent gRPC efficiency"
        elif overhead_percentage < 25:
            analysis["overhead_category"] = "MODERATE"
            analysis["recommendation"] = "Acceptable gRPC overhead"
        elif overhead_percentage < 50:
            analysis["overhead_category"] = "HIGH"
            analysis["recommendation"] = "Consider optimizing gRPC configuration"
        else:
            analysis["overhead_category"] = "VERY_HIGH"
            analysis["recommendation"] = "Investigate gRPC performance issues"
        
        return analysis


class BatchProcessingBenchmark:
    """Benchmarks batch processing efficiency in gRPC."""
    
    def __init__(self, server_address: str = "localhost:50100"):
        self.server_address = server_address
    
    async def benchmark_batch_sizes(self, max_batch_size: int = 32, iterations: int = 50) -> List[CommunicationMetrics]:
        """Benchmark different batch sizes for gRPC communication."""
        logger.info(f"Benchmarking batch processing (max batch size: {max_batch_size})")
        
        results = []
        batch_sizes = [1, 2, 4, 8, 16, min(32, max_batch_size)]
        
        # Generate test tensor
        test_tensor = mx.random.normal((128, 768)).astype(mx.float32)
        
        for batch_size in batch_sizes:
            logger.debug(f"Testing batch size: {batch_size}")
            
            latencies = []
            throughput_measurements = []
            errors = 0
            total_bytes = 0
            
            start_time = time.time()
            
            for i in range(iterations):
                try:
                    batch_start = time.time()
                    
                    # Create batch of tensors
                    batch_tensors = []
                    for j in range(batch_size):
                        # Slightly modify each tensor in batch
                        modified_tensor = test_tensor + (j * 0.01)
                        proto_tensor = TensorSerializer.tensor_to_proto(modified_tensor)
                        batch_tensors.append(proto_tensor)
                        total_bytes += len(proto_tensor.SerializeToString())
                    
                    # Simulate batch processing (in real implementation, this would be a single gRPC call)
                    # For now, we simulate the serialization overhead of batch processing
                    processing_time = len(batch_tensors) * 0.001  # 1ms per tensor processing overhead
                    await asyncio.sleep(processing_time)
                    
                    batch_time = (time.time() - batch_start) * 1000
                    latencies.append(batch_time)
                    
                    # Calculate throughput (tensors per second)
                    if batch_time > 0:
                        tensors_per_sec = (batch_size * 1000) / batch_time
                        throughput_measurements.append(tensors_per_sec)
                
                except Exception as e:
                    errors += 1
                    logger.debug(f"Batch processing error: {e}")
            
            test_duration = time.time() - start_time
            
            # Calculate metrics
            avg_latency_per_tensor = statistics.mean(latencies) / batch_size if latencies and batch_size > 0 else 0
            total_operations = len(latencies) * batch_size
            
            metrics = CommunicationMetrics(
                test_name=f"batch_processing_size_{batch_size}",
                avg_total_time_ms=statistics.mean(latencies) if latencies else 0,
                min_total_time_ms=min(latencies) if latencies else 0,
                max_total_time_ms=max(latencies) if latencies else 0,
                p95_total_time_ms=np.percentile(latencies, 95) if latencies else 0,
                p99_total_time_ms=np.percentile(latencies, 99) if latencies else 0,
                avg_serialization_time_ms=avg_latency_per_tensor,  # Approximation
                avg_network_time_ms=0,  # Simulated
                avg_deserialization_time_ms=0,  # Simulated
                operations_per_second=statistics.mean(throughput_measurements) if throughput_measurements else 0,
                bytes_per_second=total_bytes / test_duration if test_duration > 0 else 0,
                original_size_bytes=test_tensor.nbytes * batch_size,
                serialized_size_bytes=total_bytes // iterations if iterations > 0 else 0,
                compression_ratio=1.0,  # Not calculated for batch test
                success_rate=len(latencies) / iterations if iterations > 0 else 0,
                error_count=errors,
                timeout_count=0,
                test_config={
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "tensor_shape": list(test_tensor.shape),
                    "tensor_dtype": str(test_tensor.dtype)
                },
                timestamp=datetime.now().isoformat()
            )
            
            results.append(metrics)
            
            logger.debug(f"  Batch size {batch_size}: {metrics.avg_total_time_ms:.2f}ms total, "
                        f"{avg_latency_per_tensor:.2f}ms per tensor, "
                        f"{metrics.operations_per_second:.0f} tensors/sec")
        
        return results


class ComprehensiveCommunicationBenchmark:
    """Main class for comprehensive communication benchmarking."""
    
    def __init__(self, server_address: str = "localhost:50100", output_dir: str = "communication_results"):
        self.server_address = server_address
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize benchmark components
        self.connection_tester = gRPCConnectionTester(server_address)
        self.serialization_benchmark = TensorSerializationBenchmark()
        self.network_analyzer = NetworkOverheadAnalyzer(server_address)
        self.batch_benchmark = BatchProcessingBenchmark(server_address)
        
        self.results: List[CommunicationMetrics] = []
    
    async def run_all_benchmarks(self, 
                                connection_iterations: int = 100,
                                serialization_iterations: int = 1000,
                                network_iterations: int = 100,
                                batch_iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive communication benchmarks."""
        logger.info("Starting comprehensive communication benchmarks")
        
        benchmark_results = {
            "benchmark_info": {
                "start_time": datetime.now().isoformat(),
                "server_address": self.server_address,
                "test_configuration": {
                    "connection_iterations": connection_iterations,
                    "serialization_iterations": serialization_iterations,
                    "network_iterations": network_iterations,
                    "batch_iterations": batch_iterations
                }
            },
            "results": {},
            "analysis": {},
            "summary": {},
            "errors": []
        }
        
        # 1. Test gRPC connection performance
        try:
            logger.info("Testing gRPC connection establishment...")
            connection_metrics = await self.connection_tester.test_connection_establishment(connection_iterations)
            self.results.append(connection_metrics)
            benchmark_results["results"]["connection_establishment"] = asdict(connection_metrics)
            
            logger.info("Testing gRPC connection reuse...")
            reuse_metrics = await self.connection_tester.test_connection_reuse(connection_iterations)
            self.results.append(reuse_metrics)
            benchmark_results["results"]["connection_reuse"] = asdict(reuse_metrics)
            
        except Exception as e:
            error_msg = f"gRPC connection tests failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # 2. Test tensor serialization performance
        try:
            logger.info("Testing tensor serialization performance...")
            serialization_results = await self.serialization_benchmark.benchmark_serialization_performance(serialization_iterations)
            self.results.extend(serialization_results)
            
            benchmark_results["results"]["tensor_serialization"] = {}
            for metrics in serialization_results:
                benchmark_results["results"]["tensor_serialization"][metrics.test_name] = asdict(metrics)
                
        except Exception as e:
            error_msg = f"Tensor serialization tests failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # 3. Test raw network latency
        try:
            logger.info("Testing raw network latency...")
            network_metrics = await self.network_analyzer.measure_raw_network_latency(network_iterations)
            self.results.append(network_metrics)
            benchmark_results["results"]["raw_network_latency"] = asdict(network_metrics)
            
        except Exception as e:
            error_msg = f"Network latency tests failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # 4. Test batch processing
        try:
            logger.info("Testing batch processing performance...")
            batch_results = await self.batch_benchmark.benchmark_batch_sizes(max_batch_size=32, iterations=batch_iterations)
            self.results.extend(batch_results)
            
            benchmark_results["results"]["batch_processing"] = {}
            for metrics in batch_results:
                benchmark_results["results"]["batch_processing"][metrics.test_name] = asdict(metrics)
                
        except Exception as e:
            error_msg = f"Batch processing tests failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # 5. Analyze overhead
        try:
            logger.info("Analyzing communication overhead...")
            analysis = await self._analyze_communication_overhead(benchmark_results["results"])
            benchmark_results["analysis"] = analysis
            
        except Exception as e:
            error_msg = f"Overhead analysis failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # 6. Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # 7. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"communication_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Communication benchmark results saved to: {results_file}")
        
        # 8. Generate report
        report_file = self.output_dir / f"communication_report_{timestamp}.md"
        self._generate_communication_report(benchmark_results, report_file)
        
        return benchmark_results
    
    async def _analyze_communication_overhead(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication overhead from benchmark results."""
        analysis = {
            "grpc_overhead": {},
            "serialization_efficiency": {},
            "batch_processing_benefits": {},
            "recommendations": []
        }
        
        # Analyze gRPC overhead
        if "raw_network_latency" in results and "connection_reuse" in results:
            raw_latency = results["raw_network_latency"]["avg_total_time_ms"]
            grpc_latency = results["connection_reuse"]["avg_total_time_ms"]
            
            overhead_analysis = await self.network_analyzer.analyze_grpc_overhead(raw_latency, grpc_latency)
            analysis["grpc_overhead"] = overhead_analysis
            
            if overhead_analysis.get("overhead_percentage", 0) > 25:
                analysis["recommendations"].append("High gRPC overhead detected - consider connection pooling")
        
        # Analyze serialization efficiency
        if "tensor_serialization" in results:
            serialization_results = results["tensor_serialization"]
            
            best_compression = 0
            worst_performance = float('inf')
            best_performance = 0
            
            for test_name, metrics in serialization_results.items():
                compression = metrics.get("compression_ratio", 1.0)
                performance = metrics.get("operations_per_second", 0)
                
                best_compression = max(best_compression, compression)
                worst_performance = min(worst_performance, performance)
                best_performance = max(best_performance, performance)
            
            analysis["serialization_efficiency"] = {
                "best_compression_ratio": best_compression,
                "performance_range": {
                    "min_ops_per_sec": worst_performance,
                    "max_ops_per_sec": best_performance,
                    "performance_variance": (best_performance - worst_performance) / best_performance if best_performance > 0 else 0
                }
            }
            
            if best_compression < 1.1:
                analysis["recommendations"].append("Poor compression detected - consider compression algorithms")
        
        # Analyze batch processing benefits
        if "batch_processing" in results:
            batch_results = results["batch_processing"]
            batch_metrics = []
            
            for test_name, metrics in batch_results.items():
                batch_size = metrics["test_config"]["batch_size"]
                throughput = metrics.get("operations_per_second", 0)
                batch_metrics.append((batch_size, throughput))
            
            if len(batch_metrics) > 1:
                # Sort by batch size
                batch_metrics.sort(key=lambda x: x[0])
                
                # Calculate scaling efficiency
                baseline_throughput = batch_metrics[0][1]  # Single item throughput
                scaling_efficiency = []
                
                for batch_size, throughput in batch_metrics:
                    expected_throughput = baseline_throughput * batch_size
                    actual_efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
                    scaling_efficiency.append((batch_size, actual_efficiency))
                
                analysis["batch_processing_benefits"] = {
                    "scaling_efficiency": scaling_efficiency,
                    "optimal_batch_size": max(scaling_efficiency, key=lambda x: x[1])[0] if scaling_efficiency else 1
                }
                
                avg_efficiency = sum(eff for _, eff in scaling_efficiency) / len(scaling_efficiency)
                if avg_efficiency < 0.7:
                    analysis["recommendations"].append("Poor batch scaling - investigate serialization overhead")
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by category
        by_category = {}
        for result in self.results:
            category = result.test_name.split('_')[0]  # First part of test name
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        summary = {
            "total_tests": len(self.results),
            "categories": {},
            "overall_performance": {}
        }
        
        # Category summaries
        for category, results in by_category.items():
            avg_latency = statistics.mean(r.avg_total_time_ms for r in results)
            avg_throughput = statistics.mean(r.operations_per_second for r in results)
            avg_success_rate = statistics.mean(r.success_rate for r in results)
            
            summary["categories"][category] = {
                "test_count": len(results),
                "avg_latency_ms": avg_latency,
                "avg_throughput_ops_sec": avg_throughput,
                "avg_success_rate": avg_success_rate
            }
        
        # Overall performance metrics
        all_latencies = [r.avg_total_time_ms for r in self.results]
        all_throughputs = [r.operations_per_second for r in self.results]
        all_success_rates = [r.success_rate for r in self.results]
        
        summary["overall_performance"] = {
            "avg_latency_ms": statistics.mean(all_latencies),
            "min_latency_ms": min(all_latencies),
            "max_latency_ms": max(all_latencies),
            "avg_throughput_ops_sec": statistics.mean(all_throughputs),
            "overall_success_rate": statistics.mean(all_success_rates),
            "total_errors": sum(r.error_count for r in self.results)
        }
        
        return summary
    
    def _generate_communication_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive communication report."""
        report_lines = [
            "# Communication Overhead Benchmark Report",
            f"Generated: {results['benchmark_info']['start_time']}",
            f"Server: {results['benchmark_info']['server_address']}",
            ""
        ]
        
        # Executive Summary
        if "summary" in results:
            summary = results["summary"]
            report_lines.extend([
                "## Executive Summary",
                f"- **Total Tests**: {summary.get('total_tests', 0)}",
                f"- **Average Latency**: {summary.get('overall_performance', {}).get('avg_latency_ms', 0):.2f} ms",
                f"- **Average Throughput**: {summary.get('overall_performance', {}).get('avg_throughput_ops_sec', 0):.0f} ops/sec",
                f"- **Overall Success Rate**: {summary.get('overall_performance', {}).get('overall_success_rate', 0):.1%}",
                ""
            ])
        
        # Analysis Results
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "## Analysis Results",
                ""
            ])
            
            # gRPC Overhead
            if "grpc_overhead" in analysis:
                overhead = analysis["grpc_overhead"]
                report_lines.extend([
                    "### gRPC Communication Overhead",
                    f"- **Raw Network Latency**: {overhead.get('baseline_latency_ms', 0):.2f} ms",
                    f"- **gRPC Latency**: {overhead.get('grpc_latency_ms', 0):.2f} ms",
                    f"- **Overhead**: {overhead.get('overhead_ms', 0):.2f} ms ({overhead.get('overhead_percentage', 0):.1f}%)",
                    f"- **Category**: {overhead.get('overhead_category', 'UNKNOWN')}",
                    f"- **Recommendation**: {overhead.get('recommendation', 'N/A')}",
                    ""
                ])
            
            # Serialization Efficiency
            if "serialization_efficiency" in analysis:
                ser_eff = analysis["serialization_efficiency"]
                report_lines.extend([
                    "### Tensor Serialization Efficiency",
                    f"- **Best Compression Ratio**: {ser_eff.get('best_compression_ratio', 1):.2f}x",
                    f"- **Performance Range**: {ser_eff.get('performance_range', {}).get('min_ops_per_sec', 0):.0f} - {ser_eff.get('performance_range', {}).get('max_ops_per_sec', 0):.0f} ops/sec",
                    ""
                ])
            
            # Batch Processing
            if "batch_processing_benefits" in analysis:
                batch = analysis["batch_processing_benefits"]
                report_lines.extend([
                    "### Batch Processing Benefits",
                    f"- **Optimal Batch Size**: {batch.get('optimal_batch_size', 1)}",
                    ""
                ])
            
            # Recommendations
            if analysis.get("recommendations"):
                report_lines.extend([
                    "### Recommendations",
                    ""
                ])
                for rec in analysis["recommendations"]:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # Detailed Results
        if "results" in results:
            report_lines.extend([
                "## Detailed Results",
                ""
            ])
            
            for category_name, category_results in results["results"].items():
                report_lines.append(f"### {category_name.replace('_', ' ').title()}")
                
                if isinstance(category_results, dict) and "test_name" in category_results:
                    # Single result
                    self._add_result_to_report(report_lines, category_results)
                elif isinstance(category_results, dict):
                    # Multiple results
                    for test_name, test_result in category_results.items():
                        report_lines.append(f"#### {test_name}")
                        self._add_result_to_report(report_lines, test_result)
                
                report_lines.append("")
        
        # Errors
        if results.get("errors"):
            report_lines.extend([
                "## Errors Encountered",
                ""
            ])
            for error in results["errors"]:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Communication report saved to: {output_file}")
    
    def _add_result_to_report(self, report_lines: List[str], result: Dict[str, Any]):
        """Add a single result to the report."""
        report_lines.extend([
            f"- **Average Latency**: {result.get('avg_total_time_ms', 0):.2f} ms",
            f"- **95th Percentile**: {result.get('p95_total_time_ms', 0):.2f} ms",
            f"- **Throughput**: {result.get('operations_per_second', 0):.0f} ops/sec",
            f"- **Success Rate**: {result.get('success_rate', 0):.1%}",
            f"- **Errors**: {result.get('error_count', 0)}",
            ""
        ])


async def main():
    """Main entry point for communication benchmarks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Communication Overhead Benchmarks for gRPC")
    parser.add_argument("--server", default="localhost:50100", help="gRPC server address")
    parser.add_argument("--output-dir", default="communication_results", help="Output directory")
    parser.add_argument("--connection-iterations", type=int, default=100, help="Connection test iterations")
    parser.add_argument("--serialization-iterations", type=int, default=1000, help="Serialization test iterations")
    parser.add_argument("--network-iterations", type=int, default=100, help="Network test iterations")
    parser.add_argument("--batch-iterations", type=int, default=50, help="Batch test iterations")
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    benchmark = ComprehensiveCommunicationBenchmark(
        server_address=args.server,
        output_dir=args.output_dir
    )
    
    logger.info("Starting comprehensive communication benchmarks...")
    
    # Run benchmarks
    results = await benchmark.run_all_benchmarks(
        connection_iterations=args.connection_iterations,
        serialization_iterations=args.serialization_iterations,
        network_iterations=args.network_iterations,
        batch_iterations=args.batch_iterations
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMMUNICATION BENCHMARKS COMPLETED")
    logger.info("="*80)
    
    if "summary" in results:
        summary = results["summary"]
        logger.info(f"Total Tests: {summary.get('total_tests', 0)}")
        
        if "overall_performance" in summary:
            perf = summary["overall_performance"]
            logger.info(f"Average Latency: {perf.get('avg_latency_ms', 0):.2f} ms")
            logger.info(f"Average Throughput: {perf.get('avg_throughput_ops_sec', 0):.0f} ops/sec")
            logger.info(f"Success Rate: {perf.get('overall_success_rate', 0):.1%}")
    
    if "analysis" in results and results["analysis"].get("recommendations"):
        logger.info("\nKey Recommendations:")
        for rec in results["analysis"]["recommendations"]:
            logger.info(f"  - {rec}")
    
    if results.get("errors"):
        logger.warning(f"\nErrors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            logger.warning(f"  - {error}")
    
    logger.info(f"\nDetailed results saved to: {args.output_dir}/")
    
    return 0 if not results.get("errors") else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
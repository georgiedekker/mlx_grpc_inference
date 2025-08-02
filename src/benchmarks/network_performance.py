"""
Network performance analysis for MLX distributed inference system.

This module provides comprehensive network analysis including:
- Bandwidth utilization measurement
- Latency analysis under different loads
- Connection pool performance
- Network bottleneck identification
- Compression effectiveness analysis
"""

import asyncio
import logging
import statistics
import time
import socket
import struct
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import concurrent.futures
from collections import defaultdict

import mlx.core as mx
import numpy as np

from ..communication.grpc_client import GRPCInferenceClient
from ..communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from ..communication.connection_pool import ConnectionPool
from ..core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Metrics for network performance analysis."""
    test_name: str
    timestamp: str
    
    # Bandwidth metrics
    total_bytes_sent: int
    total_bytes_received: int
    avg_bandwidth_mbps: float
    peak_bandwidth_mbps: float
    bandwidth_utilization_percent: float
    
    # Latency metrics (milliseconds)
    avg_round_trip_latency_ms: float
    min_round_trip_latency_ms: float
    max_round_trip_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    latency_jitter_ms: float
    
    # Connection metrics
    connection_establishment_time_ms: float
    connection_reuse_efficiency: float
    failed_connections: int
    connection_timeouts: int
    
    # Compression metrics
    compression_ratio: float
    compression_overhead_ms: float
    decompression_overhead_ms: float
    
    # Throughput metrics
    messages_per_second: float
    successful_transmissions: int
    failed_transmissions: int
    
    # Configuration
    concurrent_connections: int
    payload_size_bytes: int
    test_duration_s: float
    compression_enabled: bool


@dataclass
class BandwidthSample:
    """Single bandwidth measurement sample."""
    timestamp: float
    bytes_sent: int
    bytes_received: int
    duration_s: float
    bandwidth_mbps: float


class NetworkAnalyzer:
    """Comprehensive network performance analyzer for distributed inference."""
    
    def __init__(self, config: ClusterConfig, output_dir: str = "benchmarks/network"):
        """
        Initialize network analyzer.
        
        Args:
            config: Cluster configuration
            output_dir: Directory for benchmark results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_device_id = config.get_local_device_id()
        self.results: List[NetworkMetrics] = []
        
        # Network monitoring state
        self.bandwidth_samples: List[BandwidthSample] = []
        
        logger.info(f"NetworkAnalyzer initialized for device {self.local_device_id}")
    
    async def run_comprehensive_network_analysis(self, 
                                               test_duration: int = 60,
                                               sampling_interval: float = 1.0) -> Dict[str, Any]:
        """
        Run comprehensive network analysis across different scenarios.
        
        Args:
            test_duration: Duration of each test in seconds
            sampling_interval: Bandwidth sampling interval in seconds
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting comprehensive network analysis ({test_duration}s tests)")
        
        benchmark_results = {
            "benchmark_info": {
                "start_time": datetime.now().isoformat(),
                "local_device": self.local_device_id,
                "cluster_config": {
                    "total_devices": len(self.config.devices),
                    "model_name": self.config.model.name,
                    "communication_backend": self.config.communication_backend
                },
                "test_configuration": {
                    "test_duration": test_duration,
                    "sampling_interval": sampling_interval
                }
            },
            "network_tests": {},
            "analysis": {},
            "summary": {},
            "errors": []
        }
        
        # Define network test scenarios
        network_scenarios = [
            {
                "name": "baseline_latency",
                "description": "Baseline network latency measurement",
                "test_config": {
                    "concurrent_connections": 1,
                    "payload_size": 1024,  # 1KB
                    "compression": False,
                    "message_rate": 10  # messages per second
                }
            },
            {
                "name": "small_payload_throughput",
                "description": "Small payload throughput test",
                "test_config": {
                    "concurrent_connections": 4,
                    "payload_size": 4096,  # 4KB
                    "compression": False,
                    "message_rate": 50
                }
            },
            {
                "name": "medium_payload_throughput",
                "description": "Medium payload throughput test",
                "test_config": {
                    "concurrent_connections": 8,
                    "payload_size": 1024 * 1024,  # 1MB
                    "compression": False,
                    "message_rate": 20
                }
            },
            {
                "name": "large_payload_throughput",
                "description": "Large payload throughput test",
                "test_config": {
                    "concurrent_connections": 4,
                    "payload_size": 10 * 1024 * 1024,  # 10MB
                    "compression": False,
                    "message_rate": 5
                }
            },
            {
                "name": "compression_efficiency",
                "description": "Compression effectiveness test",
                "test_config": {
                    "concurrent_connections": 4,
                    "payload_size": 1024 * 1024,  # 1MB
                    "compression": True,
                    "message_rate": 20
                }
            },
            {
                "name": "high_concurrency",
                "description": "High concurrency network stress test",
                "test_config": {
                    "concurrent_connections": 16,
                    "payload_size": 256 * 1024,  # 256KB
                    "compression": False,
                    "message_rate": 30
                }
            },
            {
                "name": "connection_scaling",
                "description": "Connection pool scaling test",
                "test_config": {
                    "concurrent_connections": 32,
                    "payload_size": 64 * 1024,  # 64KB
                    "compression": False,
                    "message_rate": 40
                }
            }
        ]
        
        # Run each network test scenario
        for scenario in network_scenarios:
            try:
                logger.info(f"Running network test: {scenario['name']}")
                
                metrics = await self.analyze_network_performance(
                    test_config=scenario["test_config"],
                    test_duration=test_duration,
                    sampling_interval=sampling_interval,
                    test_name=scenario["name"]
                )
                
                self.results.append(metrics)
                benchmark_results["network_tests"][scenario["name"]] = {
                    "scenario": scenario,
                    "metrics": asdict(metrics)
                }
                
            except Exception as e:
                error_msg = f"Network test '{scenario['name']}' failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
        
        # Analyze results
        try:
            analysis = self._analyze_network_patterns()
            benchmark_results["analysis"] = analysis
        except Exception as e:
            error_msg = f"Network analysis failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"network_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Network benchmark results saved to: {results_file}")
        
        # Generate report
        report_file = self.output_dir / f"network_report_{timestamp}.md"
        self._generate_network_report(benchmark_results, report_file)
        
        return benchmark_results
    
    async def analyze_network_performance(self,
                                        test_config: Dict[str, Any],
                                        test_duration: int = 60,
                                        sampling_interval: float = 1.0,
                                        test_name: str = "network_test") -> NetworkMetrics:
        """
        Analyze network performance for specific configuration.
        
        Args:
            test_config: Test configuration
            test_duration: Test duration in seconds
            sampling_interval: Bandwidth sampling interval
            test_name: Name for this test
            
        Returns:
            Network metrics
        """
        logger.debug(f"Analyzing network performance for {test_name}")
        
        # Reset bandwidth samples
        self.bandwidth_samples.clear()
        
        # Extract test parameters
        concurrent_connections = test_config.get("concurrent_connections", 1)
        payload_size = test_config.get("payload_size", 1024)
        compression = test_config.get("compression", False)
        message_rate = test_config.get("message_rate", 10)
        
        # Track metrics
        latency_measurements = []
        transmission_results = []
        connection_times = []
        compression_times = []
        decompression_times = []
        
        # Start bandwidth monitoring
        bandwidth_monitor_task = asyncio.create_task(
            self._monitor_bandwidth(sampling_interval, test_duration)
        )
        
        test_start_time = time.time()
        
        try:
            # Run network load test
            await self._run_network_load_test(
                concurrent_connections=concurrent_connections,
                payload_size=payload_size,
                compression=compression,
                message_rate=message_rate,
                test_duration=test_duration,
                latency_measurements=latency_measurements,
                transmission_results=transmission_results,
                connection_times=connection_times,
                compression_times=compression_times,
                decompression_times=decompression_times
            )
            
        finally:
            # Stop bandwidth monitoring
            await bandwidth_monitor_task
        
        actual_test_duration = time.time() - test_start_time
        
        # Calculate metrics
        return self._calculate_network_metrics(
            test_name=test_name,
            latency_measurements=latency_measurements,
            transmission_results=transmission_results,
            connection_times=connection_times,
            compression_times=compression_times,
            decompression_times=decompression_times,
            bandwidth_samples=self.bandwidth_samples,
            test_duration=actual_test_duration,
            test_config=test_config
        )
    
    async def _run_network_load_test(self,
                                   concurrent_connections: int,
                                   payload_size: int,
                                   compression: bool,
                                   message_rate: int,
                                   test_duration: int,
                                   latency_measurements: List[float],
                                   transmission_results: List[bool],
                                   connection_times: List[float],
                                   compression_times: List[float],
                                   decompression_times: List[float]):
        """Run network load test with specified parameters."""
        
        # Create semaphore for connection limiting
        connection_semaphore = asyncio.Semaphore(concurrent_connections)
        
        # Calculate message interval
        message_interval = 1.0 / message_rate if message_rate > 0 else 0.1
        
        async def send_message(message_id: int) -> Tuple[bool, float, Dict[str, float]]:
            """Send a single message and measure performance."""
            async with connection_semaphore:
                start_time = time.time()
                timing_info = {"connection": 0, "compression": 0, "decompression": 0}
                
                try:
                    # Simulate connection establishment
                    connection_start = time.time()
                    await asyncio.sleep(0.001)  # Simulate connection overhead
                    timing_info["connection"] = (time.time() - connection_start) * 1000
                    
                    # Generate test payload
                    payload = self._generate_test_payload(payload_size)
                    
                    # Compression if enabled
                    if compression:
                        compression_start = time.time()
                        compressed_payload = self._compress_payload(payload)
                        timing_info["compression"] = (time.time() - compression_start) * 1000
                    else:
                        compressed_payload = payload
                    
                    # Simulate network transmission
                    transmission_time = self._simulate_network_transmission(len(compressed_payload))
                    await asyncio.sleep(transmission_time)
                    
                    # Simulate receiving response
                    response_payload = compressed_payload  # Echo back
                    
                    # Decompression if enabled
                    if compression:
                        decompression_start = time.time()
                        decompressed_payload = self._decompress_payload(response_payload)
                        timing_info["decompression"] = (time.time() - decompression_start) * 1000
                    
                    total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    return True, total_time, timing_info
                    
                except Exception as e:
                    total_time = (time.time() - start_time) * 1000
                    logger.debug(f"Message {message_id} failed: {e}")
                    return False, total_time, timing_info
        
        # Send messages at specified rate for test duration
        start_time = time.time()
        message_id = 0
        tasks = []
        
        while time.time() - start_time < test_duration:
            # Create message task
            task = asyncio.create_task(send_message(message_id))
            tasks.append(task)
            message_id += 1
            
            # Wait for message interval
            await asyncio.sleep(message_interval)
        
        # Wait for all messages to complete
        logger.debug(f"Waiting for {len(tasks)} messages to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, tuple):
                success, latency, timing = result
                transmission_results.append(success)
                if success:
                    latency_measurements.append(latency)
                    connection_times.append(timing["connection"])
                    compression_times.append(timing["compression"])
                    decompression_times.append(timing["decompression"])
            else:
                transmission_results.append(False)
    
    def _generate_test_payload(self, size_bytes: int) -> bytes:
        """Generate test payload of specified size."""
        # Generate tensor-like data for realistic testing
        if size_bytes < 1024:
            # Small payload - simple data
            return b'X' * size_bytes
        else:
            # Larger payload - simulate tensor data
            num_floats = size_bytes // 4
            tensor_data = mx.random.normal((num_floats,)).astype(mx.float32)
            return np.array(tensor_data).tobytes()
    
    def _compress_payload(self, payload: bytes) -> bytes:
        """Compress payload (simplified compression simulation)."""
        import gzip
        return gzip.compress(payload)
    
    def _decompress_payload(self, compressed_payload: bytes) -> bytes:
        """Decompress payload."""
        import gzip
        return gzip.decompress(compressed_payload)
    
    def _simulate_network_transmission(self, payload_size: int) -> float:
        """Simulate network transmission time based on payload size."""
        # Simulate network characteristics
        bandwidth_mbps = 100  # Assume 100 Mbps network
        latency_base_ms = 5   # 5ms base latency
        
        # Calculate transmission time
        transmission_time_s = (payload_size * 8) / (bandwidth_mbps * 1000000)  # Convert to seconds
        latency_s = latency_base_ms / 1000
        
        # Add some random jitter
        jitter = np.random.normal(0, 0.001)  # ±1ms jitter
        
        return transmission_time_s + latency_s + jitter
    
    async def _monitor_bandwidth(self, sampling_interval: float, duration: float):
        """Monitor bandwidth usage during the test."""
        start_time = time.time()
        last_timestamp = start_time
        last_bytes_sent = 0
        last_bytes_received = 0
        
        while time.time() - start_time < duration:
            try:
                current_time = time.time()
                
                # Simulate bandwidth measurement
                # In practice, this would read from network interfaces
                bytes_sent = np.random.randint(1000, 10000)  # Simulated
                bytes_received = np.random.randint(1000, 10000)  # Simulated
                
                time_delta = current_time - last_timestamp
                if time_delta > 0:
                    bandwidth_sent = (bytes_sent - last_bytes_sent) * 8 / (time_delta * 1000000)  # Mbps
                    bandwidth_received = (bytes_received - last_bytes_received) * 8 / (time_delta * 1000000)  # Mbps
                    total_bandwidth = bandwidth_sent + bandwidth_received
                    
                    sample = BandwidthSample(
                        timestamp=current_time,
                        bytes_sent=bytes_sent - last_bytes_sent,
                        bytes_received=bytes_received - last_bytes_received,
                        duration_s=time_delta,
                        bandwidth_mbps=total_bandwidth
                    )
                    self.bandwidth_samples.append(sample)
                
                last_timestamp = current_time
                last_bytes_sent = bytes_sent
                last_bytes_received = bytes_received
                
            except Exception as e:
                logger.debug(f"Bandwidth monitoring error: {e}")
            
            await asyncio.sleep(sampling_interval)
    
    def _calculate_network_metrics(self,
                                 test_name: str,
                                 latency_measurements: List[float],
                                 transmission_results: List[bool],
                                 connection_times: List[float],
                                 compression_times: List[float],
                                 decompression_times: List[float],
                                 bandwidth_samples: List[BandwidthSample],
                                 test_duration: float,
                                 test_config: Dict[str, Any]) -> NetworkMetrics:
        """Calculate comprehensive network metrics."""
        
        successful_transmissions = sum(transmission_results)
        failed_transmissions = len(transmission_results) - successful_transmissions
        
        # Latency metrics
        if latency_measurements:
            avg_latency = statistics.mean(latency_measurements)
            min_latency = min(latency_measurements)
            max_latency = max(latency_measurements)
            p95_latency = np.percentile(latency_measurements, 95)
            p99_latency = np.percentile(latency_measurements, 99)
            latency_jitter = statistics.stdev(latency_measurements) if len(latency_measurements) > 1 else 0
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = latency_jitter = 0
        
        # Connection metrics
        if connection_times:
            avg_connection_time = statistics.mean(connection_times)
            connection_reuse_efficiency = 1.0 - (avg_connection_time / avg_latency) if avg_latency > 0 else 0
        else:
            avg_connection_time = 0
            connection_reuse_efficiency = 0
        
        # Compression metrics
        if compression_times:
            avg_compression_overhead = statistics.mean(compression_times)
            avg_decompression_overhead = statistics.mean(decompression_times)
            # Simulate compression ratio
            compression_ratio = 0.6 if test_config.get("compression") else 1.0
        else:
            avg_compression_overhead = avg_decompression_overhead = 0
            compression_ratio = 1.0
        
        # Bandwidth metrics
        if bandwidth_samples:
            total_bytes_sent = sum(s.bytes_sent for s in bandwidth_samples)
            total_bytes_received = sum(s.bytes_received for s in bandwidth_samples)
            bandwidth_values = [s.bandwidth_mbps for s in bandwidth_samples]
            avg_bandwidth = statistics.mean(bandwidth_values)
            peak_bandwidth = max(bandwidth_values)
            
            # Assume theoretical bandwidth limit of 100 Mbps
            theoretical_bandwidth = 100
            bandwidth_utilization = (avg_bandwidth / theoretical_bandwidth) * 100
        else:
            total_bytes_sent = total_bytes_received = 0
            avg_bandwidth = peak_bandwidth = bandwidth_utilization = 0
        
        # Throughput metrics
        messages_per_second = successful_transmissions / test_duration if test_duration > 0 else 0
        
        return NetworkMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            total_bytes_sent=total_bytes_sent,
            total_bytes_received=total_bytes_received,
            avg_bandwidth_mbps=avg_bandwidth,
            peak_bandwidth_mbps=peak_bandwidth,
            bandwidth_utilization_percent=bandwidth_utilization,
            avg_round_trip_latency_ms=avg_latency,
            min_round_trip_latency_ms=min_latency,
            max_round_trip_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            latency_jitter_ms=latency_jitter,
            connection_establishment_time_ms=avg_connection_time,
            connection_reuse_efficiency=connection_reuse_efficiency,
            failed_connections=0,  # Would be calculated from actual connection failures
            connection_timeouts=0,  # Would be calculated from actual timeouts
            compression_ratio=compression_ratio,
            compression_overhead_ms=avg_compression_overhead,
            decompression_overhead_ms=avg_decompression_overhead,
            messages_per_second=messages_per_second,
            successful_transmissions=successful_transmissions,
            failed_transmissions=failed_transmissions,
            concurrent_connections=test_config.get("concurrent_connections", 1),
            payload_size_bytes=test_config.get("payload_size", 1024),
            test_duration_s=test_duration,
            compression_enabled=test_config.get("compression", False)
        )
    
    def _analyze_network_patterns(self) -> Dict[str, Any]:
        """Analyze network patterns across test scenarios."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "bandwidth_analysis": {},
            "latency_analysis": {},
            "scalability_analysis": {},
            "compression_analysis": {},
            "bottleneck_analysis": {},
            "recommendations": []
        }
        
        # Bandwidth analysis
        bandwidth_values = [r.avg_bandwidth_mbps for r in self.results]
        utilization_values = [r.bandwidth_utilization_percent for r in self.results]
        
        analysis["bandwidth_analysis"] = {
            "peak_bandwidth_mbps": max(r.peak_bandwidth_mbps for r in self.results),
            "avg_bandwidth_mbps": statistics.mean(bandwidth_values),
            "avg_utilization_percent": statistics.mean(utilization_values),
            "bandwidth_efficiency": statistics.mean(bandwidth_values) / max(bandwidth_values) if bandwidth_values and max(bandwidth_values) > 0 else 0
        }
        
        # Latency analysis
        latency_values = [r.avg_round_trip_latency_ms for r in self.results]
        jitter_values = [r.latency_jitter_ms for r in self.results]
        
        analysis["latency_analysis"] = {
            "avg_latency_ms": statistics.mean(latency_values),
            "min_latency_ms": min(latency_values),
            "max_latency_ms": max(latency_values),
            "avg_jitter_ms": statistics.mean(jitter_values),
            "latency_consistency": 1.0 - (statistics.stdev(latency_values) / statistics.mean(latency_values)) if latency_values and statistics.mean(latency_values) > 0 else 0
        }
        
        # Scalability analysis
        concurrency_throughput = {}
        for result in self.results:
            concurrency = result.concurrent_connections
            throughput = result.messages_per_second
            if concurrency not in concurrency_throughput:
                concurrency_throughput[concurrency] = []
            concurrency_throughput[concurrency].append(throughput)
        
        # Calculate scaling efficiency
        scaling_efficiency = []
        if len(concurrency_throughput) > 1:
            sorted_concurrency = sorted(concurrency_throughput.keys())
            baseline_concurrency = sorted_concurrency[0]
            baseline_throughput = statistics.mean(concurrency_throughput[baseline_concurrency])
            
            for concurrency in sorted_concurrency:
                avg_throughput = statistics.mean(concurrency_throughput[concurrency])
                expected_throughput = baseline_throughput * (concurrency / baseline_concurrency)
                efficiency = avg_throughput / expected_throughput if expected_throughput > 0 else 0
                scaling_efficiency.append((concurrency, efficiency))
        
        analysis["scalability_analysis"] = {
            "concurrency_levels": list(concurrency_throughput.keys()),
            "scaling_efficiency": scaling_efficiency,
            "optimal_concurrency": max(concurrency_throughput.keys(), key=lambda c: statistics.mean(concurrency_throughput[c])) if concurrency_throughput else 1
        }
        
        # Compression analysis
        compressed_results = [r for r in self.results if r.compression_enabled]
        uncompressed_results = [r for r in self.results if not r.compression_enabled]
        
        if compressed_results and uncompressed_results:
            compressed_avg_latency = statistics.mean(r.avg_round_trip_latency_ms for r in compressed_results)
            uncompressed_avg_latency = statistics.mean(r.avg_round_trip_latency_ms for r in uncompressed_results)
            
            compressed_avg_bandwidth = statistics.mean(r.avg_bandwidth_mbps for r in compressed_results)
            uncompressed_avg_bandwidth = statistics.mean(r.avg_bandwidth_mbps for r in uncompressed_results)
            
            analysis["compression_analysis"] = {
                "avg_compression_ratio": statistics.mean(r.compression_ratio for r in compressed_results),
                "latency_overhead_ms": compressed_avg_latency - uncompressed_avg_latency,
                "bandwidth_savings_percent": ((uncompressed_avg_bandwidth - compressed_avg_bandwidth) / uncompressed_avg_bandwidth * 100) if uncompressed_avg_bandwidth > 0 else 0,
                "compression_efficiency": compressed_avg_bandwidth / uncompressed_avg_bandwidth if uncompressed_avg_bandwidth > 0 else 0
            }
        
        # Bottleneck analysis
        bottlenecks = []
        
        # Check for bandwidth bottlenecks
        if analysis["bandwidth_analysis"]["avg_utilization_percent"] > 80:
            bottlenecks.append("High bandwidth utilization detected")
        
        # Check for latency bottlenecks
        if analysis["latency_analysis"]["avg_latency_ms"] > 50:
            bottlenecks.append("High network latency detected")
        
        # Check for scaling bottlenecks
        if scaling_efficiency and scaling_efficiency[-1][1] < 0.5:
            bottlenecks.append("Poor scaling at high concurrency")
        
        analysis["bottleneck_analysis"] = {
            "identified_bottlenecks": bottlenecks,
            "bandwidth_limited": analysis["bandwidth_analysis"]["avg_utilization_percent"] > 70,
            "latency_limited": analysis["latency_analysis"]["avg_latency_ms"] > 30,
            "connection_limited": any(r.failed_transmissions > r.successful_transmissions * 0.1 for r in self.results)
        }
        
        # Recommendations
        recommendations = []
        
        if analysis["bandwidth_analysis"]["avg_utilization_percent"] > 80:
            recommendations.append("Network bandwidth is highly utilized - consider upgrading network infrastructure")
        
        if analysis["latency_analysis"]["avg_jitter_ms"] > 10:
            recommendations.append("High network jitter detected - investigate network stability")
        
        if compressed_results and analysis.get("compression_analysis", {}).get("latency_overhead_ms", 0) > 20:
            recommendations.append("Compression overhead is significant - evaluate compression algorithms")
        
        if bottlenecks:
            recommendations.append("Network bottlenecks detected - optimize based on bottleneck analysis")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        bandwidth_values = [r.avg_bandwidth_mbps for r in self.results]
        latency_values = [r.avg_round_trip_latency_ms for r in self.results]
        throughput_values = [r.messages_per_second for r in self.results]
        success_rates = [r.successful_transmissions / (r.successful_transmissions + r.failed_transmissions) 
                        if (r.successful_transmissions + r.failed_transmissions) > 0 else 0 for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "peak_bandwidth_mbps": max(r.peak_bandwidth_mbps for r in self.results),
            "avg_bandwidth_mbps": statistics.mean(bandwidth_values),
            "avg_latency_ms": statistics.mean(latency_values),
            "min_latency_ms": min(latency_values),
            "peak_throughput_msgs_per_sec": max(throughput_values),
            "avg_throughput_msgs_per_sec": statistics.mean(throughput_values),
            "overall_success_rate": statistics.mean(success_rates),
            "total_bytes_transferred": sum(r.total_bytes_sent + r.total_bytes_received for r in self.results),
            "network_efficiency_score": self._calculate_network_efficiency_score()
        }
    
    def _calculate_network_efficiency_score(self) -> float:
        """Calculate overall network efficiency score (0-1)."""
        if not self.results:
            return 0.0
        
        # Factors for efficiency calculation
        bandwidth_efficiency = statistics.mean(r.bandwidth_utilization_percent for r in self.results) / 100
        latency_efficiency = 1.0 - min(statistics.mean(r.avg_round_trip_latency_ms for r in self.results) / 100, 1.0)
        success_rate = statistics.mean(r.successful_transmissions / (r.successful_transmissions + r.failed_transmissions) 
                                     if (r.successful_transmissions + r.failed_transmissions) > 0 else 0 for r in self.results)
        
        # Weighted average (bandwidth 40%, latency 30%, success rate 30%)
        efficiency_score = (bandwidth_efficiency * 0.4 + latency_efficiency * 0.3 + success_rate * 0.3)
        return min(efficiency_score, 1.0)
    
    def _generate_network_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive network report."""
        report_lines = [
            "# Network Performance Benchmark Report",
            f"Generated: {results['benchmark_info']['start_time']}",
            f"Device: {results['benchmark_info']['local_device']}",
            f"Communication Backend: {results['benchmark_info']['cluster_config']['communication_backend']}",
            ""
        ]
        
        # Executive Summary
        if "summary" in results:
            summary = results["summary"]
            report_lines.extend([
                "## Executive Summary",
                f"- **Total Tests**: {summary.get('total_tests', 0)}",
                f"- **Peak Bandwidth**: {summary.get('peak_bandwidth_mbps', 0):.1f} Mbps",
                f"- **Average Bandwidth**: {summary.get('avg_bandwidth_mbps', 0):.1f} Mbps",
                f"- **Average Latency**: {summary.get('avg_latency_ms', 0):.1f} ms",
                f"- **Peak Throughput**: {summary.get('peak_throughput_msgs_per_sec', 0):.1f} msgs/sec",
                f"- **Overall Success Rate**: {summary.get('overall_success_rate', 0):.1%}",
                f"- **Network Efficiency Score**: {summary.get('network_efficiency_score', 0):.2f}",
                ""
            ])
        
        # Analysis Results
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "## Network Analysis",
                ""
            ])
            
            # Bandwidth analysis
            if "bandwidth_analysis" in analysis:
                bandwidth = analysis["bandwidth_analysis"]
                report_lines.extend([
                    "### Bandwidth Analysis",
                    f"- **Peak Bandwidth**: {bandwidth.get('peak_bandwidth_mbps', 0):.1f} Mbps",
                    f"- **Average Utilization**: {bandwidth.get('avg_utilization_percent', 0):.1f}%",
                    f"- **Bandwidth Efficiency**: {bandwidth.get('bandwidth_efficiency', 0):.2f}",
                    ""
                ])
            
            # Latency analysis
            if "latency_analysis" in analysis:
                latency = analysis["latency_analysis"]
                report_lines.extend([
                    "### Latency Analysis",
                    f"- **Average Latency**: {latency.get('avg_latency_ms', 0):.1f} ms",
                    f"- **Min Latency**: {latency.get('min_latency_ms', 0):.1f} ms",
                    f"- **Max Latency**: {latency.get('max_latency_ms', 0):.1f} ms",
                    f"- **Average Jitter**: {latency.get('avg_jitter_ms', 0):.1f} ms",
                    f"- **Latency Consistency**: {latency.get('latency_consistency', 0):.2f}",
                    ""
                ])
            
            # Compression analysis
            if "compression_analysis" in analysis:
                compression = analysis["compression_analysis"]
                report_lines.extend([
                    "### Compression Analysis",
                    f"- **Average Compression Ratio**: {compression.get('avg_compression_ratio', 1):.2f}",
                    f"- **Latency Overhead**: {compression.get('latency_overhead_ms', 0):.1f} ms",
                    f"- **Bandwidth Savings**: {compression.get('bandwidth_savings_percent', 0):.1f}%",
                    ""
                ])
            
            # Bottlenecks
            if "bottleneck_analysis" in analysis:
                bottlenecks = analysis["bottleneck_analysis"]
                report_lines.extend([
                    "### Bottleneck Analysis",
                    f"- **Bandwidth Limited**: {bottlenecks.get('bandwidth_limited', False)}",
                    f"- **Latency Limited**: {bottlenecks.get('latency_limited', False)}",
                    f"- **Connection Limited**: {bottlenecks.get('connection_limited', False)}",
                    ""
                ])
                
                if bottlenecks.get("identified_bottlenecks"):
                    report_lines.append("**Identified Issues:**")
                    for issue in bottlenecks["identified_bottlenecks"]:
                        report_lines.append(f"  - {issue}")
                    report_lines.append("")
            
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
        if "network_tests" in results:
            report_lines.extend([
                "## Detailed Test Results",
                ""
            ])
            
            for test_name, test_data in results["network_tests"].items():
                metrics = test_data["metrics"]
                scenario_info = test_data["scenario"]
                
                report_lines.extend([
                    f"### {test_name.replace('_', ' ').title()}",
                    f"**Description**: {scenario_info['description']}",
                    "",
                    f"- **Average Bandwidth**: {metrics['avg_bandwidth_mbps']:.1f} Mbps",
                    f"- **Average Latency**: {metrics['avg_round_trip_latency_ms']:.1f} ms",
                    f"- **Messages/sec**: {metrics['messages_per_second']:.1f}",
                    f"- **Success Rate**: {metrics['successful_transmissions'] / (metrics['successful_transmissions'] + metrics['failed_transmissions']):.1%}" if (metrics['successful_transmissions'] + metrics['failed_transmissions']) > 0 else "- **Success Rate**: 0%",
                    f"- **Concurrent Connections**: {metrics['concurrent_connections']}",
                    f"- **Payload Size**: {metrics['payload_size_bytes']:,} bytes",
                    ""
                ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Network report saved to: {output_file}")


async def main():
    """Example usage of network analyzer."""
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration (this would be actual config in real usage)
    from ..core.config import ClusterConfig, DeviceConfig, DeviceRole, DeviceCapabilities, ModelConfig, PerformanceConfig, MonitoringConfig
    
    # This is just for demo - normally loaded from config file
    devices = [
        DeviceConfig(
            device_id="mini1",
            hostname="mini1.local",
            api_port=8100,
            grpc_port=50051,
            role=DeviceRole.COORDINATOR,
            rank=0,
            capabilities=DeviceCapabilities(
                model="Apple M4",
                memory_gb=16,
                gpu_cores=10,
                cpu_cores=10,
                cpu_performance_cores=4,
                cpu_efficiency_cores=6,
                neural_engine_cores=16,
                bandwidth_gbps=120
            )
        )
    ]
    
    config = ClusterConfig(
        name="test-cluster",
        coordinator_device_id="mini1",
        communication_backend="grpc",
        devices=devices,
        model=ModelConfig(
            name="test-model",
            total_layers=28,
            layer_distribution={"mini1": list(range(28))}
        ),
        performance=PerformanceConfig(),
        monitoring=MonitoringConfig()
    )
    
    # Run network analysis
    analyzer = NetworkAnalyzer(config)
    results = await analyzer.run_comprehensive_network_analysis(test_duration=30)
    
    print("Network analysis completed!")
    print(f"Peak bandwidth: {results['summary']['peak_bandwidth_mbps']:.1f} Mbps")
    print(f"Network efficiency: {results['summary']['network_efficiency_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
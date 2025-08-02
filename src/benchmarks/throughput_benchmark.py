"""
Throughput benchmarking for MLX distributed inference system.

This module provides comprehensive throughput analysis including:
- Requests per second measurement
- Tokens per second analysis
- Concurrent request handling
- Batch processing efficiency
- Resource utilization correlation
"""

import asyncio
import logging
import statistics
import time
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
from ..core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """Metrics for throughput analysis."""
    test_name: str
    timestamp: str
    
    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float
    avg_concurrent_requests: float
    
    # Request metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    
    # Timing metrics (seconds)
    test_duration_s: float
    avg_request_time_s: float
    p95_request_time_s: float
    p99_request_time_s: float
    
    # Batch metrics
    avg_batch_size: float
    batch_efficiency: float  # Actual vs theoretical throughput
    
    # Resource metrics
    avg_cpu_utilization: float
    avg_memory_usage_gb: float
    peak_memory_usage_gb: float
    
    # Configuration
    max_concurrent_requests: int
    test_duration: int
    input_config: Dict[str, Any]


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: str
    start_time: float
    end_time: float
    success: bool
    error_message: Optional[str]
    input_tokens: int
    output_tokens: int
    batch_size: int


class ThroughputBenchmark:
    """Comprehensive throughput benchmarking for distributed inference."""
    
    def __init__(self, config: ClusterConfig, output_dir: str = "benchmarks/throughput"):
        """
        Initialize throughput benchmark.
        
        Args:
            config: Cluster configuration
            output_dir: Directory for benchmark results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_device_id = config.get_local_device_id()
        self.results: List[ThroughputMetrics] = []
        self.request_history: List[RequestResult] = []
        
        logger.info(f"ThroughputBenchmark initialized for device {self.local_device_id}")
    
    async def run_comprehensive_throughput_analysis(self, 
                                                  test_duration: int = 60,
                                                  warmup_duration: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive throughput analysis across different load patterns.
        
        Args:
            test_duration: Duration of each test in seconds
            warmup_duration: Warmup duration in seconds
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting comprehensive throughput analysis ({test_duration}s tests)")
        
        benchmark_results = {
            "benchmark_info": {
                "start_time": datetime.now().isoformat(),
                "local_device": self.local_device_id,
                "cluster_config": {
                    "total_devices": len(self.config.devices),
                    "model_name": self.config.model.name,
                    "total_layers": self.config.model.total_layers
                },
                "test_configuration": {
                    "test_duration": test_duration,
                    "warmup_duration": warmup_duration
                }
            },
            "load_tests": {},
            "analysis": {},
            "summary": {},
            "errors": []
        }
        
        # Define load test scenarios
        load_scenarios = [
            {
                "name": "single_request",
                "description": "Single request at a time (baseline)",
                "max_concurrent": 1,
                "request_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                }
            },
            {
                "name": "low_concurrency",
                "description": "Low concurrent load (2-4 requests)",
                "max_concurrent": 4,
                "request_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                }
            },
            {
                "name": "medium_concurrency",
                "description": "Medium concurrent load (8-16 requests)",
                "max_concurrent": 16,
                "request_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                }
            },
            {
                "name": "high_concurrency",
                "description": "High concurrent load (32+ requests)",
                "max_concurrent": 32,
                "request_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                }
            },
            {
                "name": "batch_processing",
                "description": "Batch processing efficiency test",
                "max_concurrent": 8,
                "request_config": {
                    "input_shape": [4, 512],
                    "sequence_length": 512,
                    "batch_size": 4
                }
            },
            {
                "name": "large_batch",
                "description": "Large batch processing test",
                "max_concurrent": 4,
                "request_config": {
                    "input_shape": [8, 512],
                    "sequence_length": 512,
                    "batch_size": 8
                }
            }
        ]
        
        # Run each load test scenario
        for scenario in load_scenarios:
            try:
                logger.info(f"Running throughput test: {scenario['name']}")
                
                metrics = await self.benchmark_throughput(
                    max_concurrent_requests=scenario["max_concurrent"],
                    test_duration=test_duration,
                    warmup_duration=warmup_duration,
                    input_config=scenario["request_config"],
                    test_name=scenario["name"]
                )
                
                self.results.append(metrics)
                benchmark_results["load_tests"][scenario["name"]] = {
                    "scenario": scenario,
                    "metrics": asdict(metrics)
                }
                
            except Exception as e:
                error_msg = f"Throughput test '{scenario['name']}' failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
        
        # Analyze results
        try:
            analysis = self._analyze_throughput_patterns()
            benchmark_results["analysis"] = analysis
        except Exception as e:
            error_msg = f"Throughput analysis failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"throughput_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Throughput benchmark results saved to: {results_file}")
        
        # Generate report
        report_file = self.output_dir / f"throughput_report_{timestamp}.md"
        self._generate_throughput_report(benchmark_results, report_file)
        
        return benchmark_results
    
    async def benchmark_throughput(self,
                                 max_concurrent_requests: int,
                                 test_duration: int = 60,
                                 warmup_duration: int = 10,
                                 input_config: Dict[str, Any] = None,
                                 test_name: str = "throughput_test") -> ThroughputMetrics:
        """
        Benchmark throughput for specific load configuration.
        
        Args:
            max_concurrent_requests: Maximum concurrent requests
            test_duration: Test duration in seconds
            warmup_duration: Warmup duration in seconds
            input_config: Input configuration
            test_name: Name for this test
            
        Returns:
            Throughput metrics
        """
        if input_config is None:
            input_config = {
                "input_shape": [1, 512],
                "sequence_length": 512,
                "batch_size": 1
            }
        
        logger.debug(f"Benchmarking {test_name}: max_concurrent={max_concurrent_requests}, duration={test_duration}s")
        
        # Reset request history for this test
        test_requests: List[RequestResult] = []
        request_counter = 0
        
        # Resource monitoring
        resource_samples = []
        
        # Warmup phase
        if warmup_duration > 0:
            logger.debug(f"Running {warmup_duration}s warmup...")
            await self._run_load_test(
                max_concurrent=max_concurrent_requests,
                duration=warmup_duration,
                input_config=input_config,
                collect_results=False
            )
        
        # Main test phase
        logger.debug(f"Running {test_duration}s throughput test...")
        
        test_start_time = time.time()
        
        # Start resource monitoring
        resource_monitor_task = asyncio.create_task(
            self._monitor_resources(resource_samples, test_duration)
        )
        
        # Run load test
        test_requests = await self._run_load_test(
            max_concurrent=max_concurrent_requests,
            duration=test_duration,
            input_config=input_config,
            collect_results=True
        )
        
        # Wait for resource monitoring to complete
        await resource_monitor_task
        
        actual_test_duration = time.time() - test_start_time
        
        # Calculate metrics
        return self._calculate_throughput_metrics(
            test_requests=test_requests,
            resource_samples=resource_samples,
            test_duration=actual_test_duration,
            max_concurrent_requests=max_concurrent_requests,
            input_config=input_config,
            test_name=test_name
        )
    
    async def _run_load_test(self,
                           max_concurrent: int,
                           duration: int,
                           input_config: Dict[str, Any],
                           collect_results: bool = True) -> List[RequestResult]:
        """Run load test with specified concurrency for specified duration."""
        
        requests = []
        request_counter = 0
        start_time = time.time()
        
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def make_request(request_id: str) -> Optional[RequestResult]:
            """Make a single request with concurrency control."""
            async with semaphore:
                return await self._execute_single_request(request_id, input_config)
        
        # Keep generating requests until duration is reached
        tasks = []
        while time.time() - start_time < duration:
            request_id = f"req_{request_counter:06d}"
            task = asyncio.create_task(make_request(request_id))
            tasks.append(task)
            request_counter += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        # Wait for all requests to complete
        if collect_results:
            logger.debug(f"Waiting for {len(tasks)} requests to complete...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, RequestResult):
                    requests.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Request failed: {result}")
        else:
            # For warmup, we don't need to wait for all results
            # Cancel remaining tasks after duration
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        return requests
    
    async def _execute_single_request(self, request_id: str, input_config: Dict[str, Any]) -> RequestResult:
        """Execute a single inference request."""
        start_time = time.time()
        
        try:
            # Generate input tensor
            input_shape = input_config["input_shape"]
            sequence_length = input_config["sequence_length"]
            batch_size = input_config["batch_size"]
            
            input_tensor = mx.random.randint(0, 1000, input_shape).astype(mx.int32)
            
            # Simulate inference (in practice, this would call actual inference pipeline)
            processing_time = self._estimate_processing_time(input_tensor, sequence_length)
            await asyncio.sleep(processing_time)
            
            # Simulate output generation
            output_tokens = sequence_length + 50  # Assume some generation
            
            end_time = time.time()
            
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                success=True,
                error_message=None,
                input_tokens=sequence_length,
                output_tokens=output_tokens,
                batch_size=batch_size
            )
            
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message=str(e),
                input_tokens=input_config["sequence_length"],
                output_tokens=0,
                batch_size=input_config["batch_size"]
            )
    
    def _estimate_processing_time(self, input_tensor: mx.array, sequence_length: int) -> float:
        """Estimate processing time based on input characteristics."""
        # Simulate realistic processing times
        base_time = 0.05  # 50ms base processing time
        
        # Add time based on tensor size
        tensor_factor = input_tensor.size / 1000000  # Scale by millions of elements
        
        # Add time based on sequence length
        sequence_factor = sequence_length / 1000  # Scale by thousands of tokens
        
        # Random variance to simulate real-world conditions
        variance = np.random.normal(0, 0.01)  # ±10ms variance
        
        total_time = base_time + tensor_factor * 0.02 + sequence_factor * 0.03 + variance
        return max(0.01, total_time)  # Minimum 10ms
    
    async def _monitor_resources(self, resource_samples: List[Dict[str, float]], duration: int):
        """Monitor resource usage during the test."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Simulate resource monitoring (in practice, would use actual monitoring)
                sample = {
                    "timestamp": time.time(),
                    "cpu_utilization": np.random.uniform(30, 90),  # Simulated CPU usage
                    "memory_usage_gb": np.random.uniform(2, 8),     # Simulated memory usage
                    "gpu_utilization": np.random.uniform(50, 95)   # Simulated GPU usage
                }
                resource_samples.append(sample)
                
            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")
            
            await asyncio.sleep(1.0)  # Sample every second
    
    def _calculate_throughput_metrics(self,
                                    test_requests: List[RequestResult],
                                    resource_samples: List[Dict[str, float]],
                                    test_duration: float,
                                    max_concurrent_requests: int,
                                    input_config: Dict[str, Any],
                                    test_name: str) -> ThroughputMetrics:
        """Calculate throughput metrics from test results."""
        
        successful_requests = [r for r in test_requests if r.success]
        failed_requests = [r for r in test_requests if not r.success]
        
        if not test_requests:
            logger.warning(f"No requests completed for {test_name}")
            return self._create_empty_throughput_metrics(
                test_name, max_concurrent_requests, test_duration, input_config
            )
        
        # Basic metrics
        total_requests = len(test_requests)
        successful_count = len(successful_requests)
        failed_count = len(failed_requests)
        
        # Throughput calculations
        requests_per_second = successful_count / test_duration if test_duration > 0 else 0
        
        total_tokens = sum(r.input_tokens + r.output_tokens for r in successful_requests)
        tokens_per_second = total_tokens / test_duration if test_duration > 0 else 0
        
        # Timing analysis
        request_durations = [(r.end_time - r.start_time) for r in successful_requests]
        avg_request_time = statistics.mean(request_durations) if request_durations else 0
        p95_request_time = np.percentile(request_durations, 95) if request_durations else 0
        p99_request_time = np.percentile(request_durations, 99) if request_durations else 0
        
        # Batch analysis
        batch_sizes = [r.batch_size for r in successful_requests]
        avg_batch_size = statistics.mean(batch_sizes) if batch_sizes else 1
        
        # Theoretical vs actual throughput (batch efficiency)
        theoretical_throughput = requests_per_second * avg_batch_size
        actual_throughput = tokens_per_second
        batch_efficiency = actual_throughput / theoretical_throughput if theoretical_throughput > 0 else 0
        
        # Concurrency analysis
        # Estimate average concurrent requests (simplified)
        overlapping_requests = self._calculate_average_concurrency(successful_requests)
        
        # Resource metrics
        cpu_utilizations = [s["cpu_utilization"] for s in resource_samples]
        memory_usages = [s["memory_usage_gb"] for s in resource_samples]
        
        avg_cpu_utilization = statistics.mean(cpu_utilizations) if cpu_utilizations else 0
        avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0
        peak_memory_usage = max(memory_usages) if memory_usages else 0
        
        return ThroughputMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            avg_concurrent_requests=overlapping_requests,
            total_requests=total_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            timeout_requests=0,  # Would be calculated from actual timeout detection
            test_duration_s=test_duration,
            avg_request_time_s=avg_request_time,
            p95_request_time_s=p95_request_time,
            p99_request_time_s=p99_request_time,
            avg_batch_size=avg_batch_size,
            batch_efficiency=batch_efficiency,
            avg_cpu_utilization=avg_cpu_utilization,
            avg_memory_usage_gb=avg_memory_usage,
            peak_memory_usage_gb=peak_memory_usage,
            max_concurrent_requests=max_concurrent_requests,
            test_duration=int(test_duration),
            input_config=input_config
        )
    
    def _calculate_average_concurrency(self, requests: List[RequestResult]) -> float:
        """Calculate average number of concurrent requests."""
        if not requests:
            return 0
        
        # Create timeline of request starts and ends
        events = []
        for req in requests:
            events.append((req.start_time, 1))   # Request start
            events.append((req.end_time, -1))    # Request end
        
        # Sort events by time
        events.sort(key=lambda x: x[0])
        
        # Calculate average concurrency
        total_time = 0
        weighted_concurrency = 0
        current_concurrency = 0
        last_time = events[0][0] if events else 0
        
        for event_time, delta in events:
            # Add weighted time for previous concurrency level
            time_diff = event_time - last_time
            weighted_concurrency += current_concurrency * time_diff
            total_time += time_diff
            
            # Update concurrency level
            current_concurrency += delta
            last_time = event_time
        
        return weighted_concurrency / total_time if total_time > 0 else 0
    
    def _create_empty_throughput_metrics(self, test_name: str, max_concurrent: int, 
                                       test_duration: float, input_config: Dict[str, Any]) -> ThroughputMetrics:
        """Create empty metrics for failed tests."""
        return ThroughputMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            requests_per_second=0.0,
            tokens_per_second=0.0,
            avg_concurrent_requests=0.0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            timeout_requests=0,
            test_duration_s=test_duration,
            avg_request_time_s=0.0,
            p95_request_time_s=0.0,
            p99_request_time_s=0.0,
            avg_batch_size=0.0,
            batch_efficiency=0.0,
            avg_cpu_utilization=0.0,
            avg_memory_usage_gb=0.0,
            peak_memory_usage_gb=0.0,
            max_concurrent_requests=max_concurrent,
            test_duration=int(test_duration),
            input_config=input_config
        )
    
    def _analyze_throughput_patterns(self) -> Dict[str, Any]:
        """Analyze throughput patterns across test scenarios."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "scalability": {},
            "efficiency": {},
            "resource_utilization": {},
            "bottleneck_analysis": {},
            "recommendations": []
        }
        
        # Scalability analysis
        throughput_by_concurrency = {}
        for result in self.results:
            concurrency = result.max_concurrent_requests
            throughput = result.requests_per_second
            throughput_by_concurrency[concurrency] = throughput
        
        if len(throughput_by_concurrency) > 1:
            concurrency_levels = sorted(throughput_by_concurrency.keys())
            throughput_values = [throughput_by_concurrency[c] for c in concurrency_levels]
            
            # Calculate scaling efficiency
            baseline_throughput = throughput_values[0]
            scaling_factors = []
            
            for i, (concurrency, throughput) in enumerate(zip(concurrency_levels, throughput_values)):
                expected_throughput = baseline_throughput * concurrency / concurrency_levels[0]
                efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
                scaling_factors.append(efficiency)
            
            analysis["scalability"] = {
                "concurrency_levels": concurrency_levels,
                "throughput_values": throughput_values,
                "scaling_efficiency": scaling_factors,
                "peak_throughput": max(throughput_values),
                "optimal_concurrency": concurrency_levels[throughput_values.index(max(throughput_values))]
            }
        
        # Efficiency analysis
        batch_results = [r for r in self.results if "batch" in r.test_name]
        single_results = [r for r in self.results if "batch" not in r.test_name and r.avg_batch_size <= 1]
        
        if batch_results and single_results:
            avg_batch_efficiency = statistics.mean(r.batch_efficiency for r in batch_results)
            avg_single_throughput = statistics.mean(r.tokens_per_second for r in single_results)
            avg_batch_throughput = statistics.mean(r.tokens_per_second for r in batch_results)
            
            analysis["efficiency"] = {
                "batch_efficiency": avg_batch_efficiency,
                "single_item_throughput": avg_single_throughput,
                "batch_throughput": avg_batch_throughput,
                "batch_improvement_factor": avg_batch_throughput / avg_single_throughput if avg_single_throughput > 0 else 1
            }
        
        # Resource utilization analysis
        cpu_utilizations = [r.avg_cpu_utilization for r in self.results]
        memory_utilizations = [r.avg_memory_usage_gb for r in self.results]
        throughputs = [r.requests_per_second for r in self.results]
        
        analysis["resource_utilization"] = {
            "avg_cpu_utilization": statistics.mean(cpu_utilizations),
            "avg_memory_usage_gb": statistics.mean(memory_utilizations),
            "peak_memory_usage_gb": max(r.peak_memory_usage_gb for r in self.results),
            "throughput_per_cpu_percent": statistics.mean(t / cpu for t, cpu in zip(throughputs, cpu_utilizations) if cpu > 0),
            "memory_efficiency": statistics.mean(t / mem for t, mem in zip(throughputs, memory_utilizations) if mem > 0)
        }
        
        # Bottleneck analysis
        bottlenecks = []
        avg_cpu = statistics.mean(cpu_utilizations)
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        if avg_cpu > 80:
            bottlenecks.append("High CPU utilization detected")
        if max_throughput / min_throughput > 3:
            bottlenecks.append("Significant throughput variation across concurrency levels")
        
        # Check for memory bottlenecks
        memory_growth = []
        sorted_results = sorted(self.results, key=lambda x: x.max_concurrent_requests)
        for i in range(1, len(sorted_results)):
            prev_mem = sorted_results[i-1].avg_memory_usage_gb
            curr_mem = sorted_results[i].avg_memory_usage_gb
            if prev_mem > 0:
                growth = (curr_mem - prev_mem) / prev_mem
                memory_growth.append(growth)
        
        if memory_growth and statistics.mean(memory_growth) > 0.5:
            bottlenecks.append("Excessive memory growth with increased concurrency")
        
        analysis["bottleneck_analysis"] = {
            "identified_bottlenecks": bottlenecks,
            "cpu_bound": avg_cpu > 70,
            "memory_bound": statistics.mean(memory_utilizations) > 6,  # Assume 8GB limit
            "throughput_variance": statistics.stdev(throughputs) / statistics.mean(throughputs) if throughputs else 0
        }
        
        # Recommendations
        recommendations = []
        if avg_cpu > 80:
            recommendations.append("CPU utilization is high - consider optimizing processing or adding more workers")
        if "batch" in [r.test_name for r in self.results] and avg_batch_efficiency < 0.7:
            recommendations.append("Batch processing efficiency is low - investigate serialization overhead")
        if analysis["scalability"].get("scaling_efficiency", [1])[-1] < 0.5:
            recommendations.append("Poor scaling at high concurrency - investigate bottlenecks")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        all_rps = [r.requests_per_second for r in self.results]
        all_tps = [r.tokens_per_second for r in self.results]
        all_success_rates = [r.successful_requests / r.total_requests if r.total_requests > 0 else 0 for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "peak_requests_per_second": max(all_rps),
            "peak_tokens_per_second": max(all_tps),
            "avg_requests_per_second": statistics.mean(all_rps),
            "avg_tokens_per_second": statistics.mean(all_tps),
            "overall_success_rate": statistics.mean(all_success_rates),
            "total_requests_processed": sum(r.total_requests for r in self.results),
            "total_successful_requests": sum(r.successful_requests for r in self.results),
            "total_failed_requests": sum(r.failed_requests for r in self.results)
        }
    
    def _generate_throughput_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive throughput report."""
        report_lines = [
            "# Throughput Benchmark Report",
            f"Generated: {results['benchmark_info']['start_time']}",
            f"Device: {results['benchmark_info']['local_device']}",
            f"Model: {results['benchmark_info']['cluster_config']['model_name']}",
            ""
        ]
        
        # Executive Summary
        if "summary" in results:
            summary = results["summary"]
            report_lines.extend([
                "## Executive Summary",
                f"- **Total Tests**: {summary.get('total_tests', 0)}",
                f"- **Peak Requests/sec**: {summary.get('peak_requests_per_second', 0):.1f}",
                f"- **Peak Tokens/sec**: {summary.get('peak_tokens_per_second', 0):.1f}",
                f"- **Average Requests/sec**: {summary.get('avg_requests_per_second', 0):.1f}",
                f"- **Average Tokens/sec**: {summary.get('avg_tokens_per_second', 0):.1f}",
                f"- **Overall Success Rate**: {summary.get('overall_success_rate', 0):.1%}",
                f"- **Total Requests Processed**: {summary.get('total_requests_processed', 0):,}",
                ""
            ])
        
        # Analysis Results
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "## Performance Analysis",
                ""
            ])
            
            # Scalability
            if "scalability" in analysis:
                scalability = analysis["scalability"]
                report_lines.extend([
                    "### Scalability Analysis",
                    f"- **Peak Throughput**: {scalability.get('peak_throughput', 0):.1f} requests/sec",
                    f"- **Optimal Concurrency**: {scalability.get('optimal_concurrency', 1)} concurrent requests",
                    ""
                ])
            
            # Efficiency
            if "efficiency" in analysis:
                efficiency = analysis["efficiency"]
                report_lines.extend([
                    "### Batch Processing Efficiency",
                    f"- **Batch Efficiency**: {efficiency.get('batch_efficiency', 0):.1%}",
                    f"- **Batch Improvement Factor**: {efficiency.get('batch_improvement_factor', 1):.1f}x",
                    ""
                ])
            
            # Resource Utilization
            if "resource_utilization" in analysis:
                resources = analysis["resource_utilization"]
                report_lines.extend([
                    "### Resource Utilization",
                    f"- **Average CPU**: {resources.get('avg_cpu_utilization', 0):.1f}%",
                    f"- **Average Memory**: {resources.get('avg_memory_usage_gb', 0):.1f} GB",
                    f"- **Peak Memory**: {resources.get('peak_memory_usage_gb', 0):.1f} GB",
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
        if "load_tests" in results:
            report_lines.extend([
                "## Detailed Test Results",
                ""
            ])
            
            for test_name, test_data in results["load_tests"].items():
                metrics = test_data["metrics"]
                scenario_info = test_data["scenario"]
                
                report_lines.extend([
                    f"### {test_name.replace('_', ' ').title()}",
                    f"**Description**: {scenario_info['description']}",
                    f"**Max Concurrent**: {scenario_info['max_concurrent']} requests",
                    "",
                    f"- **Requests/sec**: {metrics['requests_per_second']:.1f}",
                    f"- **Tokens/sec**: {metrics['tokens_per_second']:.1f}",
                    f"- **Success Rate**: {metrics['successful_requests'] / metrics['total_requests']:.1%}" if metrics['total_requests'] > 0 else "- **Success Rate**: 0%",
                    f"- **Average Request Time**: {metrics['avg_request_time_s']:.3f}s",
                    f"- **95th Percentile**: {metrics['p95_request_time_s']:.3f}s",
                    f"- **CPU Utilization**: {metrics['avg_cpu_utilization']:.1f}%",
                    f"- **Memory Usage**: {metrics['avg_memory_usage_gb']:.1f} GB",
                    ""
                ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Throughput report saved to: {output_file}")


async def main():
    """Example usage of throughput benchmark."""
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
    
    # Run benchmark
    benchmark = ThroughputBenchmark(config)
    results = await benchmark.run_comprehensive_throughput_analysis(test_duration=30)
    
    print("Throughput benchmark completed!")
    print(f"Peak throughput: {results['summary']['peak_requests_per_second']:.1f} req/s")
    print(f"Peak tokens/sec: {results['summary']['peak_tokens_per_second']:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
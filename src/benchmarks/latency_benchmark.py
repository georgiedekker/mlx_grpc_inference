"""
Latency benchmarking for MLX distributed inference system.

This module provides comprehensive latency analysis including:
- End-to-end inference latency
- Component-level timing (serialization, network, processing)
- Distribution analysis across different input sizes
- Percentile and statistical analysis
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

import mlx.core as mx
import numpy as np

from ..communication.grpc_client import GRPCInferenceClient
from ..communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
from ..core.config import ClusterConfig, DeviceConfig

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Metrics for latency analysis."""
    test_name: str
    timestamp: str
    
    # End-to-end metrics (milliseconds)
    avg_total_latency_ms: float
    min_total_latency_ms: float
    max_total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Component breakdown (milliseconds)
    avg_serialization_ms: float
    avg_network_ms: float
    avg_processing_ms: float
    avg_deserialization_ms: float
    
    # Statistical metrics
    latency_std_dev_ms: float
    coefficient_of_variation: float
    
    # Configuration
    input_shape: List[int]
    batch_size: int
    sequence_length: int
    iterations: int
    
    # Quality metrics
    success_rate: float
    error_count: int
    timeout_count: int


@dataclass
class ComponentTimings:
    """Detailed component timing breakdown."""
    serialization_ms: float
    network_ms: float
    processing_ms: float
    deserialization_ms: float
    total_ms: float


class LatencyBenchmark:
    """Comprehensive latency benchmarking for distributed inference."""
    
    def __init__(self, config: ClusterConfig, output_dir: str = "benchmarks/latency"):
        """
        Initialize latency benchmark.
        
        Args:
            config: Cluster configuration
            output_dir: Directory for benchmark results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_device_id = config.get_local_device_id()
        self.results: List[LatencyMetrics] = []
        
        logger.info(f"LatencyBenchmark initialized for device {self.local_device_id}")
    
    async def run_comprehensive_latency_analysis(self, 
                                               iterations: int = 100,
                                               warmup_iterations: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive latency analysis across different scenarios.
        
        Args:
            iterations: Number of iterations per test
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting comprehensive latency analysis ({iterations} iterations)")
        
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
                    "iterations": iterations,
                    "warmup_iterations": warmup_iterations
                }
            },
            "test_scenarios": {},
            "analysis": {},
            "summary": {},
            "errors": []
        }
        
        # Define test scenarios
        test_scenarios = [
            {
                "name": "small_input",
                "description": "Small input tensor (typical token)",
                "input_shape": [1, 128],
                "sequence_length": 128,
                "batch_size": 1
            },
            {
                "name": "medium_input", 
                "description": "Medium input tensor",
                "input_shape": [1, 512],
                "sequence_length": 512,
                "batch_size": 1
            },
            {
                "name": "large_input",
                "description": "Large input tensor",
                "input_shape": [1, 2048],
                "sequence_length": 2048,
                "batch_size": 1
            },
            {
                "name": "batch_2",
                "description": "Batch processing (2 sequences)",
                "input_shape": [2, 512],
                "sequence_length": 512,
                "batch_size": 2
            },
            {
                "name": "batch_4",
                "description": "Batch processing (4 sequences)",
                "input_shape": [4, 512],
                "sequence_length": 512,
                "batch_size": 4
            }
        ]
        
        # Run each test scenario
        for scenario in test_scenarios:
            try:
                logger.info(f"Running latency test: {scenario['name']}")
                
                metrics = await self.benchmark_inference_latency(
                    input_shape=scenario["input_shape"],
                    sequence_length=scenario["sequence_length"],
                    batch_size=scenario["batch_size"],
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                    test_name=scenario["name"]
                )
                
                self.results.append(metrics)
                benchmark_results["test_scenarios"][scenario["name"]] = {
                    "scenario": scenario,
                    "metrics": asdict(metrics)
                }
                
            except Exception as e:
                error_msg = f"Latency test '{scenario['name']}' failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
        
        # Analyze results
        try:
            analysis = self._analyze_latency_patterns()
            benchmark_results["analysis"] = analysis
        except Exception as e:
            error_msg = f"Latency analysis failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"latency_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Latency benchmark results saved to: {results_file}")
        
        # Generate report
        report_file = self.output_dir / f"latency_report_{timestamp}.md"
        self._generate_latency_report(benchmark_results, report_file)
        
        return benchmark_results
    
    async def benchmark_inference_latency(self,
                                        input_shape: List[int],
                                        sequence_length: int,
                                        batch_size: int = 1,
                                        iterations: int = 100,
                                        warmup_iterations: int = 10,
                                        test_name: str = "inference_latency") -> LatencyMetrics:
        """
        Benchmark inference latency for specific input configuration.
        
        Args:
            input_shape: Shape of input tensor
            sequence_length: Sequence length
            batch_size: Batch size
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            test_name: Name for this test
            
        Returns:
            Latency metrics
        """
        logger.debug(f"Benchmarking {test_name}: shape={input_shape}, seq_len={sequence_length}")
        
        # Generate test input
        test_input = mx.random.randint(0, 1000, input_shape).astype(mx.int32)
        
        # Component timings for each iteration
        component_timings: List[ComponentTimings] = []
        errors = 0
        timeouts = 0
        
        # Warmup iterations
        logger.debug(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            try:
                await self._run_single_inference(test_input, sequence_length)
            except Exception as e:
                logger.debug(f"Warmup iteration {i} failed: {e}")
        
        # Actual benchmark iterations
        logger.debug(f"Running {iterations} benchmark iterations...")
        start_time = time.time()
        
        for i in range(iterations):
            try:
                timing = await self._run_single_inference_with_timing(test_input, sequence_length)
                component_timings.append(timing)
                
            except asyncio.TimeoutError:
                timeouts += 1
                logger.debug(f"Iteration {i} timed out")
            except Exception as e:
                errors += 1
                logger.debug(f"Iteration {i} failed: {e}")
        
        benchmark_duration = time.time() - start_time
        
        # Calculate metrics
        if not component_timings:
            logger.warning(f"No successful iterations for {test_name}")
            return self._create_empty_metrics(test_name, input_shape, batch_size, sequence_length, iterations)
        
        total_latencies = [t.total_ms for t in component_timings]
        serialization_times = [t.serialization_ms for t in component_timings]
        network_times = [t.network_ms for t in component_timings]
        processing_times = [t.processing_ms for t in component_timings]
        deserialization_times = [t.deserialization_ms for t in component_timings]
        
        # Statistical analysis
        avg_latency = statistics.mean(total_latencies)
        std_dev = statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0
        cv = std_dev / avg_latency if avg_latency > 0 else 0
        
        metrics = LatencyMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            avg_total_latency_ms=avg_latency,
            min_total_latency_ms=min(total_latencies),
            max_total_latency_ms=max(total_latencies),
            p50_latency_ms=np.percentile(total_latencies, 50),
            p95_latency_ms=np.percentile(total_latencies, 95),
            p99_latency_ms=np.percentile(total_latencies, 99),
            avg_serialization_ms=statistics.mean(serialization_times),
            avg_network_ms=statistics.mean(network_times),
            avg_processing_ms=statistics.mean(processing_times),
            avg_deserialization_ms=statistics.mean(deserialization_times),
            latency_std_dev_ms=std_dev,
            coefficient_of_variation=cv,
            input_shape=input_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            iterations=iterations,
            success_rate=len(component_timings) / iterations,
            error_count=errors,
            timeout_count=timeouts
        )
        
        logger.debug(f"{test_name} completed: {avg_latency:.2f}ms avg latency")
        return metrics
    
    async def _run_single_inference(self, input_tensor: mx.array, sequence_length: int) -> mx.array:
        """Run a single inference without detailed timing."""
        # This is a simplified version - in practice would connect to actual inference pipeline
        # For now, simulate the inference process
        
        # Simulate processing time based on tensor size and sequence length
        processing_time = (input_tensor.size * sequence_length) / 1000000  # Simulated processing
        await asyncio.sleep(processing_time)
        
        # Return simulated output
        output_shape = list(input_tensor.shape)
        output_shape[-1] = 768  # Typical hidden size
        return mx.random.normal(output_shape).astype(mx.float32)
    
    async def _run_single_inference_with_timing(self, 
                                              input_tensor: mx.array, 
                                              sequence_length: int) -> ComponentTimings:
        """Run a single inference with detailed component timing."""
        
        total_start = time.time()
        
        # 1. Serialization timing
        serialization_start = time.time()
        serialized_data, metadata = serialize_mlx_array(input_tensor, compress=False)
        serialization_time = (time.time() - serialization_start) * 1000
        
        # 2. Network timing (simulated)
        network_start = time.time()
        # Simulate network latency based on data size
        data_size_mb = len(serialized_data) / (1024 * 1024)
        network_latency = data_size_mb * 10 + 5  # Base 5ms + 10ms per MB
        await asyncio.sleep(network_latency / 1000)
        network_time = (time.time() - network_start) * 1000
        
        # 3. Processing timing
        processing_start = time.time()
        # Simulate actual inference processing
        output_tensor = await self._run_single_inference(input_tensor, sequence_length)
        processing_time = (time.time() - processing_start) * 1000
        
        # 4. Deserialization timing
        deserialization_start = time.time()
        # Simulate deserializing the output
        output_serialized, output_metadata = serialize_mlx_array(output_tensor, compress=False)
        reconstructed = deserialize_mlx_array(output_serialized, output_metadata)
        deserialization_time = (time.time() - deserialization_start) * 1000
        
        total_time = (time.time() - total_start) * 1000
        
        return ComponentTimings(
            serialization_ms=serialization_time,
            network_ms=network_time,
            processing_ms=processing_time,
            deserialization_ms=deserialization_time,
            total_ms=total_time
        )
    
    def _create_empty_metrics(self, test_name: str, input_shape: List[int], 
                            batch_size: int, sequence_length: int, iterations: int) -> LatencyMetrics:
        """Create empty metrics for failed tests."""
        return LatencyMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            avg_total_latency_ms=0.0,
            min_total_latency_ms=0.0,
            max_total_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            avg_serialization_ms=0.0,
            avg_network_ms=0.0,
            avg_processing_ms=0.0,
            avg_deserialization_ms=0.0,
            latency_std_dev_ms=0.0,
            coefficient_of_variation=0.0,
            input_shape=input_shape,
            batch_size=batch_size,
            sequence_length=sequence_length,
            iterations=iterations,
            success_rate=0.0,
            error_count=iterations,
            timeout_count=0
        )
    
    def _analyze_latency_patterns(self) -> Dict[str, Any]:
        """Analyze latency patterns across test scenarios."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "scalability": {},
            "component_breakdown": {},
            "performance_characteristics": {},
            "bottleneck_analysis": {},
            "recommendations": []
        }
        
        # Scalability analysis
        batch_results = [r for r in self.results if r.batch_size > 1]
        single_results = [r for r in self.results if r.batch_size == 1]
        
        if batch_results and single_results:
            single_avg = statistics.mean(r.avg_total_latency_ms for r in single_results)
            batch_avg = statistics.mean(r.avg_total_latency_ms for r in batch_results)
            
            analysis["scalability"] = {
                "single_item_avg_latency_ms": single_avg,
                "batch_avg_latency_ms": batch_avg,
                "batch_efficiency": single_avg / (batch_avg / statistics.mean(r.batch_size for r in batch_results)),
                "batch_overhead_factor": batch_avg / single_avg
            }
        
        # Component breakdown analysis
        avg_serialization = statistics.mean(r.avg_serialization_ms for r in self.results)
        avg_network = statistics.mean(r.avg_network_ms for r in self.results)
        avg_processing = statistics.mean(r.avg_processing_ms for r in self.results)
        avg_deserialization = statistics.mean(r.avg_deserialization_ms for r in self.results)
        
        total_avg = avg_serialization + avg_network + avg_processing + avg_deserialization
        
        analysis["component_breakdown"] = {
            "serialization_ms": avg_serialization,
            "network_ms": avg_network,
            "processing_ms": avg_processing,
            "deserialization_ms": avg_deserialization,
            "serialization_percentage": (avg_serialization / total_avg * 100) if total_avg > 0 else 0,
            "network_percentage": (avg_network / total_avg * 100) if total_avg > 0 else 0,
            "processing_percentage": (avg_processing / total_avg * 100) if total_avg > 0 else 0,
            "deserialization_percentage": (avg_deserialization / total_avg * 100) if total_avg > 0 else 0
        }
        
        # Performance characteristics
        all_latencies = [r.avg_total_latency_ms for r in self.results]
        all_cvs = [r.coefficient_of_variation for r in self.results]
        
        analysis["performance_characteristics"] = {
            "overall_avg_latency_ms": statistics.mean(all_latencies),
            "latency_range_ms": max(all_latencies) - min(all_latencies),
            "avg_coefficient_of_variation": statistics.mean(all_cvs),
            "consistency_score": 1.0 - statistics.mean(all_cvs),  # Higher is more consistent
            "success_rate": statistics.mean(r.success_rate for r in self.results)
        }
        
        # Bottleneck analysis
        bottlenecks = []
        if avg_serialization > avg_processing * 0.5:
            bottlenecks.append("Serialization overhead is significant")
        if avg_network > avg_processing:
            bottlenecks.append("Network latency is a major bottleneck")
        if avg_deserialization > avg_serialization:
            bottlenecks.append("Deserialization is slower than serialization")
        
        analysis["bottleneck_analysis"] = {
            "identified_bottlenecks": bottlenecks,
            "primary_bottleneck": max([
                ("serialization", avg_serialization),
                ("network", avg_network),
                ("processing", avg_processing),
                ("deserialization", avg_deserialization)
            ], key=lambda x: x[1])[0]
        }
        
        # Recommendations
        recommendations = []
        if avg_serialization > total_avg * 0.2:
            recommendations.append("Consider implementing tensor compression to reduce serialization overhead")
        if avg_network > total_avg * 0.3:
            recommendations.append("Network latency is high - consider connection pooling or faster network")
        if statistics.mean(all_cvs) > 0.3:
            recommendations.append("High latency variance detected - investigate system stability")
        if statistics.mean(r.success_rate for r in self.results) < 0.95:
            recommendations.append("Low success rate - investigate error handling and timeouts")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        all_latencies = [r.avg_total_latency_ms for r in self.results]
        all_p95s = [r.p95_latency_ms for r in self.results]
        all_success_rates = [r.success_rate for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "overall_avg_latency_ms": statistics.mean(all_latencies),
            "overall_p95_latency_ms": statistics.mean(all_p95s),
            "best_latency_ms": min(all_latencies),
            "worst_latency_ms": max(all_latencies),
            "overall_success_rate": statistics.mean(all_success_rates),
            "total_errors": sum(r.error_count for r in self.results),
            "total_timeouts": sum(r.timeout_count for r in self.results)
        }
    
    def _generate_latency_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive latency report."""
        report_lines = [
            "# Latency Benchmark Report",
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
                f"- **Average Latency**: {summary.get('overall_avg_latency_ms', 0):.2f} ms",
                f"- **95th Percentile**: {summary.get('overall_p95_latency_ms', 0):.2f} ms",
                f"- **Best Performance**: {summary.get('best_latency_ms', 0):.2f} ms",
                f"- **Worst Performance**: {summary.get('worst_latency_ms', 0):.2f} ms",
                f"- **Success Rate**: {summary.get('overall_success_rate', 0):.1%}",
                ""
            ])
        
        # Analysis Results
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "## Performance Analysis",
                ""
            ])
            
            # Component breakdown
            if "component_breakdown" in analysis:
                breakdown = analysis["component_breakdown"]
                report_lines.extend([
                    "### Component Latency Breakdown",
                    f"- **Serialization**: {breakdown.get('serialization_ms', 0):.2f} ms ({breakdown.get('serialization_percentage', 0):.1f}%)",
                    f"- **Network**: {breakdown.get('network_ms', 0):.2f} ms ({breakdown.get('network_percentage', 0):.1f}%)",
                    f"- **Processing**: {breakdown.get('processing_ms', 0):.2f} ms ({breakdown.get('processing_percentage', 0):.1f}%)",
                    f"- **Deserialization**: {breakdown.get('deserialization_ms', 0):.2f} ms ({breakdown.get('deserialization_percentage', 0):.1f}%)",
                    ""
                ])
            
            # Bottlenecks
            if "bottleneck_analysis" in analysis:
                bottlenecks = analysis["bottleneck_analysis"]
                report_lines.extend([
                    "### Bottleneck Analysis",
                    f"- **Primary Bottleneck**: {bottlenecks.get('primary_bottleneck', 'Unknown')}",
                    "- **Identified Issues**:",
                    ""
                ])
                for issue in bottlenecks.get("identified_bottlenecks", []):
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
        if "test_scenarios" in results:
            report_lines.extend([
                "## Detailed Test Results",
                ""
            ])
            
            for scenario_name, scenario_data in results["test_scenarios"].items():
                metrics = scenario_data["metrics"]
                scenario_info = scenario_data["scenario"]
                
                report_lines.extend([
                    f"### {scenario_name.replace('_', ' ').title()}",
                    f"**Description**: {scenario_info['description']}",
                    f"**Configuration**: {scenario_info['input_shape']} shape, {scenario_info['sequence_length']} sequence length",
                    "",
                    f"- **Average Latency**: {metrics['avg_total_latency_ms']:.2f} ms",
                    f"- **95th Percentile**: {metrics['p95_latency_ms']:.2f} ms",
                    f"- **Min/Max**: {metrics['min_total_latency_ms']:.2f} / {metrics['max_total_latency_ms']:.2f} ms",
                    f"- **Standard Deviation**: {metrics['latency_std_dev_ms']:.2f} ms",
                    f"- **Coefficient of Variation**: {metrics['coefficient_of_variation']:.3f}",
                    f"- **Success Rate**: {metrics['success_rate']:.1%}",
                    ""
                ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Latency report saved to: {output_file}")


async def main():
    """Example usage of latency benchmark."""
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration (this would be actual config in real usage)
    # For demo purposes, we'll create a minimal config
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
    benchmark = LatencyBenchmark(config)
    results = await benchmark.run_comprehensive_latency_analysis(iterations=50)
    
    print("Latency benchmark completed!")
    print(f"Average latency: {results['summary']['overall_avg_latency_ms']:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
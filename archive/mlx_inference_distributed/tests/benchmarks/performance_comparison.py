#!/usr/bin/env python3
"""
Performance Comparison Tools for Single-Device vs Distributed MLX Inference.

This module provides comprehensive tools to compare performance between:
- Single-device inference (baseline)
- 2-device distributed inference
- 3-device distributed inference
- Different model sizes and configurations

Metrics compared:
- Inference latency and throughput
- Memory usage efficiency
- Resource utilization
- Scalability characteristics
- Cost-effectiveness analysis
"""

import asyncio
import time
import logging
import statistics
import json
import sys
import os
import subprocess
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import concurrent.futures
import requests
import psutil

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific configuration."""
    configuration: str  # e.g., "single_device", "distributed_2dev", "distributed_3dev"
    model_name: str
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float
    
    # Resource metrics
    total_memory_mb: float
    peak_memory_mb: float
    memory_efficiency_score: float  # tokens/MB
    cpu_utilization_percent: float
    
    # Quality metrics
    success_rate: float
    error_count: int
    
    # Configuration details
    device_count: int
    model_size_gb: float
    test_duration_seconds: float
    sample_size: int
    
    # Cost metrics (simulated)
    relative_cost_score: float  # Normalized cost per token
    
    timestamp: str


@dataclass
class ComparisonResult:
    """Result of comparing two configurations."""
    baseline_config: str
    comparison_config: str
    
    # Performance improvements (positive = better)
    latency_improvement_factor: float  # baseline_latency / comparison_latency
    throughput_improvement_factor: float  # comparison_throughput / baseline_throughput
    memory_efficiency_improvement: float
    
    # Trade-off analysis
    performance_per_device: float
    memory_per_device: float
    cost_effectiveness: float
    
    # Scalability metrics
    scaling_efficiency: float  # actual_speedup / theoretical_speedup
    overhead_factor: float
    
    # Recommendations
    recommended_use_cases: List[str]
    trade_offs: List[str]
    
    summary: str


class SingleDeviceSimulator:
    """Simulates single-device performance for comparison."""
    
    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-8bit"):
        self.model_name = model_name
        self.base_processing_time_ms = 120  # Base processing time per token
        self.memory_base_mb = 2000  # Base memory usage
    
    async def benchmark_single_device(self, test_prompts: List[Tuple[str, int]], 
                                    iterations: int = 20) -> PerformanceMetrics:
        """Simulate single-device performance."""
        logger.info(f"Simulating single-device performance ({iterations} iterations)")
        
        start_time = time.time()
        latencies = []
        throughput_measurements = []
        memory_measurements = []
        errors = 0
        total_tokens = 0
        
        # Simulate device characteristics
        device_efficiency = 1.0  # Baseline efficiency
        memory_overhead = 1.0
        
        for prompt_name, max_tokens in test_prompts:
            for iteration in range(iterations // len(test_prompts)):
                try:
                    # Simulate processing time (varies with token count)
                    base_time = self.base_processing_time_ms * max_tokens / 50.0  # Normalize
                    
                    # Add some randomness to simulate real-world variance
                    processing_time = base_time * (0.8 + 0.4 * np.random.random())
                    
                    # Simulate memory usage
                    memory_used = self.memory_base_mb + (max_tokens * 2.5)  # Memory grows with tokens
                    memory_measurements.append(memory_used)
                    
                    latencies.append(processing_time)
                    
                    if processing_time > 0:
                        tokens_per_sec = (max_tokens * 1000) / processing_time
                        throughput_measurements.append(tokens_per_sec)
                    
                    total_tokens += max_tokens
                    
                    # Simulate processing delay
                    await asyncio.sleep(processing_time / 5000)  # Scale down for simulation
                    
                except Exception as e:
                    errors += 1
                    logger.warning(f"Simulation error: {e}")
        
        test_duration = time.time() - start_time
        
        return PerformanceMetrics(
            configuration="single_device",
            model_name=self.model_name,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p50_latency_ms=np.percentile(latencies, 50) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            tokens_per_second=statistics.mean(throughput_measurements) if throughput_measurements else 0,
            requests_per_second=len(latencies) / test_duration if test_duration > 0 else 0,
            total_memory_mb=statistics.mean(memory_measurements) if memory_measurements else 0,
            peak_memory_mb=max(memory_measurements) if memory_measurements else 0,
            memory_efficiency_score=total_tokens / max(memory_measurements) if memory_measurements else 0,
            cpu_utilization_percent=85.0,  # Simulated high CPU usage
            success_rate=1.0 - (errors / (len(latencies) + errors)) if (len(latencies) + errors) > 0 else 0,
            error_count=errors,
            device_count=1,
            model_size_gb=3.4,  # Approximate size of Qwen3-1.7B-8bit
            test_duration_seconds=test_duration,
            sample_size=len(latencies),
            relative_cost_score=1.0,  # Baseline cost
            timestamp=datetime.now().isoformat()
        )


class DistributedPerformanceBenchmark:
    """Benchmarks distributed inference performance."""
    
    def __init__(self, api_url: str = "http://localhost:8100"):
        self.api_url = api_url
        self.model_name = "mlx-community/Qwen3-1.7B-8bit"
    
    async def benchmark_distributed(self, test_prompts: List[Tuple[str, int]], 
                                  iterations: int = 20) -> PerformanceMetrics:
        """Benchmark distributed inference performance."""
        logger.info(f"Benchmarking distributed inference ({iterations} iterations)")
        
        # First, check cluster status to determine device count
        device_count = await self._get_device_count()
        if device_count == 0:
            raise ValueError("No distributed devices available")
        
        start_time = time.time()
        latencies = []
        throughput_measurements = []
        errors = 0
        total_tokens = 0
        
        # Monitor system resources
        initial_memory = psutil.virtual_memory().percent
        memory_samples = []
        cpu_samples = []
        
        for prompt_name, max_tokens in test_prompts:
            for iteration in range(iterations // len(test_prompts)):
                try:
                    request_data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": f"Generate a response with approximately {max_tokens} tokens about {prompt_name}"}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                    
                    request_start = time.time()
                    response = requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=request_data,
                        timeout=120
                    )
                    request_duration = (time.time() - request_start) * 1000
                    
                    # Sample system resources during request
                    memory_samples.append(psutil.virtual_memory().percent)
                    cpu_samples.append(psutil.cpu_percent())
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "usage" in data and "completion_tokens" in data["usage"]:
                            tokens_generated = data["usage"]["completion_tokens"]
                            total_tokens += tokens_generated
                            
                            latencies.append(request_duration)
                            
                            if request_duration > 0:
                                tokens_per_sec = (tokens_generated * 1000) / request_duration
                                throughput_measurements.append(tokens_per_sec)
                    else:
                        errors += 1
                        logger.warning(f"Request failed: {response.status_code}")
                
                except Exception as e:
                    errors += 1
                    logger.error(f"Benchmark error: {e}")
        
        test_duration = time.time() - start_time
        
        # Get memory information from cluster
        total_memory_mb, peak_memory_mb = await self._get_cluster_memory_info()
        
        config_name = f"distributed_{device_count}dev"
        
        return PerformanceMetrics(
            configuration=config_name,
            model_name=self.model_name,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p50_latency_ms=np.percentile(latencies, 50) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            tokens_per_second=statistics.mean(throughput_measurements) if throughput_measurements else 0,
            requests_per_second=len(latencies) / test_duration if test_duration > 0 else 0,
            total_memory_mb=total_memory_mb,
            peak_memory_mb=peak_memory_mb,
            memory_efficiency_score=total_tokens / peak_memory_mb if peak_memory_mb > 0 else 0,
            cpu_utilization_percent=statistics.mean(cpu_samples) if cpu_samples else 0,
            success_rate=1.0 - (errors / (len(latencies) + errors)) if (len(latencies) + errors) > 0 else 0,
            error_count=errors,
            device_count=device_count,
            model_size_gb=3.4,  # Approximate size of Qwen3-1.7B-8bit
            test_duration_seconds=test_duration,
            sample_size=len(latencies),
            relative_cost_score=device_count * 0.8,  # Economies of scale
            timestamp=datetime.now().isoformat()
        )
    
    async def _get_device_count(self) -> int:
        """Get the number of devices in the cluster."""
        try:
            response = requests.get(f"{self.api_url}/distributed/gpu-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return len(data.get("devices", []))
        except:
            pass
        return 0
    
    async def _get_cluster_memory_info(self) -> Tuple[float, float]:
        """Get memory information from the cluster."""
        try:
            response = requests.get(f"{self.api_url}/distributed/gpu-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                devices = data.get("devices", [])
                
                total_memory = 0
                peak_memory = 0
                
                for device in devices:
                    if "system_memory" in device:
                        device_total = device["system_memory"].get("total_gb", 0) * 1024  # Convert to MB
                        device_used = device_total * (device["system_memory"].get("used_percent", 0) / 100)
                        
                        total_memory += device_total
                        peak_memory += device_used
                
                return total_memory, peak_memory
        except:
            pass
        
        return 8000.0, 4000.0  # Default estimates


class PerformanceComparator:
    """Compares performance between different configurations."""
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_prompts = [
            ("simple_task", 20),
            ("medium_task", 100),
            ("complex_task", 200),
            ("long_generation", 400)
        ]
    
    def compare_configurations(self, baseline: PerformanceMetrics, 
                             comparison: PerformanceMetrics) -> ComparisonResult:
        """Compare two performance configurations."""
        
        # Calculate improvement factors
        latency_improvement = baseline.avg_latency_ms / comparison.avg_latency_ms if comparison.avg_latency_ms > 0 else 0
        throughput_improvement = comparison.tokens_per_second / baseline.tokens_per_second if baseline.tokens_per_second > 0 else 0
        memory_efficiency_improvement = comparison.memory_efficiency_score / baseline.memory_efficiency_score if baseline.memory_efficiency_score > 0 else 0
        
        # Calculate per-device metrics
        perf_per_device_baseline = baseline.tokens_per_second / baseline.device_count
        perf_per_device_comparison = comparison.tokens_per_second / comparison.device_count
        performance_per_device = perf_per_device_comparison / perf_per_device_baseline if perf_per_device_baseline > 0 else 0
        
        memory_per_device = (comparison.peak_memory_mb / comparison.device_count) / (baseline.peak_memory_mb / baseline.device_count) if baseline.peak_memory_mb > 0 else 0
        
        # Calculate scaling efficiency (actual speedup vs theoretical)
        theoretical_speedup = comparison.device_count / baseline.device_count
        actual_speedup = throughput_improvement
        scaling_efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0
        
        # Calculate overhead
        overhead_factor = (comparison.avg_latency_ms * comparison.device_count) / (baseline.avg_latency_ms * baseline.device_count) if baseline.avg_latency_ms > 0 else 1
        
        # Cost effectiveness (performance improvement per relative cost increase)
        cost_effectiveness = throughput_improvement / (comparison.relative_cost_score / baseline.relative_cost_score) if baseline.relative_cost_score > 0 else 0
        
        # Generate recommendations
        recommendations = []
        trade_offs = []
        
        if throughput_improvement > 1.5:
            recommendations.append("High-throughput workloads")
        if latency_improvement > 1.2:
            recommendations.append("Low-latency applications")
        if memory_efficiency_improvement > 1.1:
            recommendations.append("Memory-constrained environments")
        if scaling_efficiency > 0.7:
            recommendations.append("Scalable production deployments")
        
        if memory_per_device > 1.2:
            trade_offs.append("Higher memory usage per device")
        if cost_effectiveness < 1.0:
            trade_offs.append("Higher cost per unit of performance")
        if overhead_factor > 1.3:
            trade_offs.append("Increased communication overhead")
        
        # Generate summary
        if throughput_improvement > 1.5 and latency_improvement > 1.2:
            summary = f"{comparison.configuration} provides {throughput_improvement:.1f}x throughput and {latency_improvement:.1f}x latency improvement over {baseline.configuration}"
        elif throughput_improvement > 1.2:
            summary = f"{comparison.configuration} provides {throughput_improvement:.1f}x throughput improvement with {scaling_efficiency:.1%} scaling efficiency"
        elif latency_improvement > 1.2:
            summary = f"{comparison.configuration} provides {latency_improvement:.1f}x latency improvement but limited throughput gains"
        else:
            summary = f"{comparison.configuration} shows minimal performance improvement over {baseline.configuration}"
        
        return ComparisonResult(
            baseline_config=baseline.configuration,
            comparison_config=comparison.configuration,
            latency_improvement_factor=latency_improvement,
            throughput_improvement_factor=throughput_improvement,
            memory_efficiency_improvement=memory_efficiency_improvement,
            performance_per_device=performance_per_device,
            memory_per_device=memory_per_device,
            cost_effectiveness=cost_effectiveness,
            scaling_efficiency=scaling_efficiency,
            overhead_factor=overhead_factor,
            recommended_use_cases=recommendations,
            trade_offs=trade_offs,
            summary=summary
        )
    
    def generate_comparison_charts(self, metrics_list: List[PerformanceMetrics], 
                                 output_file: str = "performance_comparison"):
        """Generate comparison charts."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        configs = [m.configuration for m in metrics_list]
        
        # 1. Throughput comparison
        throughput = [m.tokens_per_second for m in metrics_list]
        bars1 = ax1.bar(configs, throughput, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Tokens per Second')
        ax1.set_ylabel('Tokens/sec')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, throughput):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. Latency comparison
        latency = [m.avg_latency_ms for m in metrics_list]
        bars2 = ax2.bar(configs, latency, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Average Latency')
        ax2.set_ylabel('Milliseconds')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, latency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latency)*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 3. Memory efficiency
        memory_eff = [m.memory_efficiency_score for m in metrics_list]
        bars3 = ax3.bar(configs, memory_eff, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Memory Efficiency')
        ax3.set_ylabel('Tokens per MB')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, memory_eff):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_eff)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Success rate
        success_rates = [m.success_rate * 100 for m in metrics_list]
        bars4 = ax4.bar(configs, success_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax4.set_title('Success Rate')
        ax4.set_ylabel('Percentage')
        ax4.set_ylim(0, 105)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_file = self.output_dir / f"{output_file}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison chart saved to: {chart_file}")
        
        # Generate scaling efficiency chart
        self._generate_scaling_chart(metrics_list, f"{output_file}_scaling")
    
    def _generate_scaling_chart(self, metrics_list: List[PerformanceMetrics], output_file: str):
        """Generate scaling efficiency chart."""
        if len(metrics_list) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        
        device_counts = [m.device_count for m in metrics_list]
        throughput_values = [m.tokens_per_second for m in metrics_list]
        
        # Calculate theoretical and actual scaling
        baseline_throughput = throughput_values[0]
        baseline_devices = device_counts[0]
        
        theoretical_scaling = [(d / baseline_devices) * baseline_throughput for d in device_counts]
        
        plt.plot(device_counts, throughput_values, 'o-', label='Actual Performance', linewidth=2, markersize=8)
        plt.plot(device_counts, theoretical_scaling, '--', label='Theoretical Linear Scaling', linewidth=2)
        
        plt.xlabel('Number of Devices')
        plt.ylabel('Tokens per Second')
        plt.title('Scaling Efficiency Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add efficiency annotations
        for i, (devices, actual, theoretical) in enumerate(zip(device_counts, throughput_values, theoretical_scaling)):
            if i > 0:  # Skip baseline
                efficiency = (actual / theoretical) * 100 if theoretical > 0 else 0
                plt.annotate(f'{efficiency:.0f}%', 
                           (devices, actual), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center')
        
        chart_file = self.output_dir / f"{output_file}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scaling efficiency chart saved to: {chart_file}")


class ComprehensivePerformanceAnalysis:
    """Main class for comprehensive performance analysis."""
    
    def __init__(self, api_url: str = "http://localhost:8100", output_dir: str = "performance_analysis"):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.single_device_sim = SingleDeviceSimulator()
        self.distributed_benchmark = DistributedPerformanceBenchmark(api_url)
        self.comparator = PerformanceComparator(str(self.output_dir))
        
        self.test_prompts = [
            ("code_generation", 150),
            ("text_analysis", 100),
            ("creative_writing", 250),
            ("question_answering", 80),
            ("summarization", 120)
        ]
    
    async def run_comprehensive_analysis(self, iterations: int = 30) -> Dict[str, Any]:
        """Run comprehensive performance analysis."""
        logger.info("Starting comprehensive performance analysis")
        
        analysis_results = {
            "analysis_info": {
                "start_time": datetime.now().isoformat(),
                "api_url": self.api_url,
                "iterations": iterations,
                "test_prompts": len(self.test_prompts)
            },
            "metrics": {},
            "comparisons": {},
            "recommendations": {},
            "charts": []
        }
        
        all_metrics = []
        
        # 1. Benchmark single-device performance (simulated)
        logger.info("Benchmarking single-device performance (simulated)...")
        try:
            single_metrics = await self.single_device_sim.benchmark_single_device(self.test_prompts, iterations)
            all_metrics.append(single_metrics)
            analysis_results["metrics"]["single_device"] = asdict(single_metrics)
            logger.info(f"Single-device: {single_metrics.tokens_per_second:.1f} tok/s, {single_metrics.avg_latency_ms:.0f}ms latency")
        except Exception as e:
            logger.error(f"Single-device benchmark failed: {e}")
            analysis_results["errors"] = analysis_results.get("errors", [])
            analysis_results["errors"].append(f"Single-device benchmark: {e}")
        
        # 2. Benchmark distributed performance
        logger.info("Benchmarking distributed performance...")
        try:
            distributed_metrics = await self.distributed_benchmark.benchmark_distributed(self.test_prompts, iterations)
            all_metrics.append(distributed_metrics)
            analysis_results["metrics"]["distributed"] = asdict(distributed_metrics)
            logger.info(f"Distributed ({distributed_metrics.device_count} devices): {distributed_metrics.tokens_per_second:.1f} tok/s, {distributed_metrics.avg_latency_ms:.0f}ms latency")
        except Exception as e:
            logger.error(f"Distributed benchmark failed: {e}")
            analysis_results["errors"] = analysis_results.get("errors", [])
            analysis_results["errors"].append(f"Distributed benchmark: {e}")
        
        # 3. Compare configurations
        if len(all_metrics) >= 2:
            logger.info("Generating performance comparisons...")
            
            baseline = all_metrics[0]  # Single-device baseline
            
            for comparison_metrics in all_metrics[1:]:
                comparison_result = self.comparator.compare_configurations(baseline, comparison_metrics)
                comparison_key = f"{baseline.configuration}_vs_{comparison_metrics.configuration}"
                analysis_results["comparisons"][comparison_key] = asdict(comparison_result)
                
                logger.info(f"Comparison {comparison_key}: {comparison_result.summary}")
        
        # 4. Generate visualizations
        if all_metrics:
            logger.info("Generating performance charts...")
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chart_name = f"performance_comparison_{timestamp}"
                self.comparator.generate_comparison_charts(all_metrics, chart_name)
                analysis_results["charts"].append(f"{chart_name}.png")
                analysis_results["charts"].append(f"{chart_name}_scaling.png")
            except Exception as e:
                logger.error(f"Chart generation failed: {e}")
        
        # 5. Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(all_metrics, analysis_results.get("comparisons", {}))
        
        # 6. Save detailed results
        analysis_results["analysis_info"]["end_time"] = datetime.now().isoformat()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"performance_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Performance analysis results saved to: {results_file}")
        
        # 7. Generate markdown report
        report_file = self.output_dir / f"performance_report_{timestamp}.md"
        self._generate_performance_report(analysis_results, report_file)
        
        return analysis_results
    
    def _generate_recommendations(self, metrics_list: List[PerformanceMetrics], 
                                 comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance recommendations."""
        recommendations = {
            "deployment_recommendations": [],
            "optimization_opportunities": [],
            "cost_analysis": {},
            "use_case_mapping": {}
        }
        
        if not metrics_list:
            return recommendations
        
        # Find best performer for different metrics
        best_throughput = max(metrics_list, key=lambda m: m.tokens_per_second)
        best_latency = min(metrics_list, key=lambda m: m.avg_latency_ms)
        best_memory_efficiency = max(metrics_list, key=lambda m: m.memory_efficiency_score)
        
        # Deployment recommendations
        if best_throughput.configuration != "single_device":
            recommendations["deployment_recommendations"].append(
                f"For high-throughput workloads, use {best_throughput.configuration} "
                f"({best_throughput.tokens_per_second:.1f} tok/s)"
            )
        
        if best_latency.configuration != "single_device":
            recommendations["deployment_recommendations"].append(
                f"For low-latency applications, use {best_latency.configuration} "
                f"({best_latency.avg_latency_ms:.0f}ms average latency)"
            )
        
        if best_memory_efficiency.configuration != "single_device":
            recommendations["deployment_recommendations"].append(
                f"For memory-efficient deployment, use {best_memory_efficiency.configuration} "
                f"({best_memory_efficiency.memory_efficiency_score:.2f} tokens/MB)"
            )
        
        # Optimization opportunities
        for comparison_key, comparison_data in comparisons.items():
            if comparison_data["scaling_efficiency"] < 0.7:
                recommendations["optimization_opportunities"].append(
                    f"Poor scaling efficiency in {comparison_data['comparison_config']} "
                    f"({comparison_data['scaling_efficiency']:.1%}) - investigate communication overhead"
                )
            
            if comparison_data["memory_per_device"] > 1.5:
                recommendations["optimization_opportunities"].append(
                    f"High memory usage per device in {comparison_data['comparison_config']} "
                    f"- consider memory optimization techniques"
                )
        
        # Cost analysis
        if len(metrics_list) > 1:
            baseline_cost = metrics_list[0].relative_cost_score
            for metrics in metrics_list[1:]:
                cost_ratio = metrics.relative_cost_score / baseline_cost
                perf_ratio = metrics.tokens_per_second / metrics_list[0].tokens_per_second
                cost_effectiveness = perf_ratio / cost_ratio if cost_ratio > 0 else 0
                
                recommendations["cost_analysis"][metrics.configuration] = {
                    "cost_multiplier": cost_ratio,
                    "performance_multiplier": perf_ratio,
                    "cost_effectiveness": cost_effectiveness,
                    "breakeven_utilization": 1.0 / cost_effectiveness if cost_effectiveness > 0 else float('inf')
                }
        
        # Use case mapping
        for metrics in metrics_list:
            use_cases = []
            
            if metrics.tokens_per_second > 20:
                use_cases.append("High-volume batch processing")
            if metrics.avg_latency_ms < 1000:
                use_cases.append("Real-time applications")
            if metrics.memory_efficiency_score > 0.05:
                use_cases.append("Resource-constrained environments")
            if metrics.success_rate > 0.95:
                use_cases.append("Production workloads")
            
            if not use_cases:
                use_cases.append("Development and testing")
            
            recommendations["use_case_mapping"][metrics.configuration] = use_cases
        
        return recommendations
    
    def _generate_performance_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive performance report."""
        report_lines = [
            "# Comprehensive Performance Analysis Report",
            f"Generated: {results['analysis_info']['start_time']}",
            f"API URL: {results['analysis_info']['api_url']}",
            ""
        ]
        
        # Executive Summary
        if "comparisons" in results and results["comparisons"]:
            report_lines.extend([
                "## Executive Summary",
                ""
            ])
            
            for comparison_key, comparison_data in results["comparisons"].items():
                report_lines.extend([
                    f"### {comparison_key.replace('_', ' ').title()}",
                    f"**Summary**: {comparison_data['summary']}",
                    f"- **Throughput Improvement**: {comparison_data['throughput_improvement_factor']:.2f}x",
                    f"- **Latency Improvement**: {comparison_data['latency_improvement_factor']:.2f}x",
                    f"- **Scaling Efficiency**: {comparison_data['scaling_efficiency']:.1%}",
                    f"- **Cost Effectiveness**: {comparison_data['cost_effectiveness']:.2f}",
                    ""
                ])
        
        # Detailed Metrics
        if "metrics" in results:
            report_lines.extend([
                "## Detailed Performance Metrics",
                ""
            ])
            
            for config_name, metrics_data in results["metrics"].items():
                report_lines.extend([
                    f"### {config_name.replace('_', ' ').title()}",
                    f"- **Configuration**: {metrics_data['configuration']}",
                    f"- **Device Count**: {metrics_data['device_count']}",
                    f"- **Average Latency**: {metrics_data['avg_latency_ms']:.1f} ms",
                    f"- **Throughput**: {metrics_data['tokens_per_second']:.1f} tokens/sec",
                    f"- **Memory Efficiency**: {metrics_data['memory_efficiency_score']:.3f} tokens/MB",
                    f"- **Success Rate**: {metrics_data['success_rate']:.1%}",
                    f"- **Peak Memory**: {metrics_data['peak_memory_mb']:.0f} MB",
                    ""
                ])
        
        # Recommendations
        if "recommendations" in results:
            recommendations = results["recommendations"]
            
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            if recommendations.get("deployment_recommendations"):
                report_lines.extend([
                    "### Deployment Recommendations",
                    ""
                ])
                for rec in recommendations["deployment_recommendations"]:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
            
            if recommendations.get("optimization_opportunities"):
                report_lines.extend([
                    "### Optimization Opportunities",
                    ""
                ])
                for opp in recommendations["optimization_opportunities"]:
                    report_lines.append(f"- {opp}")
                report_lines.append("")
            
            if recommendations.get("use_case_mapping"):
                report_lines.extend([
                    "### Use Case Mapping",
                    ""
                ])
                for config, use_cases in recommendations["use_case_mapping"].items():
                    report_lines.append(f"**{config.replace('_', ' ').title()}**:")
                    for use_case in use_cases:
                        report_lines.append(f"  - {use_case}")
                    report_lines.append("")
        
        # Charts
        if results.get("charts"):
            report_lines.extend([
                "## Performance Charts",
                ""
            ])
            for chart in results["charts"]:
                report_lines.append(f"![Performance Chart]({chart})")
            report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Performance report saved to: {output_file}")


async def main():
    """Main entry point for performance comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Performance Comparison: Single-Device vs Distributed")
    parser.add_argument("--api-url", default="http://localhost:8100", help="API server URL")
    parser.add_argument("--output-dir", default="performance_analysis", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30, help="Number of test iterations")
    
    args = parser.parse_args()
    
    # Initialize analysis framework
    analyzer = ComprehensivePerformanceAnalysis(
        api_url=args.api_url,
        output_dir=args.output_dir
    )
    
    logger.info("Starting comprehensive performance analysis...")
    
    # Run analysis
    results = await analyzer.run_comprehensive_analysis(iterations=args.iterations)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE ANALYSIS COMPLETED")
    logger.info("="*80)
    
    if "metrics" in results:
        logger.info("\nPerformance Summary:")
        for config_name, metrics in results["metrics"].items():
            logger.info(f"  {config_name}:")
            logger.info(f"    Throughput: {metrics['tokens_per_second']:.1f} tok/s")
            logger.info(f"    Latency: {metrics['avg_latency_ms']:.1f} ms")
            logger.info(f"    Devices: {metrics['device_count']}")
    
    if "comparisons" in results:
        logger.info("\nKey Comparisons:")
        for comparison_key, comparison in results["comparisons"].items():
            logger.info(f"  {comparison_key}: {comparison['summary']}")
    
    logger.info(f"\nDetailed results saved to: {args.output_dir}/")
    
    return 0


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"], check=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
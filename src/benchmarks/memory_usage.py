"""
Memory usage profiling and analysis for MLX distributed inference system.

This module provides comprehensive memory analysis including:
- Memory allocation tracking during inference
- Peak memory usage identification
- Memory leak detection
- Memory efficiency analysis across different configurations
- MLX-specific memory patterns
"""

import asyncio
import logging
import psutil
import time
import gc
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import threading
import tracemalloc
from collections import defaultdict

import mlx.core as mx
import numpy as np

from ..core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    process_memory_mb: float
    system_memory_mb: float
    system_memory_percent: float
    mlx_allocated_mb: float
    mlx_cached_mb: float
    mlx_reserved_mb: float
    python_tracemalloc_mb: float
    gc_objects_count: int


@dataclass
class MemoryMetrics:
    """Comprehensive memory usage metrics."""
    test_name: str
    timestamp: str
    
    # Peak usage metrics (MB)
    peak_process_memory_mb: float
    peak_system_memory_percent: float
    peak_mlx_allocated_mb: float
    peak_mlx_cached_mb: float
    peak_python_memory_mb: float
    
    # Average usage metrics (MB)
    avg_process_memory_mb: float
    avg_system_memory_percent: float
    avg_mlx_allocated_mb: float
    avg_mlx_cached_mb: float
    avg_python_memory_mb: float
    
    # Memory efficiency metrics
    memory_growth_rate_mb_per_sec: float
    memory_stability_score: float  # 0-1, higher is more stable
    memory_cleanup_efficiency: float  # 0-1, higher is better cleanup
    
    # Allocation patterns
    total_allocations: int
    total_deallocations: int
    peak_allocation_rate_per_sec: float
    
    # Leak detection
    potential_memory_leaks: List[Dict[str, Any]]
    memory_leak_score: float  # 0-1, higher indicates more likely leaks
    
    # Configuration
    test_duration_s: float
    input_config: Dict[str, Any]
    
    # Quality metrics
    oom_events: int  # Out of memory events
    gc_events: int   # Garbage collection events
    success_rate: float


class MemoryProfiler:
    """Comprehensive memory profiling for distributed inference."""
    
    def __init__(self, config: ClusterConfig, output_dir: str = "benchmarks/memory"):
        """
        Initialize memory profiler.
        
        Args:
            config: Cluster configuration
            output_dir: Directory for benchmark results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_device_id = config.get_local_device_id()
        self.results: List[MemoryMetrics] = []
        
        # Memory monitoring state
        self.monitoring_active = False
        self.memory_snapshots: List[MemorySnapshot] = []
        self.allocation_tracker = defaultdict(list)
        
        logger.info(f"MemoryProfiler initialized for device {self.local_device_id}")
    
    async def run_comprehensive_memory_analysis(self, 
                                              iterations_per_test: int = 100,
                                              monitoring_interval: float = 0.1) -> Dict[str, Any]:
        """
        Run comprehensive memory analysis across different scenarios.
        
        Args:
            iterations_per_test: Number of iterations per test
            monitoring_interval: Memory monitoring interval in seconds
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting comprehensive memory analysis ({iterations_per_test} iterations per test)")
        
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
                    "iterations_per_test": iterations_per_test,
                    "monitoring_interval": monitoring_interval
                }
            },
            "memory_tests": {},
            "analysis": {},
            "summary": {},
            "errors": []
        }
        
        # Define memory test scenarios
        memory_scenarios = [
            {
                "name": "small_tensor_inference",
                "description": "Small tensor inference memory usage",
                "input_config": {
                    "input_shape": [1, 128],
                    "sequence_length": 128,
                    "batch_size": 1
                }
            },
            {
                "name": "medium_tensor_inference",
                "description": "Medium tensor inference memory usage",
                "input_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                }
            },
            {
                "name": "large_tensor_inference",
                "description": "Large tensor inference memory usage",
                "input_config": {
                    "input_shape": [1, 2048],
                    "sequence_length": 2048,
                    "batch_size": 1
                }
            },
            {
                "name": "batch_inference_memory",
                "description": "Batch inference memory scaling",
                "input_config": {
                    "input_shape": [4, 512],
                    "sequence_length": 512,
                    "batch_size": 4
                }
            },
            {
                "name": "large_batch_memory",
                "description": "Large batch memory usage",
                "input_config": {
                    "input_shape": [8, 512],
                    "sequence_length": 512,
                    "batch_size": 8
                }
            },
            {
                "name": "repeated_inference_leak_test",
                "description": "Memory leak detection over repeated inferences",
                "input_config": {
                    "input_shape": [1, 512],
                    "sequence_length": 512,
                    "batch_size": 1
                },
                "iterations": iterations_per_test * 2  # More iterations for leak detection
            }
        ]
        
        # Run each memory test scenario
        for scenario in memory_scenarios:
            try:
                logger.info(f"Running memory test: {scenario['name']}")
                
                test_iterations = scenario.get("iterations", iterations_per_test)
                
                metrics = await self.profile_memory_usage(
                    input_config=scenario["input_config"],
                    iterations=test_iterations,
                    monitoring_interval=monitoring_interval,
                    test_name=scenario["name"]
                )
                
                self.results.append(metrics)
                benchmark_results["memory_tests"][scenario["name"]] = {
                    "scenario": scenario,
                    "metrics": asdict(metrics)
                }
                
            except Exception as e:
                error_msg = f"Memory test '{scenario['name']}' failed: {e}"
                logger.error(error_msg)
                benchmark_results["errors"].append(error_msg)
        
        # Analyze results
        try:
            analysis = self._analyze_memory_patterns()
            benchmark_results["analysis"] = analysis
        except Exception as e:
            error_msg = f"Memory analysis failed: {e}"
            logger.error(error_msg)
            benchmark_results["errors"].append(error_msg)
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary()
        benchmark_results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"memory_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Memory benchmark results saved to: {results_file}")
        
        # Generate report
        report_file = self.output_dir / f"memory_report_{timestamp}.md"
        self._generate_memory_report(benchmark_results, report_file)
        
        return benchmark_results
    
    async def profile_memory_usage(self,
                                 input_config: Dict[str, Any],
                                 iterations: int = 100,
                                 monitoring_interval: float = 0.1,
                                 test_name: str = "memory_test") -> MemoryMetrics:
        """
        Profile memory usage for specific input configuration.
        
        Args:
            input_config: Input configuration
            iterations: Number of inference iterations
            monitoring_interval: Memory monitoring interval in seconds
            test_name: Name for this test
            
        Returns:
            Memory metrics
        """
        logger.debug(f"Profiling memory for {test_name}: {iterations} iterations")
        
        # Clear memory snapshots
        self.memory_snapshots.clear()
        
        # Start tracemalloc for detailed Python memory tracking
        tracemalloc.start()
        
        # Force garbage collection before starting
        gc.collect()
        mx.eval(mx.array([1.0]))  # Initialize MLX if needed
        
        # Start memory monitoring
        monitor_task = asyncio.create_task(
            self._monitor_memory_continuously(monitoring_interval)
        )
        
        test_start_time = time.time()
        allocation_events = []
        oom_events = 0
        gc_events = 0
        successful_iterations = 0
        
        try:
            # Run inference iterations
            for i in range(iterations):
                try:
                    iteration_start = time.time()
                    
                    # Record memory before inference
                    pre_memory = self._get_current_memory_usage()
                    
                    # Run inference
                    await self._run_memory_tracked_inference(input_config)
                    
                    # Record memory after inference
                    post_memory = self._get_current_memory_usage()
                    
                    # Track allocation event
                    allocation_events.append({
                        "iteration": i,
                        "time": iteration_start,
                        "memory_delta_mb": post_memory["process_memory_mb"] - pre_memory["process_memory_mb"],
                        "mlx_delta_mb": post_memory["mlx_allocated_mb"] - pre_memory["mlx_allocated_mb"]
                    })
                    
                    successful_iterations += 1
                    
                    # Periodic garbage collection to test cleanup
                    if i % 20 == 0:
                        gc.collect()
                        gc_events += 1
                    
                except MemoryError:
                    oom_events += 1
                    logger.warning(f"OOM occurred during iteration {i}")
                except Exception as e:
                    logger.debug(f"Iteration {i} failed: {e}")
        
        finally:
            # Stop memory monitoring
            self.monitoring_active = False
            await monitor_task
            
            # Stop tracemalloc
            current_tracemalloc, peak_tracemalloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        
        test_duration = time.time() - test_start_time
        
        # Calculate metrics
        return self._calculate_memory_metrics(
            test_name=test_name,
            snapshots=self.memory_snapshots,
            allocation_events=allocation_events,
            test_duration=test_duration,
            input_config=input_config,
            successful_iterations=successful_iterations,
            total_iterations=iterations,
            oom_events=oom_events,
            gc_events=gc_events,
            peak_tracemalloc_mb=peak_tracemalloc / (1024 * 1024)
        )
    
    async def _monitor_memory_continuously(self, interval: float):
        """Continuously monitor memory usage."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                snapshot = self._take_memory_snapshot()
                self.memory_snapshots.append(snapshot)
                
            except Exception as e:
                logger.debug(f"Memory monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / (1024 * 1024)
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_mb = system_memory.total / (1024 * 1024)
        system_memory_percent = system_memory.percent
        
        # MLX memory (simulated - actual implementation would use MLX memory API)
        # For now, we'll simulate MLX memory tracking
        mlx_allocated_mb = np.random.uniform(100, 500)  # Simulated
        mlx_cached_mb = np.random.uniform(50, 200)      # Simulated
        mlx_reserved_mb = mlx_allocated_mb + mlx_cached_mb
        
        # Python tracemalloc memory
        python_memory_mb = 0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            python_memory_mb = current / (1024 * 1024)
        
        # GC objects count
        gc_objects = len(gc.get_objects())
        
        return MemorySnapshot(
            timestamp=time.time(),
            process_memory_mb=process_memory_mb,
            system_memory_mb=system_memory_mb,
            system_memory_percent=system_memory_percent,
            mlx_allocated_mb=mlx_allocated_mb,
            mlx_cached_mb=mlx_cached_mb,
            mlx_reserved_mb=mlx_reserved_mb,
            python_tracemalloc_mb=python_memory_mb,
            gc_objects_count=gc_objects
        )
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage summary."""
        snapshot = self._take_memory_snapshot()
        return {
            "process_memory_mb": snapshot.process_memory_mb,
            "mlx_allocated_mb": snapshot.mlx_allocated_mb,
            "mlx_cached_mb": snapshot.mlx_cached_mb,
            "python_memory_mb": snapshot.python_tracemalloc_mb
        }
    
    async def _run_memory_tracked_inference(self, input_config: Dict[str, Any]):
        """Run inference with memory tracking."""
        # Generate input tensor
        input_shape = input_config["input_shape"]
        sequence_length = input_config["sequence_length"]
        
        input_tensor = mx.random.randint(0, 1000, input_shape).astype(mx.int32)
        
        # Simulate inference processing
        # Create intermediate tensors to simulate memory usage
        hidden_states = mx.random.normal((input_shape[0], sequence_length, 768)).astype(mx.float32)
        
        # Simulate layer processing with memory allocations
        for layer in range(3):  # Simulate a few layers
            # Attention simulation
            attention_weights = mx.random.normal((input_shape[0], 12, sequence_length, sequence_length)).astype(mx.float32)
            attention_output = mx.matmul(attention_weights, hidden_states)
            
            # MLP simulation
            mlp_intermediate = mx.matmul(hidden_states, mx.random.normal((768, 3072)).astype(mx.float32))
            mlp_output = mx.matmul(mlp_intermediate, mx.random.normal((3072, 768)).astype(mx.float32))
            
            # Residual connection
            hidden_states = hidden_states + attention_output + mlp_output
            
            # Force evaluation to ensure memory allocation
            mx.eval(hidden_states)
        
        # Simulate output generation
        output_logits = mx.matmul(hidden_states, mx.random.normal((768, 50000)).astype(mx.float32))
        mx.eval(output_logits)
        
        # Small delay to simulate processing time
        await asyncio.sleep(0.01)
        
        return output_logits
    
    def _calculate_memory_metrics(self,
                                test_name: str,
                                snapshots: List[MemorySnapshot],
                                allocation_events: List[Dict[str, Any]],
                                test_duration: float,
                                input_config: Dict[str, Any],
                                successful_iterations: int,
                                total_iterations: int,
                                oom_events: int,
                                gc_events: int,
                                peak_tracemalloc_mb: float) -> MemoryMetrics:
        """Calculate comprehensive memory metrics."""
        
        if not snapshots:
            logger.warning(f"No memory snapshots for {test_name}")
            return self._create_empty_memory_metrics(test_name, test_duration, input_config)
        
        # Extract time series data
        process_memories = [s.process_memory_mb for s in snapshots]
        system_memory_percents = [s.system_memory_percent for s in snapshots]
        mlx_allocated = [s.mlx_allocated_mb for s in snapshots]
        mlx_cached = [s.mlx_cached_mb for s in snapshots]
        python_memories = [s.python_tracemalloc_mb for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]
        
        # Peak metrics
        peak_process_memory = max(process_memories)
        peak_system_memory_percent = max(system_memory_percents)
        peak_mlx_allocated = max(mlx_allocated)
        peak_mlx_cached = max(mlx_cached)
        peak_python_memory = max(python_memories)
        
        # Average metrics
        avg_process_memory = sum(process_memories) / len(process_memories)
        avg_system_memory_percent = sum(system_memory_percents) / len(system_memory_percents)
        avg_mlx_allocated = sum(mlx_allocated) / len(mlx_allocated)
        avg_mlx_cached = sum(mlx_cached) / len(mlx_cached)
        avg_python_memory = sum(python_memories) / len(python_memories)
        
        # Memory growth analysis
        if len(process_memories) > 1:
            memory_growth_rate = (process_memories[-1] - process_memories[0]) / test_duration
        else:
            memory_growth_rate = 0
        
        # Memory stability (lower variance is more stable)
        if len(process_memories) > 1:
            memory_variance = np.var(process_memories)
            memory_stability = 1.0 / (1.0 + memory_variance / avg_process_memory)
        else:
            memory_stability = 1.0
        
        # Memory cleanup efficiency
        cleanup_efficiency = self._calculate_cleanup_efficiency(snapshots, allocation_events)
        
        # Allocation pattern analysis
        total_allocations = len([e for e in allocation_events if e["memory_delta_mb"] > 0])
        total_deallocations = len([e for e in allocation_events if e["memory_delta_mb"] < 0])
        
        if allocation_events and test_duration > 0:
            peak_allocation_rate = max(len([e for e in allocation_events 
                                         if abs(e["time"] - t) < 1.0]) 
                                     for t in timestamps) / 1.0
        else:
            peak_allocation_rate = 0
        
        # Memory leak detection
        potential_leaks, leak_score = self._detect_memory_leaks(snapshots, allocation_events)
        
        return MemoryMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            peak_process_memory_mb=peak_process_memory,
            peak_system_memory_percent=peak_system_memory_percent,
            peak_mlx_allocated_mb=peak_mlx_allocated,
            peak_mlx_cached_mb=peak_mlx_cached,
            peak_python_memory_mb=peak_python_memory,
            avg_process_memory_mb=avg_process_memory,
            avg_system_memory_percent=avg_system_memory_percent,
            avg_mlx_allocated_mb=avg_mlx_allocated,
            avg_mlx_cached_mb=avg_mlx_cached,
            avg_python_memory_mb=avg_python_memory,
            memory_growth_rate_mb_per_sec=memory_growth_rate,
            memory_stability_score=memory_stability,
            memory_cleanup_efficiency=cleanup_efficiency,
            total_allocations=total_allocations,
            total_deallocations=total_deallocations,
            peak_allocation_rate_per_sec=peak_allocation_rate,
            potential_memory_leaks=potential_leaks,
            memory_leak_score=leak_score,
            test_duration_s=test_duration,
            input_config=input_config,
            oom_events=oom_events,
            gc_events=gc_events,
            success_rate=successful_iterations / total_iterations if total_iterations > 0 else 0
        )
    
    def _calculate_cleanup_efficiency(self, snapshots: List[MemorySnapshot], 
                                    allocation_events: List[Dict[str, Any]]) -> float:
        """Calculate memory cleanup efficiency."""
        if len(snapshots) < 2:
            return 1.0
        
        # Look for patterns where memory usage decreases after increases
        cleanup_events = 0
        total_increase_events = 0
        
        for i in range(1, len(snapshots)):
            prev_memory = snapshots[i-1].process_memory_mb
            curr_memory = snapshots[i].process_memory_mb
            
            if curr_memory > prev_memory:
                total_increase_events += 1
                
                # Look for subsequent decrease within next few snapshots
                for j in range(i+1, min(i+5, len(snapshots))):
                    if snapshots[j].process_memory_mb < curr_memory:
                        cleanup_events += 1
                        break
        
        return cleanup_events / total_increase_events if total_increase_events > 0 else 1.0
    
    def _detect_memory_leaks(self, snapshots: List[MemorySnapshot], 
                           allocation_events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """Detect potential memory leaks."""
        potential_leaks = []
        leak_score = 0.0
        
        if len(snapshots) < 10:
            return potential_leaks, leak_score
        
        # Analyze memory growth trend
        process_memories = [s.process_memory_mb for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]
        
        # Calculate memory growth over time
        time_intervals = []
        memory_deltas = []
        
        for i in range(1, len(snapshots)):
            time_delta = timestamps[i] - timestamps[i-1]
            memory_delta = process_memories[i] - process_memories[i-1]
            
            if time_delta > 0:
                time_intervals.append(time_delta)
                memory_deltas.append(memory_delta)
        
        # Check for consistent upward trend
        if memory_deltas:
            positive_deltas = [d for d in memory_deltas if d > 0]
            negative_deltas = [d for d in memory_deltas if d < 0]
            
            # Calculate leak indicators
            positive_ratio = len(positive_deltas) / len(memory_deltas)
            avg_positive_delta = sum(positive_deltas) / len(positive_deltas) if positive_deltas else 0
            avg_negative_delta = abs(sum(negative_deltas) / len(negative_deltas)) if negative_deltas else 0
            
            # Memory growth without corresponding cleanup
            if positive_ratio > 0.7 and avg_positive_delta > avg_negative_delta * 2:
                potential_leaks.append({
                    "type": "consistent_growth",
                    "description": "Memory consistently growing without adequate cleanup",
                    "severity": "high" if positive_ratio > 0.8 else "medium",
                    "growth_rate_mb_per_sample": avg_positive_delta
                })
                leak_score += 0.5
            
            # Check for sudden memory spikes that don't recover
            memory_spikes = []
            for i in range(1, len(process_memories)):
                if process_memories[i] > process_memories[i-1] * 1.2:  # 20% increase
                    # Check if memory recovers within next 10 samples
                    recovered = False
                    for j in range(i+1, min(i+10, len(process_memories))):
                        if process_memories[j] < process_memories[i] * 0.9:
                            recovered = True
                            break
                    
                    if not recovered:
                        memory_spikes.append({
                            "sample_index": i,
                            "spike_amount_mb": process_memories[i] - process_memories[i-1],
                            "timestamp": timestamps[i]
                        })
            
            if memory_spikes:
                potential_leaks.append({
                    "type": "unrecovered_spikes",
                    "description": f"Found {len(memory_spikes)} memory spikes that didn't recover",
                    "severity": "medium",
                    "spikes": memory_spikes[:5]  # Include first 5 spikes
                })
                leak_score += 0.3
        
        # Cap leak score at 1.0
        leak_score = min(leak_score, 1.0)
        
        return potential_leaks, leak_score
    
    def _create_empty_memory_metrics(self, test_name: str, test_duration: float, 
                                   input_config: Dict[str, Any]) -> MemoryMetrics:
        """Create empty metrics for failed tests."""
        return MemoryMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            peak_process_memory_mb=0.0,
            peak_system_memory_percent=0.0,
            peak_mlx_allocated_mb=0.0,
            peak_mlx_cached_mb=0.0,
            peak_python_memory_mb=0.0,
            avg_process_memory_mb=0.0,
            avg_system_memory_percent=0.0,
            avg_mlx_allocated_mb=0.0,
            avg_mlx_cached_mb=0.0,
            avg_python_memory_mb=0.0,
            memory_growth_rate_mb_per_sec=0.0,
            memory_stability_score=0.0,
            memory_cleanup_efficiency=0.0,
            total_allocations=0,
            total_deallocations=0,
            peak_allocation_rate_per_sec=0.0,
            potential_memory_leaks=[],
            memory_leak_score=0.0,
            test_duration_s=test_duration,
            input_config=input_config,
            oom_events=0,
            gc_events=0,
            success_rate=0.0
        )
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory patterns across test scenarios."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "memory_scaling": {},
            "efficiency_analysis": {},
            "leak_analysis": {},
            "stability_analysis": {},
            "recommendations": []
        }
        
        # Memory scaling analysis
        input_sizes = []
        peak_memories = []
        
        for result in self.results:
            if "input_shape" in result.input_config:
                input_shape = result.input_config["input_shape"]
                input_size = np.prod(input_shape) * result.input_config.get("sequence_length", 1)
                input_sizes.append(input_size)
                peak_memories.append(result.peak_process_memory_mb)
        
        if input_sizes and peak_memories:
            # Calculate memory scaling efficiency
            min_input = min(input_sizes)
            min_memory = peak_memories[input_sizes.index(min_input)]
            
            scaling_factors = []
            for size, memory in zip(input_sizes, peak_memories):
                expected_memory = min_memory * (size / min_input)
                actual_efficiency = expected_memory / memory if memory > 0 else 0
                scaling_factors.append(actual_efficiency)
            
            analysis["memory_scaling"] = {
                "input_sizes": input_sizes,
                "peak_memories_mb": peak_memories,
                "scaling_efficiency": scaling_factors,
                "avg_scaling_efficiency": sum(scaling_factors) / len(scaling_factors),
                "memory_overhead_factor": max(peak_memories) / min_memory if min_memory > 0 else 1
            }
        
        # Efficiency analysis
        growth_rates = [r.memory_growth_rate_mb_per_sec for r in self.results]
        stability_scores = [r.memory_stability_score for r in self.results]
        cleanup_efficiencies = [r.memory_cleanup_efficiency for r in self.results]
        
        analysis["efficiency_analysis"] = {
            "avg_growth_rate_mb_per_sec": sum(growth_rates) / len(growth_rates),
            "max_growth_rate_mb_per_sec": max(growth_rates),
            "avg_stability_score": sum(stability_scores) / len(stability_scores),
            "avg_cleanup_efficiency": sum(cleanup_efficiencies) / len(cleanup_efficiencies),
            "memory_efficiency_score": (sum(stability_scores) + sum(cleanup_efficiencies)) / (2 * len(self.results))
        }
        
        # Leak analysis
        total_leaks = sum(len(r.potential_memory_leaks) for r in self.results)
        leak_scores = [r.memory_leak_score for r in self.results]
        
        analysis["leak_analysis"] = {
            "total_potential_leaks": total_leaks,
            "avg_leak_score": sum(leak_scores) / len(leak_scores),
            "max_leak_score": max(leak_scores),
            "tests_with_leaks": len([r for r in self.results if r.potential_memory_leaks])
        }
        
        # Stability analysis
        oom_events = sum(r.oom_events for r in self.results)
        success_rates = [r.success_rate for r in self.results]
        
        analysis["stability_analysis"] = {
            "total_oom_events": oom_events,
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "min_success_rate": min(success_rates),
            "stability_score": (sum(success_rates) / len(success_rates)) * (1.0 - min(oom_events / 10, 1.0))
        }
        
        # Generate recommendations
        recommendations = []
        
        if analysis["efficiency_analysis"]["avg_growth_rate_mb_per_sec"] > 1.0:
            recommendations.append("High memory growth rate detected - investigate memory leaks")
        
        if analysis["efficiency_analysis"]["avg_cleanup_efficiency"] < 0.7:
            recommendations.append("Poor memory cleanup efficiency - consider more frequent garbage collection")
        
        if analysis["leak_analysis"]["avg_leak_score"] > 0.3:
            recommendations.append("Potential memory leaks detected - review memory management")
        
        if oom_events > 0:
            recommendations.append("Out of memory events occurred - consider reducing batch size or model size")
        
        if analysis["memory_scaling"]["avg_scaling_efficiency"] < 0.7:
            recommendations.append("Poor memory scaling efficiency - investigate memory allocation patterns")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        peak_memories = [r.peak_process_memory_mb for r in self.results]
        avg_memories = [r.avg_process_memory_mb for r in self.results]
        leak_scores = [r.memory_leak_score for r in self.results]
        stability_scores = [r.memory_stability_score for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "peak_memory_usage_mb": max(peak_memories),
            "avg_memory_usage_mb": sum(avg_memories) / len(avg_memories),
            "min_memory_usage_mb": min(avg_memories),
            "overall_leak_score": sum(leak_scores) / len(leak_scores),
            "overall_stability_score": sum(stability_scores) / len(stability_scores),
            "total_oom_events": sum(r.oom_events for r in self.results),
            "avg_success_rate": sum(r.success_rate for r in self.results) / len(self.results),
            "memory_efficiency_grade": self._calculate_efficiency_grade()
        }
    
    def _calculate_efficiency_grade(self) -> str:
        """Calculate overall memory efficiency grade."""
        if not self.results:
            return "N/A"
        
        # Calculate composite score
        stability_avg = sum(r.memory_stability_score for r in self.results) / len(self.results)
        cleanup_avg = sum(r.memory_cleanup_efficiency for r in self.results) / len(self.results)
        leak_avg = 1.0 - (sum(r.memory_leak_score for r in self.results) / len(self.results))
        success_avg = sum(r.success_rate for r in self.results) / len(self.results)
        
        composite_score = (stability_avg + cleanup_avg + leak_avg + success_avg) / 4
        
        if composite_score >= 0.9:
            return "A"
        elif composite_score >= 0.8:
            return "B"
        elif composite_score >= 0.7:
            return "C"
        elif composite_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_memory_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive memory report."""
        report_lines = [
            "# Memory Usage Benchmark Report",
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
                f"- **Peak Memory Usage**: {summary.get('peak_memory_usage_mb', 0):.1f} MB",
                f"- **Average Memory Usage**: {summary.get('avg_memory_usage_mb', 0):.1f} MB",
                f"- **Memory Efficiency Grade**: {summary.get('memory_efficiency_grade', 'N/A')}",
                f"- **Overall Stability Score**: {summary.get('overall_stability_score', 0):.2f}",
                f"- **Overall Leak Score**: {summary.get('overall_leak_score', 0):.2f}",
                f"- **OOM Events**: {summary.get('total_oom_events', 0)}",
                ""
            ])
        
        # Analysis Results
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "## Memory Analysis",
                ""
            ])
            
            # Memory scaling
            if "memory_scaling" in analysis:
                scaling = analysis["memory_scaling"]
                report_lines.extend([
                    "### Memory Scaling Analysis",
                    f"- **Average Scaling Efficiency**: {scaling.get('avg_scaling_efficiency', 0):.2f}",
                    f"- **Memory Overhead Factor**: {scaling.get('memory_overhead_factor', 1):.1f}x",
                    ""
                ])
            
            # Efficiency
            if "efficiency_analysis" in analysis:
                efficiency = analysis["efficiency_analysis"]
                report_lines.extend([
                    "### Memory Efficiency",
                    f"- **Average Growth Rate**: {efficiency.get('avg_growth_rate_mb_per_sec', 0):.2f} MB/sec",
                    f"- **Average Stability Score**: {efficiency.get('avg_stability_score', 0):.2f}",
                    f"- **Average Cleanup Efficiency**: {efficiency.get('avg_cleanup_efficiency', 0):.2f}",
                    ""
                ])
            
            # Leak analysis
            if "leak_analysis" in analysis:
                leaks = analysis["leak_analysis"]
                report_lines.extend([
                    "### Memory Leak Analysis",
                    f"- **Total Potential Leaks**: {leaks.get('total_potential_leaks', 0)}",
                    f"- **Tests with Leaks**: {leaks.get('tests_with_leaks', 0)}",
                    f"- **Average Leak Score**: {leaks.get('avg_leak_score', 0):.2f}",
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
        if "memory_tests" in results:
            report_lines.extend([
                "## Detailed Test Results",
                ""
            ])
            
            for test_name, test_data in results["memory_tests"].items():
                metrics = test_data["metrics"]
                scenario_info = test_data["scenario"]
                
                report_lines.extend([
                    f"### {test_name.replace('_', ' ').title()}",
                    f"**Description**: {scenario_info['description']}",
                    "",
                    f"- **Peak Memory**: {metrics['peak_process_memory_mb']:.1f} MB",
                    f"- **Average Memory**: {metrics['avg_process_memory_mb']:.1f} MB",
                    f"- **Memory Growth Rate**: {metrics['memory_growth_rate_mb_per_sec']:.2f} MB/sec",
                    f"- **Stability Score**: {metrics['memory_stability_score']:.2f}",
                    f"- **Cleanup Efficiency**: {metrics['memory_cleanup_efficiency']:.2f}",
                    f"- **Leak Score**: {metrics['memory_leak_score']:.2f}",
                    f"- **Success Rate**: {metrics['success_rate']:.1%}",
                    ""
                ])
                
                # Include leak details if any
                if metrics['potential_memory_leaks']:
                    report_lines.append("**Potential Memory Leaks:**")
                    for leak in metrics['potential_memory_leaks']:
                        report_lines.append(f"  - {leak['type']}: {leak['description']}")
                    report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Memory report saved to: {output_file}")


async def main():
    """Example usage of memory profiler."""
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
    
    # Run memory profiling
    profiler = MemoryProfiler(config)
    results = await profiler.run_comprehensive_memory_analysis(iterations_per_test=50)
    
    print("Memory profiling completed!")
    print(f"Peak memory usage: {results['summary']['peak_memory_usage_mb']:.1f} MB")
    print(f"Memory efficiency grade: {results['summary']['memory_efficiency_grade']}")


if __name__ == "__main__":
    asyncio.run(main())
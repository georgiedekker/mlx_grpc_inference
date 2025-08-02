#!/usr/bin/env python3
"""
Comprehensive benchmark runner for MLX distributed inference system.

This script runs all performance benchmarks and generates a complete
performance analysis report.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ClusterConfig
from benchmarks.performance_validator import PerformanceValidator, create_default_performance_targets
from benchmarks.latency_benchmark import LatencyBenchmark
from benchmarks.throughput_benchmark import ThroughputBenchmark
from benchmarks.memory_usage import MemoryProfiler
from benchmarks.network_performance import NetworkAnalyzer

logger = logging.getLogger(__name__)


async def run_individual_benchmarks(config: ClusterConfig, output_dir: str, args):
    """Run individual benchmarks based on command line arguments."""
    results = {}
    
    if args.latency:
        logger.info("Running latency benchmark...")
        benchmark = LatencyBenchmark(config, f"{output_dir}/latency")
        results['latency'] = await benchmark.run_comprehensive_latency_analysis(
            iterations=args.iterations,
            warmup_iterations=args.warmup
        )
    
    if args.throughput:
        logger.info("Running throughput benchmark...")
        benchmark = ThroughputBenchmark(config, f"{output_dir}/throughput")
        results['throughput'] = await benchmark.run_comprehensive_throughput_analysis(
            test_duration=args.duration,
            warmup_duration=args.warmup_duration
        )
    
    if args.memory:
        logger.info("Running memory profiling...")
        profiler = MemoryProfiler(config, f"{output_dir}/memory")
        results['memory'] = await profiler.run_comprehensive_memory_analysis(
            iterations_per_test=args.memory_iterations,
            monitoring_interval=args.memory_interval
        )
    
    if args.network:
        logger.info("Running network analysis...")
        analyzer = NetworkAnalyzer(config, f"{output_dir}/network")
        results['network'] = await analyzer.run_comprehensive_network_analysis(
            test_duration=args.network_duration,
            sampling_interval=args.network_interval
        )
    
    return results


async def run_validation_suite(config: ClusterConfig, output_dir: str, args):
    """Run complete validation suite."""
    logger.info("Running comprehensive performance validation...")
    
    # Create performance targets
    targets = create_default_performance_targets()
    
    # Override targets if provided
    if args.target_latency:
        for target in targets:
            if target.metric_name == "avg_latency_ms":
                target.target_value = args.target_latency
    
    if args.target_throughput:
        for target in targets:
            if target.metric_name == "peak_requests_per_second":
                target.target_value = args.target_throughput
    
    if args.target_memory:
        for target in targets:
            if target.metric_name == "peak_memory_usage_mb":
                target.target_value = args.target_memory
    
    # Run validation
    validator = PerformanceValidator(
        config=config,
        output_dir=output_dir,
        baseline_file=args.baseline_file
    )
    
    results = await validator.run_full_validation(
        performance_targets=targets,
        create_baseline=args.create_baseline,
        regression_test=args.regression_test
    )
    
    return results


def print_summary_report(results, benchmark_type="validation"):
    """Print a summary report to console."""
    print("\n" + "="*80)
    print(f"PERFORMANCE BENCHMARK SUMMARY - {benchmark_type.upper()}")
    print("="*80)
    
    if benchmark_type == "validation":
        # Validation summary
        summary = results.get("summary", {})
        print(f"Overall Score: {summary.get('overall_score', 0):.2f}/1.00")
        print(f"Successful Benchmarks: {summary.get('successful_benchmarks', 0)}/{summary.get('total_benchmarks', 0)}")
        
        if summary.get("target_validations"):
            tv = summary["target_validations"]
            print(f"Target Validation: {tv['passed_targets']}/{tv['total_targets']} passed ({tv['pass_rate']:.1%})")
        
        print(f"Regression Status: {summary.get('regression_status', 'unknown').replace('_', ' ').title()}")
        
        # Print recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\nKey Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"  {i}. {rec}")
            if len(recommendations) > 5:
                print(f"  ... and {len(recommendations) - 5} more")
    
    else:
        # Individual benchmark summaries
        for benchmark_name, result in results.items():
            if "summary" in result:
                summary = result["summary"]
                print(f"\n{benchmark_name.title()} Benchmark:")
                
                if benchmark_name == "latency":
                    print(f"  Average Latency: {summary.get('overall_avg_latency_ms', 0):.1f} ms")
                    print(f"  95th Percentile: {summary.get('overall_p95_latency_ms', 0):.1f} ms")
                    print(f"  Success Rate: {summary.get('overall_success_rate', 0):.1%}")
                
                elif benchmark_name == "throughput":
                    print(f"  Peak Requests/sec: {summary.get('peak_requests_per_second', 0):.1f}")
                    print(f"  Peak Tokens/sec: {summary.get('peak_tokens_per_second', 0):.1f}")
                    print(f"  Success Rate: {summary.get('overall_success_rate', 0):.1%}")
                
                elif benchmark_name == "memory":
                    print(f"  Peak Memory: {summary.get('peak_memory_usage_mb', 0):.1f} MB")
                    print(f"  Efficiency Grade: {summary.get('memory_efficiency_grade', 'N/A')}")
                    print(f"  Leak Score: {summary.get('overall_leak_score', 0):.2f}")
                
                elif benchmark_name == "network":
                    print(f"  Peak Bandwidth: {summary.get('peak_bandwidth_mbps', 0):.1f} Mbps")
                    print(f"  Efficiency Score: {summary.get('network_efficiency_score', 0):.2f}")
                    print(f"  Average Latency: {summary.get('avg_latency_ms', 0):.1f} ms")
    
    print("\n" + "="*80)


async def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MLX distributed inference benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with validation
  python run_all_benchmarks.py --config cluster_config.yaml --all

  # Run only latency and throughput benchmarks
  python run_all_benchmarks.py --config cluster_config.yaml --latency --throughput

  # Run validation with custom targets
  python run_all_benchmarks.py --config cluster_config.yaml --validation --target-latency 80

  # Create new baseline
  python run_all_benchmarks.py --config cluster_config.yaml --validation --create-baseline

  # Regression testing
  python run_all_benchmarks.py --config cluster_config.yaml --validation --baseline-file baseline.json
        """
    )
    
    # Configuration
    parser.add_argument("--config", required=True, help="Path to cluster configuration file")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    
    # Benchmark selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks and validation")
    parser.add_argument("--validation", action="store_true", help="Run performance validation suite")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmarks")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory profiling")
    parser.add_argument("--network", action="store_true", help="Run network analysis")
    
    # Benchmark parameters
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for latency tests")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--warmup-duration", type=int, default=10, help="Warmup duration in seconds")
    parser.add_argument("--memory-iterations", type=int, default=50, help="Iterations for memory tests")
    parser.add_argument("--memory-interval", type=float, default=0.1, help="Memory monitoring interval")
    parser.add_argument("--network-duration", type=int, default=30, help="Network test duration")
    parser.add_argument("--network-interval", type=float, default=1.0, help="Network sampling interval")
    
    # Validation parameters
    parser.add_argument("--target-latency", type=float, help="Target average latency in ms")
    parser.add_argument("--target-throughput", type=float, help="Target requests per second")
    parser.add_argument("--target-memory", type=float, help="Target peak memory in MB")
    parser.add_argument("--create-baseline", action="store_true", help="Create new performance baseline")
    parser.add_argument("--baseline-file", help="Baseline file for regression testing")
    parser.add_argument("--regression-test", action="store_true", help="Enable regression testing")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-summary", action="store_true", help="Skip console summary")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        config = ClusterConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration for cluster: {config.name}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine what to run
    if args.all:
        args.validation = True
        args.latency = True
        args.throughput = True
        args.memory = True
        args.network = True
        args.regression_test = True
    
    if not any([args.validation, args.latency, args.throughput, args.memory, args.network]):
        logger.error("No benchmarks selected. Use --all or specify individual benchmarks.")
        return 1
    
    try:
        results = {}
        
        # Run validation suite if requested
        if args.validation:
            validation_results = await run_validation_suite(config, str(output_dir), args)
            results['validation'] = validation_results
            
            if not args.no_summary:
                print_summary_report(validation_results, "validation")
        
        # Run individual benchmarks if requested
        individual_benchmarks = [args.latency, args.throughput, args.memory, args.network]
        if any(individual_benchmarks):
            benchmark_results = await run_individual_benchmarks(config, str(output_dir), args)
            results.update(benchmark_results)
            
            if not args.no_summary and benchmark_results:
                print_summary_report(benchmark_results, "individual")
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_results_file = output_dir / f"combined_results_{timestamp}.json"
        
        import json
        with open(combined_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Combined results saved to: {combined_results_file}")
        
        # Determine exit code based on results
        exit_code = 0
        
        if 'validation' in results:
            validation_summary = results['validation'].get('summary', {})
            overall_score = validation_summary.get('overall_score', 0)
            
            if overall_score < 0.6:
                logger.warning("Overall performance score is below acceptable threshold")
                exit_code = 1
            elif validation_summary.get('regression_status') == 'regression':
                logger.warning("Performance regression detected")
                exit_code = 1
        
        # Check for critical errors
        all_errors = []
        for result in results.values():
            if isinstance(result, dict) and 'errors' in result:
                all_errors.extend(result['errors'])
        
        if all_errors:
            logger.error(f"Encountered {len(all_errors)} errors during benchmarking")
            if len(all_errors) > 5:  # More than 5 errors is critical
                exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
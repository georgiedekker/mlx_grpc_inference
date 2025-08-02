"""
Comprehensive performance validation suite for MLX distributed inference system.

This module provides end-to-end performance validation including:
- Baseline performance establishment
- Regression testing
- Performance target validation
- Multi-dimensional performance analysis
- Automated performance reports
"""

import asyncio
import logging
import json
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .latency_benchmark import LatencyBenchmark
from .throughput_benchmark import ThroughputBenchmark
from .memory_usage import MemoryProfiler
from .network_performance import NetworkAnalyzer
from ..core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTarget:
    """Performance target definition."""
    metric_name: str
    target_value: float
    tolerance_percent: float
    comparison_type: str  # "less_than", "greater_than", "equals"
    priority: str  # "critical", "important", "nice_to_have"


@dataclass
class ValidationResult:
    """Result of a performance validation."""
    test_name: str
    target: PerformanceTarget
    actual_value: float
    target_value: float
    passed: bool
    deviation_percent: float
    severity: str


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    version: str
    timestamp: str
    metrics: Dict[str, float]
    environment: Dict[str, Any]
    config_hash: str


class PerformanceValidator:
    """Comprehensive performance validation suite."""
    
    def __init__(self, config: ClusterConfig, 
                 output_dir: str = "benchmarks/validation",
                 baseline_file: Optional[str] = None):
        """
        Initialize performance validator.
        
        Args:
            config: Cluster configuration
            output_dir: Directory for validation results
            baseline_file: Path to baseline performance file
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_file = baseline_file
        self.baseline: Optional[PerformanceBaseline] = None
        
        # Initialize benchmark components
        self.latency_benchmark = LatencyBenchmark(config, str(self.output_dir / "latency"))
        self.throughput_benchmark = ThroughputBenchmark(config, str(self.output_dir / "throughput"))
        self.memory_profiler = MemoryProfiler(config, str(self.output_dir / "memory"))
        self.network_analyzer = NetworkAnalyzer(config, str(self.output_dir / "network"))
        
        # Load baseline if available
        if baseline_file and Path(baseline_file).exists():
            self._load_baseline(baseline_file)
        
        logger.info(f"PerformanceValidator initialized for cluster {config.name}")
    
    async def run_full_validation(self, 
                                 performance_targets: Optional[List[PerformanceTarget]] = None,
                                 create_baseline: bool = False,
                                 regression_test: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive performance validation.
        
        Args:
            performance_targets: List of performance targets to validate against
            create_baseline: Whether to create a new baseline
            regression_test: Whether to perform regression testing
            
        Returns:
            Complete validation results
        """
        logger.info("Starting comprehensive performance validation")
        
        validation_results = {
            "validation_info": {
                "start_time": datetime.now().isoformat(),
                "cluster_name": self.config.name,
                "model_name": self.config.model.name,
                "total_devices": len(self.config.devices),
                "validation_config": {
                    "create_baseline": create_baseline,
                    "regression_test": regression_test,
                    "has_targets": performance_targets is not None
                }
            },
            "benchmark_results": {},
            "validation_results": {},
            "baseline_comparison": {},
            "summary": {},
            "recommendations": [],
            "errors": []
        }
        
        # Run all benchmarks
        try:
            # 1. Latency benchmarks
            logger.info("Running latency benchmarks...")
            latency_results = await self.latency_benchmark.run_comprehensive_latency_analysis(
                iterations=50, warmup_iterations=5
            )
            validation_results["benchmark_results"]["latency"] = latency_results
            
        except Exception as e:
            error_msg = f"Latency benchmark failed: {e}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
        
        try:
            # 2. Throughput benchmarks
            logger.info("Running throughput benchmarks...")
            throughput_results = await self.throughput_benchmark.run_comprehensive_throughput_analysis(
                test_duration=30, warmup_duration=5
            )
            validation_results["benchmark_results"]["throughput"] = throughput_results
            
        except Exception as e:
            error_msg = f"Throughput benchmark failed: {e}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
        
        try:
            # 3. Memory profiling
            logger.info("Running memory profiling...")
            memory_results = await self.memory_profiler.run_comprehensive_memory_analysis(
                iterations_per_test=30, monitoring_interval=0.5
            )
            validation_results["benchmark_results"]["memory"] = memory_results
            
        except Exception as e:
            error_msg = f"Memory profiling failed: {e}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
        
        try:
            # 4. Network analysis
            logger.info("Running network analysis...")
            network_results = await self.network_analyzer.run_comprehensive_network_analysis(
                test_duration=30, sampling_interval=1.0
            )
            validation_results["benchmark_results"]["network"] = network_results
            
        except Exception as e:
            error_msg = f"Network analysis failed: {e}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
        
        # Extract key metrics
        key_metrics = self._extract_key_metrics(validation_results["benchmark_results"])
        
        # Validate against targets
        if performance_targets:
            logger.info("Validating against performance targets...")
            target_validations = self._validate_against_targets(key_metrics, performance_targets)
            validation_results["validation_results"]["target_validation"] = target_validations
        
        # Regression testing
        if regression_test and self.baseline:
            logger.info("Performing regression testing...")
            regression_results = self._perform_regression_testing(key_metrics)
            validation_results["baseline_comparison"] = regression_results
        
        # Create new baseline if requested
        if create_baseline:
            logger.info("Creating new performance baseline...")
            new_baseline = self._create_baseline(key_metrics)
            baseline_file = self.output_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_baseline(new_baseline, str(baseline_file))
            validation_results["validation_info"]["baseline_created"] = str(baseline_file)
        
        # Generate summary and recommendations
        validation_results["summary"] = self._generate_validation_summary(validation_results)
        validation_results["recommendations"] = self._generate_recommendations(validation_results)
        
        # Save validation results
        validation_results["validation_info"]["end_time"] = datetime.now().isoformat()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {results_file}")
        
        # Generate validation report
        report_file = self.output_dir / f"validation_report_{timestamp}.md"
        self._generate_validation_report(validation_results, report_file)
        
        return validation_results
    
    def _extract_key_metrics(self, benchmark_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from benchmark results."""
        metrics = {}
        
        # Latency metrics
        if "latency" in benchmark_results and "summary" in benchmark_results["latency"]:
            latency_summary = benchmark_results["latency"]["summary"]
            metrics.update({
                "avg_latency_ms": latency_summary.get("overall_avg_latency_ms", 0),
                "p95_latency_ms": latency_summary.get("overall_p95_latency_ms", 0),
                "min_latency_ms": latency_summary.get("best_latency_ms", 0),
                "latency_success_rate": latency_summary.get("overall_success_rate", 0)
            })
        
        # Throughput metrics
        if "throughput" in benchmark_results and "summary" in benchmark_results["throughput"]:
            throughput_summary = benchmark_results["throughput"]["summary"]
            metrics.update({
                "peak_requests_per_second": throughput_summary.get("peak_requests_per_second", 0),
                "peak_tokens_per_second": throughput_summary.get("peak_tokens_per_second", 0),
                "avg_requests_per_second": throughput_summary.get("avg_requests_per_second", 0),
                "throughput_success_rate": throughput_summary.get("overall_success_rate", 0)
            })
        
        # Memory metrics
        if "memory" in benchmark_results and "summary" in benchmark_results["memory"]:
            memory_summary = benchmark_results["memory"]["summary"]
            metrics.update({
                "peak_memory_usage_mb": memory_summary.get("peak_memory_usage_mb", 0),
                "avg_memory_usage_mb": memory_summary.get("avg_memory_usage_mb", 0),
                "memory_efficiency_score": self._calculate_memory_efficiency_score(memory_summary.get("memory_efficiency_grade", "F")),
                "memory_leak_score": memory_summary.get("overall_leak_score", 0)
            })
        
        # Network metrics
        if "network" in benchmark_results and "summary" in benchmark_results["network"]:
            network_summary = benchmark_results["network"]["summary"]
            metrics.update({
                "peak_bandwidth_mbps": network_summary.get("peak_bandwidth_mbps", 0),
                "avg_bandwidth_mbps": network_summary.get("avg_bandwidth_mbps", 0),
                "avg_network_latency_ms": network_summary.get("avg_latency_ms", 0),
                "network_efficiency_score": network_summary.get("network_efficiency_score", 0)
            })
        
        return metrics
    
    def _calculate_memory_efficiency_score(self, grade: str) -> float:
        """Convert memory efficiency grade to numeric score."""
        grade_scores = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "F": 0.2}
        return grade_scores.get(grade, 0.0)
    
    def _validate_against_targets(self, metrics: Dict[str, float], 
                                targets: List[PerformanceTarget]) -> List[ValidationResult]:
        """Validate metrics against performance targets."""
        results = []
        
        for target in targets:
            if target.metric_name not in metrics:
                logger.warning(f"Metric '{target.metric_name}' not found in results")
                continue
            
            actual_value = metrics[target.metric_name]
            target_value = target.target_value
            
            # Determine if target is met
            passed = False
            if target.comparison_type == "less_than":
                passed = actual_value <= target_value * (1 + target.tolerance_percent / 100)
            elif target.comparison_type == "greater_than":
                passed = actual_value >= target_value * (1 - target.tolerance_percent / 100)
            elif target.comparison_type == "equals":
                tolerance = target_value * target.tolerance_percent / 100
                passed = abs(actual_value - target_value) <= tolerance
            
            # Calculate deviation
            if target_value > 0:
                deviation_percent = ((actual_value - target_value) / target_value) * 100
            else:
                deviation_percent = 0
            
            # Determine severity
            severity = "pass" if passed else self._determine_severity(target, deviation_percent)
            
            result = ValidationResult(
                test_name=f"target_{target.metric_name}",
                target=target,
                actual_value=actual_value,
                target_value=target_value,
                passed=passed,
                deviation_percent=deviation_percent,
                severity=severity
            )
            
            results.append(result)
        
        return results
    
    def _determine_severity(self, target: PerformanceTarget, deviation_percent: float) -> str:
        """Determine severity of target failure."""
        abs_deviation = abs(deviation_percent)
        
        if target.priority == "critical":
            if abs_deviation > 50:
                return "critical"
            elif abs_deviation > 25:
                return "major"
            else:
                return "minor"
        elif target.priority == "important":
            if abs_deviation > 75:
                return "major"
            elif abs_deviation > 50:
                return "minor"
            else:
                return "warning"
        else:  # nice_to_have
            return "info"
    
    def _perform_regression_testing(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform regression testing against baseline."""
        if not self.baseline:
            return {"error": "No baseline available for regression testing"}
        
        regression_results = {
            "baseline_version": self.baseline.version,
            "baseline_timestamp": self.baseline.timestamp,
            "metric_comparisons": {},
            "overall_regression": False,
            "significant_regressions": [],
            "improvements": []
        }
        
        regression_threshold = 10.0  # 10% regression threshold
        improvement_threshold = 5.0   # 5% improvement threshold
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.baseline.metrics:
                continue
            
            baseline_value = self.baseline.metrics[metric_name]
            
            if baseline_value > 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_percent = 0
            
            # Determine if this is a regression or improvement
            is_regression = False
            is_improvement = False
            
            # For metrics where lower is better (latency, memory usage)
            if "latency" in metric_name.lower() or "memory" in metric_name.lower() or "leak" in metric_name.lower():
                if change_percent > regression_threshold:
                    is_regression = True
                elif change_percent < -improvement_threshold:
                    is_improvement = True
            
            # For metrics where higher is better (throughput, success rate)
            else:
                if change_percent < -regression_threshold:
                    is_regression = True
                elif change_percent > improvement_threshold:
                    is_improvement = True
            
            comparison = {
                "baseline_value": baseline_value,
                "current_value": current_value,
                "change_percent": change_percent,
                "is_regression": is_regression,
                "is_improvement": is_improvement
            }
            
            regression_results["metric_comparisons"][metric_name] = comparison
            
            if is_regression:
                regression_results["overall_regression"] = True
                regression_results["significant_regressions"].append({
                    "metric": metric_name,
                    "change_percent": change_percent,
                    "baseline": baseline_value,
                    "current": current_value
                })
            
            if is_improvement:
                regression_results["improvements"].append({
                    "metric": metric_name,
                    "change_percent": change_percent,
                    "baseline": baseline_value,
                    "current": current_value
                })
        
        return regression_results
    
    def _create_baseline(self, metrics: Dict[str, float]) -> PerformanceBaseline:
        """Create a new performance baseline."""
        return PerformanceBaseline(
            version="1.0.0",  # Would be extracted from actual version
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            environment={
                "cluster_name": self.config.name,
                "model_name": self.config.model.name,
                "total_devices": len(self.config.devices),
                "total_layers": self.config.model.total_layers
            },
            config_hash=str(hash(str(self.config)))  # Simple config hash
        )
    
    def _load_baseline(self, baseline_file: str):
        """Load performance baseline from file."""
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                self.baseline = PerformanceBaseline(**data)
            logger.info(f"Loaded baseline from {baseline_file}")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
    
    def _save_baseline(self, baseline: PerformanceBaseline, baseline_file: str):
        """Save performance baseline to file."""
        try:
            with open(baseline_file, 'w') as f:
                json.dump(asdict(baseline), f, indent=2)
            logger.info(f"Saved baseline to {baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            "total_benchmarks": 0,
            "successful_benchmarks": 0,
            "failed_benchmarks": 0,
            "target_validations": {},
            "regression_status": "unknown",
            "overall_score": 0.0
        }
        
        # Count benchmark results
        benchmark_results = validation_results.get("benchmark_results", {})
        summary["total_benchmarks"] = len(benchmark_results)
        summary["successful_benchmarks"] = len([b for b in benchmark_results.values() if "errors" not in b or not b["errors"]])
        summary["failed_benchmarks"] = summary["total_benchmarks"] - summary["successful_benchmarks"]
        
        # Target validation summary
        target_validations = validation_results.get("validation_results", {}).get("target_validation", [])
        if target_validations:
            passed = len([v for v in target_validations if v["passed"]])
            total = len(target_validations)
            summary["target_validations"] = {
                "total_targets": total,
                "passed_targets": passed,
                "failed_targets": total - passed,
                "pass_rate": passed / total if total > 0 else 0
            }
        
        # Regression status
        baseline_comparison = validation_results.get("baseline_comparison", {})
        if baseline_comparison and "overall_regression" in baseline_comparison:
            summary["regression_status"] = "regression" if baseline_comparison["overall_regression"] else "no_regression"
        
        # Calculate overall score
        score_components = []
        
        # Benchmark success rate
        if summary["total_benchmarks"] > 0:
            benchmark_score = summary["successful_benchmarks"] / summary["total_benchmarks"]
            score_components.append(benchmark_score * 40)  # 40% weight
        
        # Target validation score
        if summary["target_validations"]:
            target_score = summary["target_validations"]["pass_rate"]
            score_components.append(target_score * 40)  # 40% weight
        
        # Regression penalty
        regression_score = 1.0 if summary["regression_status"] != "regression" else 0.5
        score_components.append(regression_score * 20)  # 20% weight
        
        summary["overall_score"] = sum(score_components) / 100 if score_components else 0.0
        
        return summary
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on validation results."""
        recommendations = []
        
        # Check for failed benchmarks
        errors = validation_results.get("errors", [])
        if errors:
            recommendations.append(f"Address {len(errors)} benchmark failures before production deployment")
        
        # Check target validations
        target_validations = validation_results.get("validation_results", {}).get("target_validation", [])
        critical_failures = [v for v in target_validations if not v["passed"] and v["severity"] == "critical"]
        if critical_failures:
            recommendations.append(f"CRITICAL: {len(critical_failures)} critical performance targets failed")
        
        # Check regressions
        baseline_comparison = validation_results.get("baseline_comparison", {})
        if baseline_comparison.get("overall_regression", False):
            regressions = baseline_comparison.get("significant_regressions", [])
            recommendations.append(f"Performance regression detected in {len(regressions)} metrics")
        
        # Specific recommendations based on benchmark results
        benchmark_results = validation_results.get("benchmark_results", {})
        
        # Latency recommendations
        if "latency" in benchmark_results:
            latency_analysis = benchmark_results["latency"].get("analysis", {})
            if latency_analysis.get("recommendations"):
                recommendations.extend(latency_analysis["recommendations"])
        
        # Throughput recommendations
        if "throughput" in benchmark_results:
            throughput_analysis = benchmark_results["throughput"].get("analysis", {})
            if throughput_analysis.get("recommendations"):
                recommendations.extend(throughput_analysis["recommendations"])
        
        # Memory recommendations
        if "memory" in benchmark_results:
            memory_analysis = benchmark_results["memory"].get("analysis", {})
            if memory_analysis.get("recommendations"):
                recommendations.extend(memory_analysis["recommendations"])
        
        # Network recommendations
        if "network" in benchmark_results:
            network_analysis = benchmark_results["network"].get("analysis", {})
            if network_analysis.get("recommendations"):
                recommendations.extend(network_analysis["recommendations"])
        
        # Overall score recommendations
        summary = validation_results.get("summary", {})
        overall_score = summary.get("overall_score", 0)
        
        if overall_score < 0.6:
            recommendations.append("Overall performance score is below acceptable threshold - comprehensive optimization needed")
        elif overall_score < 0.8:
            recommendations.append("Performance has room for improvement - consider targeted optimizations")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_validation_report(self, validation_results: Dict[str, Any], output_file: Path):
        """Generate comprehensive validation report."""
        report_lines = [
            "# Performance Validation Report",
            f"Generated: {validation_results['validation_info']['start_time']}",
            f"Cluster: {validation_results['validation_info']['cluster_name']}",
            f"Model: {validation_results['validation_info']['model_name']}",
            ""
        ]
        
        # Executive Summary
        summary = validation_results.get("summary", {})
        if summary:
            report_lines.extend([
                "## Executive Summary",
                f"- **Overall Score**: {summary.get('overall_score', 0):.2f}/1.00",
                f"- **Benchmarks**: {summary.get('successful_benchmarks', 0)}/{summary.get('total_benchmarks', 0)} successful",
                f"- **Regression Status**: {summary.get('regression_status', 'unknown').replace('_', ' ').title()}",
                ""
            ])
            
            if summary.get("target_validations"):
                tv = summary["target_validations"]
                report_lines.extend([
                    f"- **Target Validation**: {tv['passed_targets']}/{tv['total_targets']} targets passed ({tv['pass_rate']:.1%})",
                    ""
                ])
        
        # Target Validation Results
        target_validations = validation_results.get("validation_results", {}).get("target_validation", [])
        if target_validations:
            report_lines.extend([
                "## Target Validation Results",
                ""
            ])
            
            for validation in target_validations:
                status = "✅ PASS" if validation["passed"] else "❌ FAIL"
                report_lines.extend([
                    f"### {validation['target']['metric_name']}",
                    f"**Status**: {status} ({validation['severity']})",
                    f"**Target**: {validation['target']['target_value']} ({validation['target']['comparison_type']})",
                    f"**Actual**: {validation['actual_value']:.2f}",
                    f"**Deviation**: {validation['deviation_percent']:.1f}%",
                    ""
                ])
        
        # Regression Testing Results
        baseline_comparison = validation_results.get("baseline_comparison", {})
        if baseline_comparison and "metric_comparisons" in baseline_comparison:
            report_lines.extend([
                "## Regression Testing Results",
                f"**Baseline**: {baseline_comparison.get('baseline_version', 'Unknown')} ({baseline_comparison.get('baseline_timestamp', 'Unknown')})",
                ""
            ])
            
            # Significant regressions
            regressions = baseline_comparison.get("significant_regressions", [])
            if regressions:
                report_lines.extend([
                    "### Significant Regressions",
                    ""
                ])
                for regression in regressions:
                    report_lines.append(f"- **{regression['metric']}**: {regression['change_percent']:.1f}% worse ({regression['baseline']:.2f} → {regression['current']:.2f})")
                report_lines.append("")
            
            # Improvements
            improvements = baseline_comparison.get("improvements", [])
            if improvements:
                report_lines.extend([
                    "### Performance Improvements",
                    ""
                ])
                for improvement in improvements:
                    report_lines.append(f"- **{improvement['metric']}**: {abs(improvement['change_percent']):.1f}% better ({improvement['baseline']:.2f} → {improvement['current']:.2f})")
                report_lines.append("")
        
        # Recommendations
        recommendations = validation_results.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Benchmark Summaries
        benchmark_results = validation_results.get("benchmark_results", {})
        if benchmark_results:
            report_lines.extend([
                "## Benchmark Results Summary",
                ""
            ])
            
            for benchmark_name, results in benchmark_results.items():
                summary_data = results.get("summary", {})
                if summary_data:
                    report_lines.extend([
                        f"### {benchmark_name.title()} Benchmark",
                        ""
                    ])
                    
                    # Add key metrics based on benchmark type
                    if benchmark_name == "latency":
                        report_lines.extend([
                            f"- **Average Latency**: {summary_data.get('overall_avg_latency_ms', 0):.1f} ms",
                            f"- **95th Percentile**: {summary_data.get('overall_p95_latency_ms', 0):.1f} ms",
                            f"- **Success Rate**: {summary_data.get('overall_success_rate', 0):.1%}",
                            ""
                        ])
                    elif benchmark_name == "throughput":
                        report_lines.extend([
                            f"- **Peak Requests/sec**: {summary_data.get('peak_requests_per_second', 0):.1f}",
                            f"- **Peak Tokens/sec**: {summary_data.get('peak_tokens_per_second', 0):.1f}",
                            f"- **Success Rate**: {summary_data.get('overall_success_rate', 0):.1%}",
                            ""
                        ])
                    elif benchmark_name == "memory":
                        report_lines.extend([
                            f"- **Peak Memory**: {summary_data.get('peak_memory_usage_mb', 0):.1f} MB",
                            f"- **Efficiency Grade**: {summary_data.get('memory_efficiency_grade', 'N/A')}",
                            f"- **Leak Score**: {summary_data.get('overall_leak_score', 0):.2f}",
                            ""
                        ])
                    elif benchmark_name == "network":
                        report_lines.extend([
                            f"- **Peak Bandwidth**: {summary_data.get('peak_bandwidth_mbps', 0):.1f} Mbps",
                            f"- **Efficiency Score**: {summary_data.get('network_efficiency_score', 0):.2f}",
                            f"- **Average Latency**: {summary_data.get('avg_latency_ms', 0):.1f} ms",
                            ""
                        ])
        
        # Errors
        errors = validation_results.get("errors", [])
        if errors:
            report_lines.extend([
                "## Errors Encountered",
                ""
            ])
            for error in errors:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Validation report saved to: {output_file}")


def create_default_performance_targets() -> List[PerformanceTarget]:
    """Create default performance targets for validation."""
    return [
        # Latency targets
        PerformanceTarget(
            metric_name="avg_latency_ms",
            target_value=100.0,  # 100ms average latency
            tolerance_percent=20.0,
            comparison_type="less_than",
            priority="critical"
        ),
        PerformanceTarget(
            metric_name="p95_latency_ms",
            target_value=200.0,  # 200ms 95th percentile
            tolerance_percent=15.0,
            comparison_type="less_than",
            priority="important"
        ),
        
        # Throughput targets
        PerformanceTarget(
            metric_name="peak_requests_per_second",
            target_value=50.0,  # 50 requests/sec
            tolerance_percent=10.0,
            comparison_type="greater_than",
            priority="important"
        ),
        
        # Memory targets
        PerformanceTarget(
            metric_name="peak_memory_usage_mb",
            target_value=8000.0,  # 8GB peak memory
            tolerance_percent=25.0,
            comparison_type="less_than",
            priority="critical"
        ),
        PerformanceTarget(
            metric_name="memory_leak_score",
            target_value=0.3,  # Low leak score
            tolerance_percent=50.0,
            comparison_type="less_than",
            priority="important"
        ),
        
        # Success rate targets
        PerformanceTarget(
            metric_name="latency_success_rate",
            target_value=0.95,  # 95% success rate
            tolerance_percent=5.0,
            comparison_type="greater_than",
            priority="critical"
        ),
        PerformanceTarget(
            metric_name="throughput_success_rate",
            target_value=0.95,  # 95% success rate
            tolerance_percent=5.0,
            comparison_type="greater_than",
            priority="critical"
        )
    ]


async def main():
    """Example usage of performance validator."""
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
    
    # Create performance targets
    targets = create_default_performance_targets()
    
    # Run validation
    validator = PerformanceValidator(config)
    results = await validator.run_full_validation(
        performance_targets=targets,
        create_baseline=True,
        regression_test=False
    )
    
    print("Performance validation completed!")
    print(f"Overall score: {results['summary']['overall_score']:.2f}")
    print(f"Recommendations: {len(results['recommendations'])}")


if __name__ == "__main__":
    asyncio.run(main())
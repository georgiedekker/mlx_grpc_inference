#!/usr/bin/env python3
"""
Complete Benchmark Suite Runner for Distributed MLX Inference System.

This script orchestrates all benchmarking components to provide a comprehensive
performance analysis and validation report for the distributed inference fixes.

Components included:
1. Comprehensive Benchmarking Framework
2. Automated Validation Scripts  
3. Performance Comparison Tools
4. Communication Overhead Analysis
5. Final Analysis Report Generation
"""

import asyncio
import time
import logging
import json
import sys
import os
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all benchmark components
try:
    from comprehensive_benchmark import ComprehensiveBenchmarkSuite
    from automated_validation import AutomatedValidationSuite
    from performance_comparison import ComprehensivePerformanceAnalysis
    from communication_benchmark import ComprehensiveCommunicationBenchmark
except ImportError as e:
    print(f"Error importing benchmark components: {e}")
    print("Make sure all benchmark files are in the same directory")
    sys.exit(1)

logger = logging.getLogger(__name__)


class FullBenchmarkSuite:
    """Orchestrates all benchmarking components."""
    
    def __init__(self, api_url: str = "http://localhost:8100", output_dir: str = "full_benchmark_results"):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all benchmark components
        self.comprehensive_benchmark = ComprehensiveBenchmarkSuite(api_url, str(self.output_dir / "comprehensive"))
        self.validation_suite = AutomatedValidationSuite(api_url, str(self.output_dir / "validation"))
        self.performance_analysis = ComprehensivePerformanceAnalysis(api_url, str(self.output_dir / "performance"))
        self.communication_benchmark = ComprehensiveCommunicationBenchmark(
            api_url.replace("http://", "").replace("https://", ""), 
            str(self.output_dir / "communication")
        )
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_complete_benchmark_suite(self, 
                                         skip_validation: bool = False,
                                         skip_communication: bool = False,
                                         iterations: Dict[str, int] = None) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        
        if iterations is None:
            iterations = {
                "comprehensive": 1000,
                "validation": 30,
                "performance": 30,
                "communication": 100
            }
        
        self.start_time = datetime.now()
        logger.info("="*80)
        logger.info("STARTING COMPLETE BENCHMARK SUITE")
        logger.info("="*80)
        logger.info(f"Start Time: {self.start_time}")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("")
        
        suite_results = {
            "suite_info": {
                "start_time": self.start_time.isoformat(),
                "api_url": self.api_url,
                "output_directory": str(self.output_dir),
                "iterations": iterations,
                "components_run": []
            },
            "results": {},
            "summary": {},
            "final_report": {},
            "errors": []
        }
        
        # 1. Run Automated Validation Suite (critical fixes validation)
        if not skip_validation:
            logger.info("🔍 PHASE 1: Running Automated Validation Suite")
            logger.info("-" * 60)
            try:
                validation_results = await self.validation_suite.run_all_validations()
                suite_results["results"]["validation"] = validation_results
                suite_results["suite_info"]["components_run"].append("validation")
                
                # Check for critical failures
                if validation_results.get("overall_status") == "FAIL":
                    logger.error("❌ Critical validation failures detected!")
                    for failure in validation_results.get("critical_failures", []):
                        logger.error(f"   - {failure}")
                    
                    # Ask user if they want to continue
                    response = input("\nCritical failures detected. Continue with benchmarks? (y/N): ")
                    if response.lower() != 'y':
                        logger.info("Benchmark suite stopped due to critical failures.")
                        return suite_results
                else:
                    logger.info("✅ All critical validations passed!")
                
            except Exception as e:
                error_msg = f"Validation suite failed: {e}"
                logger.error(error_msg)
                suite_results["errors"].append(error_msg)
        else:
            logger.info("⏭️  Skipping validation suite (--skip-validation)")
        
        # 2. Run Comprehensive Benchmark Suite (performance metrics)
        logger.info("\n📊 PHASE 2: Running Comprehensive Benchmark Suite")
        logger.info("-" * 60)
        try:
            comprehensive_results = await self.comprehensive_benchmark.run_all_benchmarks(
                tensor_iterations=iterations["comprehensive"],
                inference_iterations=iterations["comprehensive"] // 20,
                communication_iterations=iterations["comprehensive"] // 10,
                concurrent_requests=10,
                throughput_requests=100
            )
            suite_results["results"]["comprehensive"] = comprehensive_results
            suite_results["suite_info"]["components_run"].append("comprehensive")
            
        except Exception as e:
            error_msg = f"Comprehensive benchmark failed: {e}"
            logger.error(error_msg)
            suite_results["errors"].append(error_msg)
        
        # 3. Run Performance Comparison Analysis
        logger.info("\n⚖️  PHASE 3: Running Performance Comparison Analysis")
        logger.info("-" * 60)
        try:
            performance_results = await self.performance_analysis.run_comprehensive_analysis(
                iterations=iterations["performance"]
            )
            suite_results["results"]["performance"] = performance_results
            suite_results["suite_info"]["components_run"].append("performance")
            
        except Exception as e:
            error_msg = f"Performance comparison failed: {e}"
            logger.error(error_msg)
            suite_results["errors"].append(error_msg)
        
        # 4. Run Communication Overhead Benchmarks
        if not skip_communication:
            logger.info("\n🌐 PHASE 4: Running Communication Overhead Benchmarks")
            logger.info("-" * 60)
            try:
                communication_results = await self.communication_benchmark.run_all_benchmarks(
                    connection_iterations=iterations["communication"],
                    serialization_iterations=iterations["communication"] * 10,
                    network_iterations=iterations["communication"],
                    batch_iterations=iterations["communication"] // 2
                )
                suite_results["results"]["communication"] = communication_results
                suite_results["suite_info"]["components_run"].append("communication")
                
            except Exception as e:
                error_msg = f"Communication benchmark failed: {e}"
                logger.error(error_msg)
                suite_results["errors"].append(error_msg)
        else:
            logger.info("⏭️  Skipping communication benchmarks (--skip-communication)")
        
        # 5. Generate Final Analysis Report
        logger.info("\n📋 PHASE 5: Generating Final Analysis Report")
        logger.info("-" * 60)
        
        self.end_time = datetime.now()
        suite_results["suite_info"]["end_time"] = self.end_time.isoformat()
        suite_results["suite_info"]["total_duration_minutes"] = (self.end_time - self.start_time).total_seconds() / 60
        
        # Generate comprehensive summary
        suite_results["summary"] = self._generate_comprehensive_summary(suite_results)
        
        # Generate final recommendations report
        suite_results["final_report"] = self._generate_final_report(suite_results)
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"complete_benchmark_suite_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        logger.info(f"Complete benchmark results saved to: {results_file}")
        
        # Generate executive summary report
        executive_report_file = self.output_dir / f"executive_summary_{timestamp}.md"
        self._generate_executive_summary(suite_results, executive_report_file)
        
        return suite_results
    
    def _generate_comprehensive_summary(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary across all benchmark components."""
        summary = {
            "overall_status": "UNKNOWN",
            "key_metrics": {},
            "performance_improvements": {},
            "critical_issues": [],
            "recommendations": [],
            "component_summaries": {}
        }
        
        # Determine overall status
        has_critical_failures = False
        has_errors = len(suite_results.get("errors", [])) > 0
        
        if "validation" in suite_results.get("results", {}):
            validation_status = suite_results["results"]["validation"].get("overall_status")
            if validation_status == "FAIL":
                has_critical_failures = True
                summary["critical_issues"].extend(
                    suite_results["results"]["validation"].get("critical_failures", [])
                )
        
        if has_critical_failures:
            summary["overall_status"] = "CRITICAL_FAILURES"
        elif has_errors:
            summary["overall_status"] = "PARTIAL_SUCCESS"
        else:
            summary["overall_status"] = "SUCCESS"
        
        # Extract key metrics from each component
        
        # Validation metrics
        if "validation" in suite_results.get("results", {}):
            validation_data = suite_results["results"]["validation"]
            if "summary" in validation_data:
                val_summary = validation_data["summary"]
                summary["component_summaries"]["validation"] = {
                    "total_tests": val_summary.get("total_tests", 0),
                    "success_rate": val_summary.get("success_rate", 0),
                    "critical_fixes_validated": val_summary.get("critical_fixes", 0)
                }
        
        # Comprehensive benchmark metrics
        if "comprehensive" in suite_results.get("results", {}):
            comp_data = suite_results["results"]["comprehensive"]
            if "summary" in comp_data:
                comp_summary = comp_data["summary"]
                summary["component_summaries"]["comprehensive"] = {
                    "total_benchmarks": comp_summary.get("total_benchmarks", 0),
                    "overall_success_rate": comp_summary.get("overall_success_rate", 0)
                }
                
                # Extract key performance metrics
                if "benchmarks_by_type" in comp_summary:
                    for bench_type, bench_stats in comp_summary["benchmarks_by_type"].items():
                        summary["key_metrics"][f"{bench_type}_latency_ms"] = bench_stats.get("avg_latency_ms", 0)
                        summary["key_metrics"][f"{bench_type}_throughput"] = bench_stats.get("avg_throughput", 0)
        
        # Performance comparison metrics
        if "performance" in suite_results.get("results", {}):
            perf_data = suite_results["results"]["performance"]
            if "comparisons" in perf_data:
                for comparison_key, comparison_data in perf_data["comparisons"].items():
                    improvement_key = f"{comparison_key}_throughput_improvement"
                    summary["performance_improvements"][improvement_key] = comparison_data.get("throughput_improvement_factor", 1.0)
        
        # Communication benchmark metrics
        if "communication" in suite_results.get("results", {}):
            comm_data = suite_results["results"]["communication"]
            if "summary" in comm_data:
                comm_summary = comm_data["summary"]
                summary["component_summaries"]["communication"] = {
                    "total_tests": comm_summary.get("total_tests", 0),
                    "avg_latency_ms": comm_summary.get("overall_performance", {}).get("avg_latency_ms", 0),
                    "avg_throughput_ops_sec": comm_summary.get("overall_performance", {}).get("avg_throughput_ops_sec", 0)
                }
        
        # Generate high-level recommendations
        summary["recommendations"] = self._generate_high_level_recommendations(suite_results)
        
        return summary
    
    def _generate_high_level_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate high-level recommendations based on all benchmark results."""
        recommendations = []
        
        # Validation-based recommendations
        if "validation" in suite_results.get("results", {}):
            validation_data = suite_results["results"]["validation"]
            if validation_data.get("overall_status") == "FAIL":
                recommendations.append("🚨 CRITICAL: Address validation failures before production deployment")
            
            # Check specific validation results
            if "fix_status" in validation_data:
                for fix_name, fix_info in validation_data["fix_status"].items():
                    if fix_info.get("status") == "FAILED" and fix_info.get("critical", False):
                        recommendations.append(f"🔧 Fix critical issue: {fix_name}")
        
        # Performance-based recommendations
        if "performance" in suite_results.get("results", {}):
            perf_data = suite_results["results"]["performance"]
            if "recommendations" in perf_data:
                perf_recs = perf_data["recommendations"]
                
                # Add deployment recommendations
                for rec in perf_recs.get("deployment_recommendations", []):
                    recommendations.append(f"📈 {rec}")
                
                # Add optimization opportunities
                for opp in perf_recs.get("optimization_opportunities", []):
                    recommendations.append(f"⚡ {opp}")
        
        # Communication-based recommendations
        if "communication" in suite_results.get("results", {}):
            comm_data = suite_results["results"]["communication"]
            if "analysis" in comm_data and "recommendations" in comm_data["analysis"]:
                for rec in comm_data["analysis"]["recommendations"]:
                    recommendations.append(f"🌐 {rec}")
        
        # Comprehensive benchmark recommendations
        if "comprehensive" in suite_results.get("results", {}):
            comp_data = suite_results["results"]["comprehensive"]
            
            # Check for low success rates
            if "summary" in comp_data:
                success_rate = comp_data["summary"].get("overall_success_rate", 0)
                if success_rate < 0.95:
                    recommendations.append(f"🔍 Investigate reliability issues (success rate: {success_rate:.1%})")
        
        # General recommendations if no specific issues found
        if not recommendations:
            recommendations.append("✅ System performing well - consider production deployment")
            recommendations.append("📊 Monitor performance metrics in production environment")
            recommendations.append("🔄 Schedule regular performance regression testing")
        
        return recommendations
    
    def _generate_final_report(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive analysis report."""
        
        report = {
            "executive_summary": {},
            "detailed_analysis": {},
            "deployment_readiness": {},
            "next_steps": []
        }
        
        # Executive Summary
        summary = suite_results.get("summary", {})
        report["executive_summary"] = {
            "overall_status": summary.get("overall_status", "UNKNOWN"),
            "total_tests_run": sum(
                comp.get("total_tests", 0) 
                for comp in summary.get("component_summaries", {}).values()
                if isinstance(comp, dict)
            ),
            "critical_issues_count": len(summary.get("critical_issues", [])),
            "key_performance_improvements": summary.get("performance_improvements", {}),
            "primary_recommendations": summary.get("recommendations", [])[:5]  # Top 5
        }
        
        # Detailed Analysis
        report["detailed_analysis"] = {
            "validation_results": self._analyze_validation_results(suite_results),
            "performance_analysis": self._analyze_performance_results(suite_results),
            "communication_analysis": self._analyze_communication_results(suite_results),
            "reliability_assessment": self._assess_reliability(suite_results)
        }
        
        # Deployment Readiness Assessment
        report["deployment_readiness"] = self._assess_deployment_readiness(suite_results)
        
        # Next Steps
        report["next_steps"] = self._generate_next_steps(suite_results)
        
        return report
    
    def _analyze_validation_results(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze validation results."""
        if "validation" not in suite_results.get("results", {}):
            return {"status": "not_run", "message": "Validation suite was not executed"}
        
        validation_data = suite_results["results"]["validation"]
        
        analysis = {
            "overall_status": validation_data.get("overall_status", "UNKNOWN"),
            "critical_fixes_status": {},
            "test_results": {},
            "key_findings": []
        }
        
        # Analyze fix status
        if "fix_status" in validation_data:
            for fix_name, fix_info in validation_data["fix_status"].items():
                analysis["critical_fixes_status"][fix_name] = {
                    "status": fix_info.get("status"),
                    "critical": fix_info.get("critical", False),
                    "message": fix_info.get("message")
                }
                
                if fix_info.get("status") == "VALIDATED" and fix_info.get("critical"):
                    analysis["key_findings"].append(f"✅ Critical fix validated: {fix_name}")
                elif fix_info.get("status") == "FAILED" and fix_info.get("critical"):
                    analysis["key_findings"].append(f"❌ Critical fix failed: {fix_name}")
        
        return analysis
    
    def _analyze_performance_results(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance results."""
        if "performance" not in suite_results.get("results", {}):
            return {"status": "not_run", "message": "Performance analysis was not executed"}
        
        perf_data = suite_results["results"]["performance"]
        
        analysis = {
            "distributed_vs_single_device": {},
            "scaling_efficiency": {},
            "resource_utilization": {},
            "key_findings": []
        }
        
        # Analyze comparisons
        if "comparisons" in perf_data:
            for comparison_key, comparison_data in perf_data["comparisons"].items():
                throughput_improvement = comparison_data.get("throughput_improvement_factor", 1.0)
                scaling_efficiency = comparison_data.get("scaling_efficiency", 0.0)
                
                analysis["distributed_vs_single_device"][comparison_key] = {
                    "throughput_improvement": throughput_improvement,
                    "latency_improvement": comparison_data.get("latency_improvement_factor", 1.0),
                    "scaling_efficiency": scaling_efficiency,
                    "cost_effectiveness": comparison_data.get("cost_effectiveness", 1.0)
                }
                
                if throughput_improvement > 2.0:
                    analysis["key_findings"].append(f"🚀 Excellent throughput improvement: {throughput_improvement:.1f}x")
                elif throughput_improvement > 1.5:
                    analysis["key_findings"].append(f"📈 Good throughput improvement: {throughput_improvement:.1f}x")
                else:
                    analysis["key_findings"].append(f"⚠️ Limited throughput improvement: {throughput_improvement:.1f}x")
        
        return analysis
    
    def _analyze_communication_results(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication results."""
        if "communication" not in suite_results.get("results", {}):
            return {"status": "not_run", "message": "Communication benchmarks were not executed"}
        
        comm_data = suite_results["results"]["communication"]
        
        analysis = {
            "grpc_overhead": {},
            "serialization_performance": {},
            "network_efficiency": {},
            "key_findings": []
        }
        
        # Analyze gRPC overhead
        if "analysis" in comm_data and "grpc_overhead" in comm_data["analysis"]:
            overhead_data = comm_data["analysis"]["grpc_overhead"]
            overhead_pct = overhead_data.get("overhead_percentage", 0)
            
            analysis["grpc_overhead"] = {
                "overhead_percentage": overhead_pct,
                "category": overhead_data.get("overhead_category", "UNKNOWN"),
                "efficiency_score": overhead_data.get("efficiency_score", 0)
            }
            
            if overhead_pct < 15:
                analysis["key_findings"].append(f"✅ Low gRPC overhead: {overhead_pct:.1f}%")
            elif overhead_pct < 30:
                analysis["key_findings"].append(f"⚠️ Moderate gRPC overhead: {overhead_pct:.1f}%")
            else:
                analysis["key_findings"].append(f"🚨 High gRPC overhead: {overhead_pct:.1f}%")
        
        return analysis
    
    def _assess_reliability(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system reliability."""
        reliability = {
            "overall_score": 0.0,
            "success_rates": {},
            "error_analysis": {},
            "stability_assessment": "UNKNOWN"
        }
        
        success_rates = []
        total_errors = 0
        
        # Collect success rates from all components
        for component_name, component_data in suite_results.get("results", {}).items():
            if "summary" in component_data:
                component_summary = component_data["summary"]
                
                # Different components store success rates differently
                if "overall_success_rate" in component_summary:
                    success_rate = component_summary["overall_success_rate"]
                elif "success_rate" in component_summary:
                    success_rate = component_summary["success_rate"]
                else:
                    success_rate = None
                
                if success_rate is not None:
                    success_rates.append(success_rate)
                    reliability["success_rates"][component_name] = success_rate
            
            # Count errors
            if "errors" in component_data:
                total_errors += len(component_data["errors"])
        
        # Calculate overall reliability score
        if success_rates:
            avg_success_rate = sum(success_rates) / len(success_rates)
            reliability["overall_score"] = avg_success_rate
            
            # Assess stability
            if avg_success_rate >= 0.99:
                reliability["stability_assessment"] = "EXCELLENT"
            elif avg_success_rate >= 0.95:
                reliability["stability_assessment"] = "GOOD"
            elif avg_success_rate >= 0.90:
                reliability["stability_assessment"] = "ACCEPTABLE"
            else:
                reliability["stability_assessment"] = "POOR"
        
        reliability["error_analysis"]["total_errors"] = total_errors
        reliability["error_analysis"]["errors_by_component"] = {}
        
        for component_name, component_data in suite_results.get("results", {}).items():
            if "errors" in component_data:
                reliability["error_analysis"]["errors_by_component"][component_name] = len(component_data["errors"])
        
        return reliability
    
    def _assess_deployment_readiness(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for production deployment."""
        readiness = {
            "overall_readiness": "NOT_READY",
            "readiness_score": 0.0,
            "blocking_issues": [],
            "warnings": [],
            "green_lights": [],
            "deployment_recommendation": ""
        }
        
        score_components = []
        
        # Validation readiness (40% of score)
        validation_ready = True
        if "validation" in suite_results.get("results", {}):
            validation_data = suite_results["results"]["validation"]
            if validation_data.get("overall_status") == "FAIL":
                validation_ready = False
                readiness["blocking_issues"].append("Critical validation failures")
            
            critical_failures = validation_data.get("critical_failures", [])
            if critical_failures:
                validation_ready = False
                for failure in critical_failures:
                    readiness["blocking_issues"].append(f"Critical fix failed: {failure}")
        
        score_components.append(0.4 if validation_ready else 0.0)
        if validation_ready:
            readiness["green_lights"].append("All critical fixes validated")
        
        # Performance readiness (30% of score)
        performance_ready = True
        if "performance" in suite_results.get("results", {}):
            perf_data = suite_results["results"]["performance"]
            
            # Check for performance improvements
            if "comparisons" in perf_data:
                improvements = []
                for comparison_key, comparison_data in perf_data["comparisons"].items():
                    throughput_improvement = comparison_data.get("throughput_improvement_factor", 1.0)
                    improvements.append(throughput_improvement)
                
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    if avg_improvement < 1.1:
                        readiness["warnings"].append("Limited performance improvement from distribution")
                    elif avg_improvement > 1.5:
                        readiness["green_lights"].append(f"Good performance improvement: {avg_improvement:.1f}x")
        
        score_components.append(0.3 if performance_ready else 0.15)
        
        # Reliability readiness (20% of score)
        reliability_data = self._assess_reliability(suite_results)
        reliability_score = reliability_data.get("overall_score", 0.0)
        
        if reliability_score >= 0.95:
            score_components.append(0.2)
            readiness["green_lights"].append(f"High reliability: {reliability_score:.1%}")
        elif reliability_score >= 0.90:
            score_components.append(0.15)
            readiness["warnings"].append(f"Moderate reliability: {reliability_score:.1%}")
        else:
            score_components.append(0.0)
            readiness["blocking_issues"].append(f"Low reliability: {reliability_score:.1%}")
        
        # Communication efficiency (10% of score)
        comm_ready = True
        if "communication" in suite_results.get("results", {}):
            comm_data = suite_results["results"]["communication"]
            if "analysis" in comm_data and "grpc_overhead" in comm_data["analysis"]:
                overhead_pct = comm_data["analysis"]["grpc_overhead"].get("overhead_percentage", 0)
                if overhead_pct > 50:
                    comm_ready = False
                    readiness["blocking_issues"].append(f"Excessive gRPC overhead: {overhead_pct:.1f}%")
                elif overhead_pct > 25:
                    readiness["warnings"].append(f"High gRPC overhead: {overhead_pct:.1f}%")
        
        score_components.append(0.1 if comm_ready else 0.05)
        
        # Calculate overall readiness score
        readiness["readiness_score"] = sum(score_components)
        
        # Determine overall readiness
        if readiness["blocking_issues"]:
            readiness["overall_readiness"] = "NOT_READY"
            readiness["deployment_recommendation"] = "Address blocking issues before deployment"
        elif readiness["readiness_score"] >= 0.8:
            readiness["overall_readiness"] = "READY"
            readiness["deployment_recommendation"] = "System ready for production deployment"
        elif readiness["readiness_score"] >= 0.6:
            readiness["overall_readiness"] = "READY_WITH_CAUTION"
            readiness["deployment_recommendation"] = "Deploy with monitoring and fallback plan"
        else:
            readiness["overall_readiness"] = "NOT_READY"
            readiness["deployment_recommendation"] = "Further optimization needed before deployment"
        
        return readiness
    
    def _generate_next_steps(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Based on deployment readiness
        deployment_readiness = self._assess_deployment_readiness(suite_results)
        
        if deployment_readiness["blocking_issues"]:
            next_steps.append("🚨 IMMEDIATE: Address all blocking issues")
            for issue in deployment_readiness["blocking_issues"]:
                next_steps.append(f"   - {issue}")
        
        if deployment_readiness["warnings"]:
            next_steps.append("⚠️ RECOMMENDED: Address warnings for optimal performance")
            for warning in deployment_readiness["warnings"]:
                next_steps.append(f"   - {warning}")
        
        # General next steps based on results
        overall_status = suite_results.get("summary", {}).get("overall_status", "UNKNOWN")
        
        if overall_status == "SUCCESS":
            next_steps.extend([
                "✅ Consider production deployment",
                "📊 Set up production monitoring dashboards",
                "🔄 Schedule regular performance regression tests",
                "📚 Document deployment procedures and monitoring guidelines"
            ])
        elif overall_status == "PARTIAL_SUCCESS":
            next_steps.extend([
                "🔍 Investigate and resolve partial failures",
                "📈 Optimize performance bottlenecks identified",
                "🧪 Run additional focused tests on problem areas"
            ])
        elif overall_status == "CRITICAL_FAILURES":
            next_steps.extend([
                "🔧 Fix all critical validation failures",
                "🧪 Re-run validation suite after fixes",
                "📋 Review system architecture for fundamental issues"
            ])
        
        # Add component-specific next steps
        if "performance" in suite_results.get("results", {}):
            perf_data = suite_results["results"]["performance"]
            if "recommendations" in perf_data:
                next_steps.append("📈 Performance optimizations:")
                for rec in perf_data["recommendations"].get("optimization_opportunities", [])[:3]:
                    next_steps.append(f"   - {rec}")
        
        return next_steps
    
    def _generate_executive_summary(self, suite_results: Dict[str, Any], output_file: Path):
        """Generate executive summary report."""
        
        summary = suite_results.get("summary", {})
        final_report = suite_results.get("final_report", {})
        
        report_lines = [
            "# MLX Distributed Inference System - Executive Benchmark Report",
            "",
            f"**Generated:** {suite_results['suite_info']['start_time']}",
            f"**Duration:** {suite_results['suite_info'].get('total_duration_minutes', 0):.1f} minutes",
            f"**API Endpoint:** {suite_results['suite_info']['api_url']}",
            "",
            "## Executive Summary",
            "",
            f"**Overall Status:** {summary.get('overall_status', 'UNKNOWN')}",
            f"**Tests Executed:** {final_report.get('executive_summary', {}).get('total_tests_run', 0)}",
            f"**Critical Issues:** {final_report.get('executive_summary', {}).get('critical_issues_count', 0)}",
            ""
        ]
        
        # Deployment Readiness
        if "deployment_readiness" in final_report:
            deployment = final_report["deployment_readiness"]
            report_lines.extend([
                "## 🚀 Deployment Readiness Assessment",
                "",
                f"**Overall Readiness:** {deployment.get('overall_readiness', 'UNKNOWN')}",
                f"**Readiness Score:** {deployment.get('readiness_score', 0):.1%}",
                f"**Recommendation:** {deployment.get('deployment_recommendation', 'N/A')}",
                ""
            ])
            
            if deployment.get("blocking_issues"):
                report_lines.extend([
                    "### 🚨 Blocking Issues",
                    ""
                ])
                for issue in deployment["blocking_issues"]:
                    report_lines.append(f"- {issue}")
                report_lines.append("")
            
            if deployment.get("green_lights"):
                report_lines.extend([
                    "### ✅ System Strengths",
                    ""
                ])
                for strength in deployment["green_lights"]:
                    report_lines.append(f"- {strength}")
                report_lines.append("")
        
        # Key Performance Improvements
        if summary.get("performance_improvements"):
            report_lines.extend([
                "## 📈 Key Performance Improvements",
                ""
            ])
            for improvement_key, improvement_value in summary["performance_improvements"].items():
                readable_key = improvement_key.replace("_", " ").title()
                report_lines.append(f"- **{readable_key}:** {improvement_value:.2f}x")
            report_lines.append("")
        
        # Top Recommendations
        if summary.get("recommendations"):
            report_lines.extend([
                "## 🎯 Top Recommendations",
                ""
            ])
            for i, rec in enumerate(summary["recommendations"][:5], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Next Steps
        if "next_steps" in final_report:
            report_lines.extend([
                "## 📋 Immediate Next Steps",
                ""
            ])
            for step in final_report["next_steps"][:10]:  # Top 10 next steps
                report_lines.append(f"- {step}")
            report_lines.append("")
        
        # Component Results Summary
        if summary.get("component_summaries"):
            report_lines.extend([
                "## 🧪 Component Test Results",
                ""
            ])
            for component, component_summary in summary["component_summaries"].items():
                report_lines.append(f"### {component.title()}")
                if isinstance(component_summary, dict):
                    for key, value in component_summary.items():
                        readable_key = key.replace("_", " ").title()
                        if isinstance(value, float):
                            if "rate" in key.lower():
                                report_lines.append(f"- **{readable_key}:** {value:.1%}")
                            else:
                                report_lines.append(f"- **{readable_key}:** {value:.2f}")
                        else:
                            report_lines.append(f"- **{readable_key}:** {value}")
                report_lines.append("")
        
        # Errors and Issues
        if suite_results.get("errors"):
            report_lines.extend([
                "## ⚠️ Issues Encountered",
                ""
            ])
            for error in suite_results["errors"]:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---",
            "",
            f"*Report generated by MLX Distributed Inference Benchmark Suite*",
            f"*Detailed results available in: {self.output_dir}/*"
        ])
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Executive summary saved to: {output_file}")


async def main():
    """Main entry point for the full benchmark suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark_suite.log')
        ]
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Complete MLX Distributed Inference Benchmark Suite")
    parser.add_argument("--api-url", default="http://localhost:8100", help="API server URL")
    parser.add_argument("--output-dir", default="full_benchmark_results", help="Output directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    parser.add_argument("--skip-communication", action="store_true", help="Skip communication tests")
    parser.add_argument("--quick", action="store_true", help="Run with reduced iterations for quick testing")
    parser.add_argument("--iterations-comprehensive", type=int, default=1000, help="Comprehensive benchmark iterations")
    parser.add_argument("--iterations-validation", type=int, default=30, help="Validation test iterations")
    parser.add_argument("--iterations-performance", type=int, default=30, help="Performance comparison iterations")
    parser.add_argument("--iterations-communication", type=int, default=100, help="Communication benchmark iterations")
    
    args = parser.parse_args()
    
    # Adjust iterations for quick mode
    if args.quick:
        iterations = {
            "comprehensive": 100,
            "validation": 10,
            "performance": 10,
            "communication": 20
        }
        logger.info("Running in quick mode with reduced iterations")
    else:
        iterations = {
            "comprehensive": args.iterations_comprehensive,
            "validation": args.iterations_validation,
            "performance": args.iterations_performance,
            "communication": args.iterations_communication
        }
    
    # Initialize and run benchmark suite
    benchmark_suite = FullBenchmarkSuite(
        api_url=args.api_url,
        output_dir=args.output_dir
    )
    
    try:
        results = await benchmark_suite.run_complete_benchmark_suite(
            skip_validation=args.skip_validation,
            skip_communication=args.skip_communication,
            iterations=iterations
        )
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("COMPLETE BENCHMARK SUITE FINISHED")
        logger.info("="*80)
        
        if "summary" in results:
            summary = results["summary"]
            logger.info(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
            
            # Print deployment readiness
            if "final_report" in results and "deployment_readiness" in results["final_report"]:
                deployment = results["final_report"]["deployment_readiness"]
                logger.info(f"Deployment Readiness: {deployment.get('overall_readiness', 'UNKNOWN')}")
                logger.info(f"Readiness Score: {deployment.get('readiness_score', 0):.1%}")
            
            # Print top recommendations
            if summary.get("recommendations"):
                logger.info("\nTop Recommendations:")
                for i, rec in enumerate(summary["recommendations"][:3], 1):
                    logger.info(f"  {i}. {rec}")
        
        logger.info(f"\nComplete results saved to: {args.output_dir}/")
        logger.info("Check executive_summary_*.md for a high-level overview")
        
        # Exit with appropriate code
        overall_status = results.get("summary", {}).get("overall_status", "UNKNOWN")
        if overall_status == "CRITICAL_FAILURES":
            return 2  # Critical failures
        elif overall_status == "PARTIAL_SUCCESS" or results.get("errors"):
            return 1  # Partial success or errors
        else:
            return 0  # Success
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark suite interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark suite failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
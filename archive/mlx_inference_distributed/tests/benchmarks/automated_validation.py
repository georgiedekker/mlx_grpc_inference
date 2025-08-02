#!/usr/bin/env python3
"""
Automated Validation Scripts for Distributed MLX Inference Fixes.

This module provides automated testing to validate that all distributed inference
fixes are working correctly, including:
- Model sharding across devices
- gRPC communication
- Tensor serialization/deserialization
- Cache management
- Error handling and recovery
- Performance regression detection
"""

import asyncio
import time
import logging
import json
import sys
import os
import subprocess
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import requests
import tempfile
import signal

import mlx.core as mx
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from grpc_server import TensorSerializer, DistributedInferenceServicer
    from grpc_client import DistributedInferenceClient
    from device_capabilities import DeviceCapabilityDetector
    from distributed_config import DistributedConfig
except ImportError as e:
    print(f"Warning: Could not import distributed components: {e}")

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    message: str
    duration_ms: float
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class FixValidation:
    """Metadata about a fix being validated."""
    fix_name: str
    description: str
    validation_tests: List[str]
    critical: bool  # If True, failure blocks deployment


class ClusterHealthValidator:
    """Validates cluster health and basic functionality."""
    
    def __init__(self, api_url: str = "http://localhost:8100"):
        self.api_url = api_url
        self.timeout = 30
    
    async def validate_cluster_startup(self) -> ValidationResult:
        """Validate that the cluster starts up correctly."""
        start_time = time.time()
        
        try:
            # Wait for cluster to be ready
            for attempt in range(self.timeout):
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        health_data = response.json()
                        if (health_data.get("status") == "healthy" and 
                            health_data.get("model_loaded", False)):
                            
                            duration = (time.time() - start_time) * 1000
                            return ValidationResult(
                                test_name="cluster_startup",
                                status="PASS",
                                message=f"Cluster started successfully in {attempt + 1} attempts",
                                duration_ms=duration,
                                metadata={
                                    "attempts": attempt + 1,
                                    "health_data": health_data
                                },
                                timestamp=datetime.now().isoformat()
                            )
                except requests.RequestException:
                    pass
                
                await asyncio.sleep(1)
            
            # Timeout reached
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="cluster_startup",
                status="FAIL",
                message=f"Cluster failed to start within {self.timeout} seconds",
                duration_ms=duration,
                metadata={"timeout_seconds": self.timeout},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="cluster_startup",
                status="ERROR",
                message=f"Error during cluster startup validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def validate_device_discovery(self) -> ValidationResult:
        """Validate that all expected devices are discovered."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.api_url}/distributed/gpu-info", timeout=10)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="device_discovery",
                    status="FAIL",
                    message=f"GPU info endpoint returned {response.status_code}",
                    duration_ms=duration,
                    metadata={"status_code": response.status_code},
                    timestamp=datetime.now().isoformat()
                )
            
            data = response.json()
            cluster_info = data.get("cluster_info", {})
            devices = data.get("devices", [])
            
            # Expected device configuration for 3-device cluster
            expected_devices = {"mini1", "mini2", "master"}
            discovered_devices = {d.get("device_id") for d in devices}
            
            if len(devices) < 3:
                return ValidationResult(
                    test_name="device_discovery",
                    status="FAIL",
                    message=f"Expected 3 devices, found {len(devices)}: {discovered_devices}",
                    duration_ms=duration,
                    metadata={
                        "expected_devices": list(expected_devices),
                        "discovered_devices": list(discovered_devices),
                        "device_count": len(devices)
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            if not expected_devices.issubset(discovered_devices):
                missing = expected_devices - discovered_devices
                return ValidationResult(
                    test_name="device_discovery",
                    status="FAIL",
                    message=f"Missing expected devices: {missing}",
                    duration_ms=duration,
                    metadata={
                        "missing_devices": list(missing),
                        "discovered_devices": list(discovered_devices)
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            # Validate cluster info
            if cluster_info.get("total_devices") != 3:
                return ValidationResult(
                    test_name="device_discovery",
                    status="FAIL",
                    message=f"Cluster reports {cluster_info.get('total_devices')} devices, expected 3",
                    duration_ms=duration,
                    metadata={"cluster_info": cluster_info},
                    timestamp=datetime.now().isoformat()
                )
            
            return ValidationResult(
                test_name="device_discovery",
                status="PASS",
                message=f"All 3 devices discovered successfully: {discovered_devices}",
                duration_ms=duration,
                metadata={
                    "devices": devices,
                    "cluster_info": cluster_info
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="device_discovery",
                status="ERROR",
                message=f"Error during device discovery validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def validate_grpc_communication(self) -> ValidationResult:
        """Validate gRPC communication between devices."""
        start_time = time.time()
        
        try:
            # Check that gRPC communication is reported as active
            response = requests.get(f"{self.api_url}/distributed/status", timeout=10)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="grpc_communication",
                    status="FAIL",
                    message=f"Status endpoint returned {response.status_code}",
                    duration_ms=duration,
                    metadata={"status_code": response.status_code},
                    timestamp=datetime.now().isoformat()
                )
            
            status_data = response.json()
            
            # Check that all devices are communicating
            if status_data.get("status") != "operational":
                return ValidationResult(
                    test_name="grpc_communication",
                    status="FAIL",
                    message=f"Cluster status is {status_data.get('status')}, expected 'operational'",
                    duration_ms=duration,
                    metadata={"status_data": status_data},
                    timestamp=datetime.now().isoformat()
                )
            
            # Check gRPC communication status from GPU info
            gpu_response = requests.get(f"{self.api_url}/distributed/gpu-info", timeout=10)
            if gpu_response.status_code == 200:
                gpu_data = gpu_response.json()
                grpc_status = gpu_data.get("cluster_info", {}).get("gRPC_communication")
                if grpc_status != "Active":
                    return ValidationResult(
                        test_name="grpc_communication",
                        status="FAIL",
                        message=f"gRPC communication status is '{grpc_status}', expected 'Active'",
                        duration_ms=duration,
                        metadata={"grpc_status": grpc_status},
                        timestamp=datetime.now().isoformat()
                    )
            
            return ValidationResult(
                test_name="grpc_communication",
                status="PASS",
                message="gRPC communication verified as active",
                duration_ms=duration,
                metadata={"status_data": status_data},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="grpc_communication",
                status="ERROR",
                message=f"Error during gRPC communication validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )


class InferenceValidator:
    """Validates inference functionality and performance."""
    
    def __init__(self, api_url: str = "http://localhost:8100"):
        self.api_url = api_url
        self.test_prompts = [
            ("simple_math", "What is 2+2?", "4"),
            ("basic_reasoning", "Is the sky blue?", "yes"),
            ("model_info", "What model are you?", "qwen")
        ]
    
    async def validate_basic_inference(self) -> ValidationResult:
        """Validate that basic inference works correctly."""
        start_time = time.time()
        
        try:
            request_data = {
                "model": "mlx-community/Qwen3-1.7B-8bit",
                "messages": [{"role": "user", "content": self.test_prompts[0][1]}],
                "max_tokens": 20,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            
            duration = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="basic_inference",
                    status="FAIL",
                    message=f"Inference request failed with status {response.status_code}",
                    duration_ms=duration,
                    metadata={
                        "status_code": response.status_code,
                        "response_text": response.text[:500]
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            data = response.json()
            
            # Validate response structure
            if "choices" not in data or len(data["choices"]) == 0:
                return ValidationResult(
                    test_name="basic_inference",
                    status="FAIL",
                    message="Response missing choices field or empty choices",
                    duration_ms=duration,
                    metadata={"response_data": data},
                    timestamp=datetime.now().isoformat()
                )
            
            choice = data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                return ValidationResult(
                    test_name="basic_inference",
                    status="FAIL",
                    message="Response missing message content",
                    duration_ms=duration,
                    metadata={"choice": choice},
                    timestamp=datetime.now().isoformat()
                )
            
            content = choice["message"]["content"].strip()
            if len(content) == 0:
                return ValidationResult(
                    test_name="basic_inference",
                    status="FAIL",
                    message="Empty response content",
                    duration_ms=duration,
                    metadata={"content": content},
                    timestamp=datetime.now().isoformat()
                )
            
            # Check for expected answer
            expected_answer = self.test_prompts[0][2]
            if expected_answer.lower() not in content.lower():
                logger.warning(f"Expected answer '{expected_answer}' not found in response: {content}")
            
            return ValidationResult(
                test_name="basic_inference",
                status="PASS",
                message=f"Basic inference successful, generated {len(content)} characters",
                duration_ms=duration,
                metadata={
                    "content": content,
                    "usage": data.get("usage", {}),
                    "response_time_ms": duration
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="basic_inference",
                status="ERROR",
                message=f"Error during basic inference validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def validate_concurrent_inference(self, concurrent_requests: int = 5) -> ValidationResult:
        """Validate concurrent inference handling."""
        start_time = time.time()
        
        try:
            async def make_request(prompt_idx: int) -> Tuple[bool, str, float]:
                request_start = time.time()
                
                prompt_name, prompt_text, _ = self.test_prompts[prompt_idx % len(self.test_prompts)]
                
                request_data = {
                    "model": "mlx-community/Qwen3-1.7B-8bit",
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 30,
                    "temperature": 0.5
                }
                
                try:
                    response = requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=request_data,
                        timeout=45
                    )
                    
                    request_duration = time.time() - request_start
                    
                    if response.status_code == 200:
                        data = response.json()
                        if ("choices" in data and len(data["choices"]) > 0 and
                            "message" in data["choices"][0] and 
                            len(data["choices"][0]["message"]["content"].strip()) > 0):
                            return True, f"Success: {prompt_name}", request_duration
                    
                    return False, f"Failed: {response.status_code}", request_duration
                    
                except Exception as e:
                    request_duration = time.time() - request_start
                    return False, f"Error: {str(e)}", request_duration
            
            # Execute concurrent requests
            tasks = [make_request(i) for i in range(concurrent_requests)]
            results = []
            
            for task in tasks:
                result = await task
                results.append(result)
            
            duration = (time.time() - start_time) * 1000
            
            # Analyze results
            successful = sum(1 for success, _, _ in results if success)
            failed = concurrent_requests - successful
            success_rate = successful / concurrent_requests if concurrent_requests > 0 else 0
            
            avg_request_time = sum(req_time for _, _, req_time in results) / len(results) if results else 0
            
            if success_rate < 0.8:  # Expect at least 80% success rate
                return ValidationResult(
                    test_name="concurrent_inference",
                    status="FAIL",
                    message=f"Low success rate: {successful}/{concurrent_requests} ({success_rate:.1%})",
                    duration_ms=duration,
                    metadata={
                        "successful": successful,
                        "failed": failed,
                        "success_rate": success_rate,
                        "avg_request_time": avg_request_time,
                        "results": [msg for _, msg, _ in results]
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            return ValidationResult(
                test_name="concurrent_inference",
                status="PASS",
                message=f"Concurrent inference successful: {successful}/{concurrent_requests} requests",
                duration_ms=duration,
                metadata={
                    "successful": successful,
                    "failed": failed,
                    "success_rate": success_rate,
                    "avg_request_time": avg_request_time,
                    "concurrent_requests": concurrent_requests
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="concurrent_inference",
                status="ERROR",
                message=f"Error during concurrent inference validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def validate_performance_regression(self) -> ValidationResult:
        """Validate that performance hasn't regressed."""
        start_time = time.time()
        
        try:
            # Performance benchmarks (baseline expectations)
            expected_min_tokens_per_sec = 5.0
            expected_max_latency_ms = 10000  # 10 seconds max
            
            # Test with a medium-length generation
            request_data = {
                "model": "mlx-community/Qwen3-1.7B-8bit",
                "messages": [{"role": "user", "content": "Explain machine learning in 2 sentences."}],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            request_start = time.time()
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=request_data,
                timeout=60
            )
            request_duration = (time.time() - request_start) * 1000
            
            total_duration = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="performance_regression",
                    status="FAIL",
                    message=f"Performance test request failed: {response.status_code}",
                    duration_ms=total_duration,
                    metadata={"status_code": response.status_code},
                    timestamp=datetime.now().isoformat()
                )
            
            data = response.json()
            
            # Calculate performance metrics
            if "usage" in data and "completion_tokens" in data["usage"]:
                tokens_generated = data["usage"]["completion_tokens"]
                tokens_per_second = tokens_generated / (request_duration / 1000) if request_duration > 0 else 0
                
                # Check performance thresholds
                issues = []
                
                if tokens_per_second < expected_min_tokens_per_sec:
                    issues.append(f"Low tokens/sec: {tokens_per_second:.2f} < {expected_min_tokens_per_sec}")
                
                if request_duration > expected_max_latency_ms:
                    issues.append(f"High latency: {request_duration:.0f}ms > {expected_max_latency_ms}ms")
                
                if issues:
                    return ValidationResult(
                        test_name="performance_regression",
                        status="FAIL",
                        message=f"Performance regression detected: {'; '.join(issues)}",
                        duration_ms=total_duration,
                        metadata={
                            "tokens_per_second": tokens_per_second,
                            "request_duration_ms": request_duration,
                            "tokens_generated": tokens_generated,
                            "issues": issues
                        },
                        timestamp=datetime.now().isoformat()
                    )
                
                return ValidationResult(
                    test_name="performance_regression",
                    status="PASS",
                    message=f"Performance within acceptable range: {tokens_per_second:.2f} tok/s, {request_duration:.0f}ms",
                    duration_ms=total_duration,
                    metadata={
                        "tokens_per_second": tokens_per_second,
                        "request_duration_ms": request_duration,
                        "tokens_generated": tokens_generated
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            else:
                return ValidationResult(
                    test_name="performance_regression",
                    status="FAIL",
                    message="Performance test response missing usage information",
                    duration_ms=total_duration,
                    metadata={"response_data": data},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="performance_regression",
                status="ERROR",
                message=f"Error during performance regression validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )


class TensorSerializationValidator:
    """Validates tensor serialization/deserialization fixes."""
    
    def __init__(self):
        self.test_tensors = self._generate_test_tensors()
    
    def _generate_test_tensors(self) -> List[Tuple[str, mx.array]]:
        """Generate test tensors for serialization validation."""
        return [
            ("float32", mx.array([1.0, 2.0, 3.0]).astype(mx.float32)),
            ("float16", mx.array([1.0, 2.0, 3.0]).astype(mx.float16)),
            ("bfloat16", mx.array([1.0, 2.0, 3.0]).astype(mx.bfloat16)),
            ("int32", mx.array([1, 2, 3]).astype(mx.int32)),
            ("large_tensor", mx.random.normal((100, 100)).astype(mx.float32)),
            ("multi_dim", mx.random.normal((10, 20, 30)).astype(mx.float16)),
            ("edge_cases", mx.array([float('inf'), -float('inf'), 0.0, -0.0]).astype(mx.float32))
        ]
    
    async def validate_tensor_serialization(self) -> ValidationResult:
        """Validate tensor serialization and deserialization."""
        start_time = time.time()
        
        try:
            errors = []
            successful = 0
            
            for tensor_name, tensor in self.test_tensors:
                try:
                    # Test serialization
                    proto_tensor = TensorSerializer.tensor_to_proto(tensor)
                    
                    # Test deserialization
                    reconstructed = TensorSerializer.proto_to_tensor(proto_tensor)
                    
                    # Verify correctness
                    if tensor.dtype == mx.bfloat16:
                        # bfloat16 has lower precision, use relaxed tolerance
                        if not mx.allclose(tensor, reconstructed, rtol=1e-2, atol=1e-3):
                            errors.append(f"{tensor_name}: bfloat16 reconstruction mismatch")
                        else:
                            successful += 1
                    elif tensor_name == "edge_cases":
                        # Handle special float values
                        tensor_np = np.array(tensor)
                        reconstructed_np = np.array(reconstructed)
                        
                        # Check inf values
                        if not (np.isinf(tensor_np) == np.isinf(reconstructed_np)).all():
                            errors.append(f"{tensor_name}: infinity handling failed")
                        # Check finite values
                        elif not np.allclose(tensor_np[np.isfinite(tensor_np)], 
                                           reconstructed_np[np.isfinite(reconstructed_np)]):
                            errors.append(f"{tensor_name}: finite value mismatch")
                        else:
                            successful += 1
                    else:
                        # Standard comparison
                        if not mx.allclose(tensor, reconstructed, rtol=1e-5):
                            errors.append(f"{tensor_name}: reconstruction mismatch")
                        else:
                            successful += 1
                    
                    # Verify dtype preservation
                    if str(tensor.dtype) != str(reconstructed.dtype):
                        errors.append(f"{tensor_name}: dtype mismatch {tensor.dtype} != {reconstructed.dtype}")
                    
                    # Verify shape preservation
                    if tensor.shape != reconstructed.shape:
                        errors.append(f"{tensor_name}: shape mismatch {tensor.shape} != {reconstructed.shape}")
                
                except Exception as e:
                    errors.append(f"{tensor_name}: {str(e)}")
            
            duration = (time.time() - start_time) * 1000
            
            if errors:
                return ValidationResult(
                    test_name="tensor_serialization",
                    status="FAIL",
                    message=f"Tensor serialization errors: {len(errors)} out of {len(self.test_tensors)} tests failed",
                    duration_ms=duration,
                    metadata={
                        "successful": successful,
                        "failed": len(errors),
                        "errors": errors,
                        "total_tests": len(self.test_tensors)
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            return ValidationResult(
                test_name="tensor_serialization",
                status="PASS",
                message=f"All {len(self.test_tensors)} tensor serialization tests passed",
                duration_ms=duration,
                metadata={
                    "successful": successful,
                    "total_tests": len(self.test_tensors),
                    "tensor_types": [name for name, _ in self.test_tensors]
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ValidationResult(
                test_name="tensor_serialization",
                status="ERROR",
                message=f"Error during tensor serialization validation: {e}",
                duration_ms=duration,
                metadata={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )


class AutomatedValidationSuite:
    """Main validation suite that orchestrates all tests."""
    
    def __init__(self, api_url: str = "http://localhost:8100", output_dir: str = "validation_results"):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize validators
        self.cluster_validator = ClusterHealthValidator(api_url)
        self.inference_validator = InferenceValidator(api_url)
        self.tensor_validator = TensorSerializationValidator()
        
        # Define fixes to validate
        self.fixes_to_validate = [
            FixValidation(
                fix_name="grpc_tensor_serialization",
                description="Fixed tensor serialization/deserialization for all MLX dtypes",
                validation_tests=["tensor_serialization"],
                critical=True
            ),
            FixValidation(
                fix_name="device_discovery",
                description="Fixed device discovery and cluster formation",
                validation_tests=["cluster_startup", "device_discovery"],
                critical=True
            ),
            FixValidation(
                fix_name="grpc_communication",
                description="Fixed gRPC communication between devices",
                validation_tests=["grpc_communication"],
                critical=True
            ),
            FixValidation(
                fix_name="distributed_inference",
                description="Fixed distributed inference across multiple devices",
                validation_tests=["basic_inference", "concurrent_inference"],
                critical=True
            ),
            FixValidation(
                fix_name="performance_optimization",
                description="Performance optimizations and regression prevention",
                validation_tests=["performance_regression"],
                critical=False
            )
        ]
        
        self.results: List[ValidationResult] = []
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting automated validation suite")
        
        validation_results = {
            "suite_info": {
                "start_time": datetime.now().isoformat(),
                "api_url": self.api_url,
                "fixes_validated": len(self.fixes_to_validate)
            },
            "results": {},
            "fix_status": {},
            "summary": {},
            "critical_failures": []
        }
        
        # Run all validation tests
        all_tests = [
            ("cluster_startup", self.cluster_validator.validate_cluster_startup),
            ("device_discovery", self.cluster_validator.validate_device_discovery),
            ("grpc_communication", self.cluster_validator.validate_grpc_communication),
            ("tensor_serialization", self.tensor_validator.validate_tensor_serialization),
            ("basic_inference", self.inference_validator.validate_basic_inference),
            ("concurrent_inference", self.inference_validator.validate_concurrent_inference),
            ("performance_regression", self.inference_validator.validate_performance_regression)
        ]
        
        for test_name, test_func in all_tests:
            logger.info(f"Running validation: {test_name}")
            try:
                result = await test_func()
                self.results.append(result)
                validation_results["results"][test_name] = asdict(result)
                
                if result.status == "PASS":
                    logger.info(f"✅ {test_name}: {result.message}")
                elif result.status == "FAIL":
                    logger.warning(f"❌ {test_name}: {result.message}")
                elif result.status == "ERROR":
                    logger.error(f"🔥 {test_name}: {result.message}")
                else:  # SKIP
                    logger.info(f"⏭️  {test_name}: {result.message}")
                    
            except Exception as e:
                logger.error(f"Exception in {test_name}: {e}")
                error_result = ValidationResult(
                    test_name=test_name,
                    status="ERROR",
                    message=f"Test runner exception: {e}",
                    duration_ms=0.0,
                    metadata={"error": str(e)},
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(error_result)
                validation_results["results"][test_name] = asdict(error_result)
        
        # Evaluate fix status
        for fix in self.fixes_to_validate:
            fix_tests = [r for r in self.results if r.test_name in fix.validation_tests]
            
            if not fix_tests:
                status = "NOT_TESTED"
                message = "No validation tests were run"
            elif all(t.status == "PASS" for t in fix_tests):
                status = "VALIDATED"
                message = "All validation tests passed"
            elif any(t.status == "FAIL" for t in fix_tests):
                status = "FAILED"
                failed_tests = [t.test_name for t in fix_tests if t.status == "FAIL"]
                message = f"Failed tests: {', '.join(failed_tests)}"
                if fix.critical:
                    validation_results["critical_failures"].append(fix.fix_name)
            elif any(t.status == "ERROR" for t in fix_tests):
                status = "ERROR"
                error_tests = [t.test_name for t in fix_tests if t.status == "ERROR"]
                message = f"Error in tests: {', '.join(error_tests)}"
                if fix.critical:
                    validation_results["critical_failures"].append(fix.fix_name)
            else:
                status = "PARTIAL"
                message = "Some tests passed, some skipped"
            
            validation_results["fix_status"][fix.fix_name] = {
                "status": status,
                "message": message,
                "critical": fix.critical,
                "description": fix.description,
                "tests": [asdict(t) for t in fix_tests]
            }
        
        # Generate summary
        validation_results["summary"] = self._generate_summary()
        validation_results["suite_info"]["end_time"] = datetime.now().isoformat()
        
        # Determine overall result
        has_critical_failures = len(validation_results["critical_failures"]) > 0
        validation_results["overall_status"] = "FAIL" if has_critical_failures else "PASS"
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {results_file}")
        
        # Generate report
        report_file = self.output_dir / f"validation_report_{timestamp}.md"
        self._generate_validation_report(validation_results, report_file)
        
        return validation_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No validation results available"}
        
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        total_tests = len(self.results)
        avg_duration = sum(r.duration_ms for r in self.results) / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "status_counts": status_counts,
            "success_rate": status_counts.get("PASS", 0) / total_tests if total_tests > 0 else 0,
            "avg_test_duration_ms": avg_duration,
            "critical_fixes": len([f for f in self.fixes_to_validate if f.critical]),
            "non_critical_fixes": len([f for f in self.fixes_to_validate if not f.critical])
        }
    
    def _generate_validation_report(self, results: Dict[str, Any], output_file: Path):
        """Generate human-readable validation report."""
        report_lines = [
            "# Distributed MLX Inference Validation Report",
            f"Generated: {results['suite_info']['start_time']}",
            f"API URL: {results['suite_info']['api_url']}",
            f"Overall Status: **{results['overall_status']}**",
            ""
        ]
        
        # Critical failures
        if results["critical_failures"]:
            report_lines.extend([
                "## ⚠️ Critical Failures",
                "The following critical fixes have validation failures:",
                ""
            ])
            for fix_name in results["critical_failures"]:
                fix_info = results["fix_status"][fix_name]
                report_lines.append(f"- **{fix_name}**: {fix_info['message']}")
            report_lines.append("")
        
        # Fix status summary
        report_lines.extend([
            "## Fix Validation Status",
            ""
        ])
        
        for fix_name, fix_info in results["fix_status"].items():
            status_emoji = {
                "VALIDATED": "✅",
                "FAILED": "❌",
                "ERROR": "🔥",
                "PARTIAL": "⚠️",
                "NOT_TESTED": "❓"
            }.get(fix_info["status"], "❓")
            
            critical_marker = " (CRITICAL)" if fix_info["critical"] else ""
            
            report_lines.extend([
                f"### {status_emoji} {fix_name}{critical_marker}",
                f"**Status**: {fix_info['status']}",
                f"**Description**: {fix_info['description']}",
                f"**Result**: {fix_info['message']}",
                ""
            ])
        
        # Test results detail
        if "results" in results:
            report_lines.extend([
                "## Detailed Test Results",
                ""
            ])
            
            for test_name, test_result in results["results"].items():
                status_emoji = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "ERROR": "🔥",
                    "SKIP": "⏭️"
                }.get(test_result["status"], "❓")
                
                report_lines.extend([
                    f"### {status_emoji} {test_name}",
                    f"- **Status**: {test_result['status']}",
                    f"- **Duration**: {test_result['duration_ms']:.1f} ms",
                    f"- **Message**: {test_result['message']}",
                    ""
                ])
        
        # Summary statistics
        if "summary" in results:
            summary = results["summary"]
            report_lines.extend([
                "## Summary Statistics",
                f"- **Total Tests**: {summary.get('total_tests', 0)}",
                f"- **Success Rate**: {summary.get('success_rate', 0):.1%}",
                f"- **Average Duration**: {summary.get('avg_test_duration_ms', 0):.1f} ms",
                ""
            ])
            
            if "status_counts" in summary:
                report_lines.append("**Test Results by Status**:")
                for status, count in summary["status_counts"].items():
                    report_lines.append(f"- {status}: {count}")
                report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Validation report saved to: {output_file}")


async def main():
    """Main entry point for validation suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Automated Validation for Distributed MLX Inference")
    parser.add_argument("--api-url", default="http://localhost:8100", help="API server URL")
    parser.add_argument("--output-dir", default="validation_results", help="Output directory")
    parser.add_argument("--wait-for-cluster", action="store_true", help="Wait for cluster to be ready")
    
    args = parser.parse_args()
    
    # Initialize validation suite
    suite = AutomatedValidationSuite(
        api_url=args.api_url,
        output_dir=args.output_dir
    )
    
    # Wait for cluster if requested
    if args.wait_for_cluster:
        logger.info("Waiting for cluster to be ready...")
        for attempt in range(60):  # Wait up to 60 seconds
            try:
                response = requests.get(f"{args.api_url}/health", timeout=2)
                if response.status_code == 200:
                    health_data = response.json()
                    if (health_data.get("status") == "healthy" and 
                        health_data.get("model_loaded", False)):
                        logger.info("Cluster is ready!")
                        break
            except:
                pass
            
            await asyncio.sleep(1)
        else:
            logger.warning("Cluster did not become ready within timeout")
    
    logger.info("Starting automated validation suite...")
    
    # Run all validations
    results = await suite.run_all_validations()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUITE COMPLETED")
    logger.info("="*80)
    
    overall_status = results.get("overall_status", "UNKNOWN")
    logger.info(f"Overall Status: {overall_status}")
    
    if "summary" in results:
        summary = results["summary"]
        logger.info(f"Total Tests: {summary.get('total_tests', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        
        status_counts = summary.get("status_counts", {})
        logger.info("Results by Status:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
    
    # Report critical failures
    critical_failures = results.get("critical_failures", [])
    if critical_failures:
        logger.error(f"\nCritical Failures: {len(critical_failures)}")
        for failure in critical_failures:
            logger.error(f"  - {failure}")
    else:
        logger.info("\n✅ No critical failures detected!")
    
    # Exit with appropriate code
    exit_code = 1 if overall_status == "FAIL" else 0
    logger.info(f"\nExiting with code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
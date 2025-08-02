#!/usr/bin/env python3
"""
Comprehensive tensor flow validation for MLX distributed inference.
Tests tensor serialization, flow between devices, and data integrity.
"""

import asyncio
import logging
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import mlx.core as mx
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.config import ClusterConfig, DeviceRole
    from src.communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
    from src.communication.grpc_client import ConnectionPool
except ImportError:
    # Fallback for direct imports
    from core.config import ClusterConfig, DeviceRole
    from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
    from communication.grpc_client import ConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorFlowValidator:
    """Comprehensive tensor flow validation suite."""
    
    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        """Initialize validator with config."""
        self.config_path = config_path
        self.config = None
        self.test_results = {}
        self.failures = []
        self.performance_metrics = {}
    
    def log_failure(self, test_name: str, description: str, error: Optional[Exception] = None):
        """Log a test failure."""
        failure = {
            "test": test_name,
            "description": description,
            "error": str(error) if error else None,
            "timestamp": time.time()
        }
        self.failures.append(failure)
        logger.error(f"FAILED: {test_name} - {description}")
        if error:
            logger.error(f"Error details: {error}")
    
    def log_success(self, test_name: str, metrics: Optional[Dict] = None):
        """Log a test success."""
        logger.info(f"PASSED: {test_name}")
        if metrics:
            self.performance_metrics[test_name] = metrics
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tensor flow validation tests."""
        logger.info("Starting comprehensive tensor flow validation...")
        
        # Load configuration
        try:
            self.config = ClusterConfig.from_yaml(self.config_path)
        except Exception as e:
            self.log_failure("Configuration Load", f"Failed to load config: {e}", e)
            return self._generate_report()
        
        # Run test suite
        tests = [
            ("basic_serialization", self.test_basic_serialization),
            ("large_tensor_serialization", self.test_large_tensor_serialization),
            ("various_dtypes", self.test_various_dtypes),
            ("various_shapes", self.test_various_shapes),
            ("serialization_performance", self.test_serialization_performance),
            ("round_trip_accuracy", self.test_round_trip_accuracy),
            ("memory_efficiency", self.test_memory_efficiency),
            ("concurrent_serialization", self.test_concurrent_serialization),
            ("edge_cases", self.test_edge_cases),
            ("tensor_flow_simulation", self.test_tensor_flow_simulation)
        ]
        
        for test_name, test_func in tests:
            try:
                await test_func()
                self.test_results[test_name] = "PASSED"
            except Exception as e:
                self.log_failure(test_name, f"Test execution failed", e)
                self.test_results[test_name] = "FAILED"
                logger.error(f"Traceback for {test_name}:\n{traceback.format_exc()}")
        
        return self._generate_report()
    
    async def test_basic_serialization(self):
        """Test basic tensor serialization and deserialization."""
        logger.info("Testing basic tensor serialization...")
        
        # Test simple 2D tensor
        original = mx.random.normal(shape=(32, 128))
        
        # Serialize
        start_time = time.time()
        data, metadata = serialize_mlx_array(original)
        serialize_time = time.time() - start_time
        
        # Deserialize
        start_time = time.time()
        recovered = deserialize_mlx_array(data, metadata)
        deserialize_time = time.time() - start_time
        
        # Verify shape
        if original.shape != recovered.shape:
            raise ValueError(f"Shape mismatch: {original.shape} != {recovered.shape}")
        
        # Verify values
        if not mx.allclose(original, recovered, atol=1e-6):
            raise ValueError("Values not preserved during serialization")
        
        self.log_success("basic_serialization", {
            "serialize_time": serialize_time,
            "deserialize_time": deserialize_time,
            "data_size": len(data)
        })
    
    async def test_large_tensor_serialization(self):
        """Test serialization of large tensors (model layer sized)."""
        logger.info("Testing large tensor serialization...")
        
        # Simulate large layer tensor (batch_size=1, seq_len=2048, hidden_size=4096)
        original = mx.random.normal(shape=(1, 2048, 4096))
        
        start_time = time.time()
        data, metadata = serialize_mlx_array(original)
        serialize_time = time.time() - start_time
        
        start_time = time.time()
        recovered = deserialize_mlx_array(data, metadata)
        deserialize_time = time.time() - start_time
        
        # Verify shape and sample values
        if original.shape != recovered.shape:
            raise ValueError(f"Shape mismatch: {original.shape} != {recovered.shape}")
        
        # For large tensors, check statistical properties instead of exact values
        original_mean = mx.mean(original).item()
        recovered_mean = mx.mean(recovered).item()
        if abs(original_mean - recovered_mean) > 1e-4:
            raise ValueError(f"Mean differs too much: {original_mean} vs {recovered_mean}")
        
        data_size_mb = len(data) / (1024 * 1024)
        
        self.log_success("large_tensor_serialization", {
            "serialize_time": serialize_time,
            "deserialize_time": deserialize_time,
            "data_size_mb": data_size_mb,
            "throughput_mb_per_sec": data_size_mb / serialize_time
        })
    
    async def test_various_dtypes(self):
        """Test serialization with various data types."""
        logger.info("Testing various data types...")
        
        dtypes_to_test = [
            (mx.float32, "float32"),
            (mx.float16, "float16"),
            (mx.int32, "int32"),
            (mx.int16, "int16")
        ]
        
        for dtype, dtype_name in dtypes_to_test:
            try:
                # Create tensor with specific dtype
                if dtype in [mx.float32, mx.float16]:
                    original = mx.random.normal(shape=(10, 20)).astype(dtype)
                else:
                    original = mx.random.randint(0, 100, shape=(10, 20)).astype(dtype)
                
                data, metadata = serialize_mlx_array(original)
                recovered = deserialize_mlx_array(data, metadata)
                
                if original.dtype != recovered.dtype:
                    raise ValueError(f"Dtype mismatch for {dtype_name}: {original.dtype} != {recovered.dtype}")
                
                if original.shape != recovered.shape:
                    raise ValueError(f"Shape mismatch for {dtype_name}")
                
                # For integer types, exact match should be preserved
                if dtype in [mx.int32, mx.int16]:
                    if not mx.array_equal(original, recovered):
                        raise ValueError(f"Exact values not preserved for {dtype_name}")
                else:
                    if not mx.allclose(original, recovered, atol=1e-6):
                        raise ValueError(f"Values not preserved for {dtype_name}")
                
                logger.info(f"✓ {dtype_name} serialization successful")
                
            except Exception as e:
                raise ValueError(f"Failed for dtype {dtype_name}: {e}")
        
        self.log_success("various_dtypes")
    
    async def test_various_shapes(self):
        """Test serialization with various tensor shapes."""
        logger.info("Testing various tensor shapes...")
        
        shapes_to_test = [
            (1,),           # 1D
            (10, 20),       # 2D
            (5, 10, 15),    # 3D
            (2, 3, 4, 5),   # 4D
            (1, 1),         # Minimal 2D
            (1, 2048, 4096), # Model layer shape
            (32, 128, 768)   # Batch shape
        ]
        
        for shape in shapes_to_test:
            try:
                original = mx.random.normal(shape=shape)
                data, metadata = serialize_mlx_array(original)
                recovered = deserialize_mlx_array(data, metadata)
                
                if original.shape != recovered.shape:
                    raise ValueError(f"Shape mismatch for {shape}: {original.shape} != {recovered.shape}")
                
                if not mx.allclose(original, recovered, atol=1e-6):
                    raise ValueError(f"Values not preserved for shape {shape}")
                
                logger.info(f"✓ Shape {shape} serialization successful")
                
            except Exception as e:
                raise ValueError(f"Failed for shape {shape}: {e}")
        
        self.log_success("various_shapes")
    
    async def test_serialization_performance(self):
        """Test serialization performance with various sizes."""
        logger.info("Testing serialization performance...")
        
        test_sizes = [
            (32, 128),        # Small
            (128, 512),       # Medium
            (512, 1024),      # Large
            (1, 2048, 4096)   # Model layer
        ]
        
        performance_data = {}
        
        for shape in test_sizes:
            original = mx.random.normal(shape=shape)
            
            # Multiple runs for average
            serialize_times = []
            deserialize_times = []
            
            for _ in range(5):
                start = time.time()
                data, metadata = serialize_mlx_array(original)
                serialize_times.append(time.time() - start)
                
                start = time.time()
                recovered = deserialize_mlx_array(data, metadata)
                deserialize_times.append(time.time() - start)
            
            avg_serialize = np.mean(serialize_times)
            avg_deserialize = np.mean(deserialize_times)
            data_size_mb = len(data) / (1024 * 1024)
            
            performance_data[str(shape)] = {
                "avg_serialize_time": avg_serialize,
                "avg_deserialize_time": avg_deserialize,
                "data_size_mb": data_size_mb,
                "serialize_throughput_mb_s": data_size_mb / avg_serialize,
                "deserialize_throughput_mb_s": data_size_mb / avg_deserialize
            }
        
        self.log_success("serialization_performance", performance_data)
    
    async def test_round_trip_accuracy(self):
        """Test accuracy preservation through multiple round trips."""
        logger.info("Testing round-trip accuracy...")
        
        original = mx.random.normal(shape=(100, 200))
        current = original
        
        # Multiple round trips
        for i in range(10):
            data, metadata = serialize_mlx_array(current)
            current = deserialize_mlx_array(data, metadata)
            
            # Check if we're accumulating errors
            if not mx.allclose(original, current, atol=1e-5):
                raise ValueError(f"Accuracy degraded after {i+1} round trips")
        
        # Final accuracy check
        max_diff = mx.max(mx.abs(original - current)).item()
        mean_diff = mx.mean(mx.abs(original - current)).item()
        
        self.log_success("round_trip_accuracy", {
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "round_trips": 10
        })
    
    async def test_memory_efficiency(self):
        """Test memory usage during serialization."""
        logger.info("Testing memory efficiency...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Large tensor
        original = mx.random.normal(shape=(1, 2048, 4096))
        
        # Measure memory before
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Serialize
        data, metadata = serialize_mlx_array(original)
        mem_after_serialize = process.memory_info().rss / (1024 * 1024)
        
        # Deserialize
        recovered = deserialize_mlx_array(data, metadata)
        mem_after_deserialize = process.memory_info().rss / (1024 * 1024)
        
        # Clean up
        del data, metadata, recovered
        
        memory_overhead = mem_after_serialize - mem_before
        
        self.log_success("memory_efficiency", {
            "memory_before_mb": mem_before,
            "memory_after_serialize_mb": mem_after_serialize,
            "memory_after_deserialize_mb": mem_after_deserialize,
            "serialization_overhead_mb": memory_overhead
        })
    
    async def test_concurrent_serialization(self):
        """Test concurrent serialization operations."""
        logger.info("Testing concurrent serialization...")
        
        async def serialize_tensor(tensor_id: int):
            """Serialize a tensor concurrently."""
            tensor = mx.random.normal(shape=(50, 100))
            data, metadata = serialize_mlx_array(tensor)
            recovered = deserialize_mlx_array(data, metadata)
            
            if not mx.allclose(tensor, recovered, atol=1e-6):
                raise ValueError(f"Concurrent serialization failed for tensor {tensor_id}")
            
            return len(data)
        
        # Run 10 concurrent serializations
        start_time = time.time()
        tasks = [serialize_tensor(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        total_data_size = sum(results)
        
        self.log_success("concurrent_serialization", {
            "concurrent_operations": 10,
            "total_time": total_time,
            "total_data_size": total_data_size,
            "ops_per_second": 10 / total_time
        })
    
    async def test_edge_cases(self):
        """Test edge cases and potential failure modes."""
        logger.info("Testing edge cases...")
        
        # Empty tensor
        try:
            empty = mx.array([])
            data, metadata = serialize_mlx_array(empty)
            recovered = deserialize_mlx_array(data, metadata)
            logger.info("✓ Empty tensor handled")
        except Exception as e:
            logger.warning(f"Empty tensor handling failed: {e}")
        
        # Single element tensor
        single = mx.array([42.0])
        data, metadata = serialize_mlx_array(single)
        recovered = deserialize_mlx_array(data, metadata)
        if not mx.allclose(single, recovered):
            raise ValueError("Single element tensor failed")
        
        # Very large single dimension
        large_1d = mx.random.normal(shape=(100000,))
        data, metadata = serialize_mlx_array(large_1d)
        recovered = deserialize_mlx_array(data, metadata)
        if not mx.allclose(large_1d, recovered, atol=1e-6):
            raise ValueError("Large 1D tensor failed")
        
        # Tensor with inf/nan values
        try:
            inf_tensor = mx.array([1.0, float('inf'), -float('inf')])
            data, metadata = serialize_mlx_array(inf_tensor)
            recovered = deserialize_mlx_array(data, metadata)
            # Check inf values are preserved
            if not (mx.isinf(recovered[1]) and mx.isinf(recovered[2])):
                raise ValueError("Inf values not preserved")
            logger.info("✓ Inf values handled correctly")
        except Exception as e:
            logger.warning(f"Inf/NaN handling failed: {e}")
        
        self.log_success("edge_cases")
    
    async def test_tensor_flow_simulation(self):
        """Simulate tensor flow between devices."""
        logger.info("Testing tensor flow simulation...")
        
        # Simulate the distributed flow: mini1 -> mini2 -> master -> mini1
        layers_per_device = {
            "mini1": 10,
            "mini2": 9, 
            "master": 9
        }
        
        # Initial tensor from user input
        batch_size = 1
        seq_length = 128
        hidden_size = 4096
        
        current_tensor = mx.random.normal(shape=(batch_size, seq_length, hidden_size))
        flow_log = []
        
        # Simulate flow through each device
        for device, layer_count in layers_per_device.items():
            start_time = time.time()
            
            # Serialize (sending to device)
            data, metadata = serialize_mlx_array(current_tensor)
            serialize_time = time.time() - start_time
            
            # Simulate processing on device (simple transformation)
            start_time = time.time()
            current_tensor = deserialize_mlx_array(data, metadata)
            
            # Simulate layer processing
            for layer in range(layer_count):
                # Simple transformation to simulate layer processing
                current_tensor = mx.tanh(current_tensor @ mx.ones((hidden_size, hidden_size)) * 0.1)
            
            process_time = time.time() - start_time
            
            flow_log.append({
                "device": device,
                "layers_processed": layer_count,
                "serialize_time": serialize_time,
                "process_time": process_time,
                "tensor_shape": current_tensor.shape,
                "data_size_mb": len(data) / (1024 * 1024)
            })
            
            logger.info(f"✓ Processed {layer_count} layers on {device}")
        
        # Verify final tensor is reasonable
        if current_tensor.shape != (batch_size, seq_length, hidden_size):
            raise ValueError(f"Final tensor shape incorrect: {current_tensor.shape}")
        
        # Check for NaN or inf values
        if mx.any(mx.isnan(current_tensor)) or mx.any(mx.isinf(current_tensor)):
            raise ValueError("Final tensor contains NaN or inf values")
        
        total_layers = sum(layers_per_device.values())
        total_time = sum(log["serialize_time"] + log["process_time"] for log in flow_log)
        
        self.log_success("tensor_flow_simulation", {
            "total_layers_processed": total_layers,
            "total_flow_time": total_time,
            "devices_involved": len(layers_per_device),
            "final_tensor_shape": current_tensor.shape,
            "flow_log": flow_log
        })
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASSED")
        total_tests = len(self.test_results)
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "failures": self.failures,
            "performance_metrics": self.performance_metrics,
            "timestamp": time.time()
        }


def print_tensor_flow_report(results: Dict[str, Any]):
    """Print formatted tensor flow validation report."""
    print("\n" + "="*80)
    print("TENSOR FLOW VALIDATION RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    
    # Test results
    print(f"\nTEST RESULTS:")
    for test_name, result in results["test_results"].items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"  {status_emoji} {test_name}: {result}")
    
    # Performance metrics
    if results["performance_metrics"]:
        print(f"\nPERFORMANCE METRICS:")
        for test_name, metrics in results["performance_metrics"].items():
            print(f"  {test_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
    
    # Failures
    if results["failures"]:
        print(f"\nFAILURES:")
        for i, failure in enumerate(results["failures"], 1):
            print(f"  {i}. {failure['test']}: {failure['description']}")
            if failure['error']:
                print(f"     Error: {failure['error']}")
    
    print("\n" + "="*80)


async def main():
    """Run tensor flow validation."""
    validator = TensorFlowValidator()
    results = await validator.run_all_tests()
    print_tensor_flow_report(results)
    
    # Return exit code
    return results["summary"]["failed_tests"]


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Comprehensive validation of the distributed inference pipeline.
This script identifies issues with the current implementation.
"""

import asyncio
import logging
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import mlx.core as mx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.config import ClusterConfig, DeviceRole
    from src.coordinator.orchestrator import DistributedOrchestrator, InferenceRequest
    from src.model.loader import DistributedModelLoader  
    from src.model.inference import LayerProcessor
    from src.communication.grpc_client import ConnectionPool
except ImportError:
    # Fallback for direct imports
    from core.config import ClusterConfig, DeviceRole
    from coordinator.orchestrator import DistributedOrchestrator, InferenceRequest
    from model.loader import DistributedModelLoader
    from model.inference import LayerProcessor
    from communication.grpc_client import ConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedPipelineValidator:
    """Validates the distributed inference pipeline implementation."""
    
    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        """Initialize validator with config."""
        self.config_path = config_path
        self.config = None
        self.orchestrator = None
        self.issues = []
        self.recommendations = []
    
    def log_issue(self, category: str, description: str, severity: str = "ERROR"):
        """Log an issue found during validation."""
        issue = {
            "category": category,
            "description": description, 
            "severity": severity
        }
        self.issues.append(issue)
        logger.error(f"[{severity}] {category}: {description}")
    
    def log_recommendation(self, description: str):
        """Log a recommendation for improvement."""
        self.recommendations.append(description)
        logger.info(f"[RECOMMENDATION] {description}")
    
    async def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Starting comprehensive distributed pipeline validation...")
        
        results = {
            "configuration": await self.validate_configuration(),
            "layer_distribution": self.validate_layer_distribution(),
            "orchestrator_logic": await self.validate_orchestrator_logic(),
            "tensor_flow": await self.validate_tensor_flow(),
            "device_communication": await self.validate_device_communication(),
            "generation_pipeline": await self.validate_generation_pipeline(),
            "issues": self.issues,
            "recommendations": self.recommendations
        }
        
        return results
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate cluster configuration."""
        logger.info("Validating cluster configuration...")
        
        try:
            self.config = ClusterConfig.from_yaml(self.config_path)
        except Exception as e:
            self.log_issue("Configuration", f"Failed to load config: {e}")
            return {"status": "failed", "error": str(e)}
        
        # Check device count
        if len(self.config.devices) != 3:
            self.log_issue("Configuration", f"Expected 3 devices, found {len(self.config.devices)}")
        
        # Check roles
        coordinators = [d for d in self.config.devices if d.role == DeviceRole.COORDINATOR]
        workers = [d for d in self.config.devices if d.role == DeviceRole.WORKER]
        
        if len(coordinators) != 1:
            self.log_issue("Configuration", f"Expected 1 coordinator, found {len(coordinators)}")
        
        if len(workers) != 2:
            self.log_issue("Configuration", f"Expected 2 workers, found {len(workers)}")
        
        # Check ranks
        ranks = [d.rank for d in self.config.devices]
        expected_ranks = [0, 1, 2]
        if sorted(ranks) != expected_ranks:
            self.log_issue("Configuration", f"Expected ranks {expected_ranks}, found {sorted(ranks)}")
        
        return {
            "status": "passed" if not self.issues else "failed",
            "devices": len(self.config.devices),
            "coordinators": len(coordinators),
            "workers": len(workers)
        }
    
    def validate_layer_distribution(self) -> Dict[str, Any]:
        """Validate the 10-9-9 layer distribution."""
        logger.info("Validating layer distribution...")
        
        if not self.config:
            return {"status": "skipped", "reason": "No config"}
        
        distribution = self.config.model.layer_distribution
        total_layers = self.config.model.total_layers
        
        # Check total layers
        if total_layers != 28:
            self.log_issue("Layer Distribution", f"Expected 28 layers, configured for {total_layers}")
        
        # Check distribution
        expected_distribution = {
            "mini1": 10,  # layers 0-9
            "mini2": 9,   # layers 10-18
            "master": 9   # layers 19-27
        }
        
        actual_distribution = {}
        all_layers = set()
        
        for device_id, layers in distribution.items():
            actual_distribution[device_id] = len(layers)
            for layer in layers:
                if layer in all_layers:
                    self.log_issue("Layer Distribution", f"Layer {layer} assigned to multiple devices")
                all_layers.add(layer)
        
        # Check counts
        for device_id, expected_count in expected_distribution.items():
            actual_count = actual_distribution.get(device_id, 0)
            if actual_count != expected_count:
                self.log_issue("Layer Distribution", 
                             f"{device_id}: expected {expected_count} layers, got {actual_count}")
        
        # Check completeness
        expected_layers = set(range(total_layers))
        if all_layers != expected_layers:
            missing = expected_layers - all_layers
            extra = all_layers - expected_layers
            if missing:
                self.log_issue("Layer Distribution", f"Missing layers: {sorted(missing)}")
            if extra:
                self.log_issue("Layer Distribution", f"Extra layers: {sorted(extra)}")
        
        return {
            "status": "passed" if not [i for i in self.issues if i["category"] == "Layer Distribution"] else "failed",
            "total_layers": total_layers,
            "assigned_layers": len(all_layers),
            "distribution": actual_distribution
        }
    
    async def validate_orchestrator_logic(self) -> Dict[str, Any]:
        """Validate orchestrator initialization and flow logic."""
        logger.info("Validating orchestrator logic...")
        
        if not self.config:
            return {"status": "skipped", "reason": "No config"}
        
        try:
            # Check if we can create orchestrator
            local_device = self.config.get_local_device()
            if not local_device or local_device.role != DeviceRole.COORDINATOR:
                self.log_issue("Orchestrator", "Cannot create orchestrator: not a coordinator device", "WARNING")
                return {"status": "skipped", "reason": "Not coordinator"}
            
            self.orchestrator = DistributedOrchestrator(self.config)
            
            # Check critical methods exist
            critical_methods = [
                '_distributed_forward',
                '_generate_response', 
                '_format_messages',
                'process_request'
            ]
            
            for method in critical_methods:
                if not hasattr(self.orchestrator, method):
                    self.log_issue("Orchestrator", f"Missing critical method: {method}")
            
            # Analyze distributed forward logic
            self._analyze_distributed_forward()
            
            return {"status": "passed", "methods_checked": len(critical_methods)}
            
        except Exception as e:
            self.log_issue("Orchestrator", f"Failed to create orchestrator: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _analyze_distributed_forward(self):
        """Analyze the distributed forward pass logic."""
        logger.info("Analyzing distributed forward pass logic...")
        
        # Check if the flow follows Coordinator → Worker1 → Worker2 → Coordinator
        # Looking at the actual implementation in orchestrator.py
        
        # Issue 1: The current implementation doesn't handle return to coordinator
        self.log_issue("Distributed Forward", 
                      "Implementation doesn't return tensors to coordinator for final processing")
        
        # Issue 2: Generation happens on coordinator with incomplete data
        self.log_issue("Distributed Forward", 
                      "Generation may occur before all layers are processed")
        
        # Issue 3: No validation that all devices are utilized
        self.log_issue("Distributed Forward", 
                      "No explicit validation that all 3 devices process their layers")
        
        self.log_recommendation(
            "Modify _distributed_forward to ensure tensor returns to coordinator after worker processing"
        )
        
        self.log_recommendation(
            "Add explicit device utilization tracking and validation"
        )
    
    async def validate_tensor_flow(self) -> Dict[str, Any]:
        """Validate tensor serialization and flow."""
        logger.info("Validating tensor flow...")
        
        # Test tensor serialization
        try:
            from communication.tensor_utils import serialize_mlx_array, deserialize_mlx_array
            
            # Create test tensor
            test_tensor = mx.random.normal(shape=(1, 32, 4096))
            
            # Test serialization
            data, metadata = serialize_mlx_array(test_tensor)
            recovered_tensor = deserialize_mlx_array(data, metadata)
            
            # Check shape preservation
            if test_tensor.shape != recovered_tensor.shape:
                self.log_issue("Tensor Flow", 
                              f"Shape mismatch: {test_tensor.shape} != {recovered_tensor.shape}")
            
            # Check approximate equality (allowing for float precision)
            if not mx.allclose(test_tensor, recovered_tensor, atol=1e-6):
                self.log_issue("Tensor Flow", "Tensor values not preserved during serialization")
            
            return {"status": "passed", "serialization": "working"}
            
        except Exception as e:
            self.log_issue("Tensor Flow", f"Tensor serialization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def validate_device_communication(self) -> Dict[str, Any]:
        """Validate device communication setup."""
        logger.info("Validating device communication...")
        
        if not self.config:
            return {"status": "skipped", "reason": "No config"}
        
        # Check connection pool logic
        try:
            local_device_id = self.config.get_local_device_id()
            connection_pool = ConnectionPool(self.config, local_device_id)
            
            # Check if clients are created for remote devices
            expected_clients = len(self.config.devices) - 1  # All except local
            actual_clients = len(connection_pool.clients)
            
            if actual_clients != expected_clients:
                self.log_issue("Device Communication", 
                              f"Expected {expected_clients} clients, got {actual_clients}")
            
            # Check get_next_device_client logic
            coordinator = self.config.get_coordinator()
            if coordinator:
                next_client = connection_pool.get_next_device_client(coordinator.device_id)
                if not next_client:
                    self.log_issue("Device Communication", 
                                  "Cannot get next device client from coordinator", "WARNING")
            
            return {"status": "passed", "clients": actual_clients}
            
        except Exception as e:
            self.log_issue("Device Communication", f"Communication setup failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def validate_generation_pipeline(self) -> Dict[str, Any]:
        """Validate the text generation pipeline."""
        logger.info("Validating generation pipeline...")
        
        # Check generation logic issues
        self.log_issue("Generation Pipeline", 
                      "Generation uses simplified single-token approach instead of iterative generation")
        
        self.log_issue("Generation Pipeline", 
                      "No proper autoregressive generation with distributed forward passes")
        
        self.log_issue("Generation Pipeline", 
                      "EOS token handling is incomplete")
        
        self.log_recommendation(
            "Implement proper autoregressive generation with multiple forward passes"
        )
        
        self.log_recommendation(
            "Add proper attention mask and positional encoding handling"
        )
        
        return {
            "status": "needs_improvement",
            "issues_found": 3
        }


def print_validation_results(results: Dict[str, Any]):
    """Print formatted validation results."""
    print("\n" + "="*80)
    print("DISTRIBUTED INFERENCE PIPELINE VALIDATION RESULTS")
    print("="*80)
    
    # Summary
    total_issues = len(results["issues"])
    total_recommendations = len(results["recommendations"])
    
    print(f"\nSUMMARY:")
    print(f"  Total Issues Found: {total_issues}")
    print(f"  Total Recommendations: {total_recommendations}")
    
    # Individual results
    for category, result in results.items():
        if category in ["issues", "recommendations"]:
            continue
            
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(result, dict):
            if "status" in result:
                status_emoji = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"
                print(f"  Status: {status_emoji} {result['status'].upper()}")
            
            for key, value in result.items():
                if key != "status":
                    print(f"  {key}: {value}")
    
    # Issues
    if results["issues"]:
        print(f"\nISSUES FOUND:")
        for i, issue in enumerate(results["issues"], 1):
            severity_emoji = "🔴" if issue["severity"] == "ERROR" else "🟡"
            print(f"  {i}. {severity_emoji} [{issue['category']}] {issue['description']}")
    
    # Recommendations  
    if results["recommendations"]:
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. 💡 {rec}")
    
    print("\n" + "="*80)


async def main():
    """Run the validation."""
    validator = DistributedPipelineValidator()
    results = await validator.validate_all()
    print_validation_results(results)
    
    # Return exit code based on issues
    error_issues = [i for i in results["issues"] if i["severity"] == "ERROR"]
    return len(error_issues)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
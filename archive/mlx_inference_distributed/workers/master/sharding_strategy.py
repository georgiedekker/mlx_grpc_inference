"""
Resource-aware sharding strategies for heterogeneous Apple Silicon devices.

This module implements various algorithms to distribute model layers across
devices with different memory and compute capabilities.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import logging
from device_capabilities import DeviceProfile
from model_abstraction import ModelInfo, BaseModelWrapper

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Available sharding strategies."""
    UNIFORM = "uniform"
    MEMORY_PROPORTIONAL = "memory_proportional"
    COMPUTE_PROPORTIONAL = "compute_proportional"
    BALANCED = "balanced"  # Considers memory, compute, and bandwidth
    CUSTOM = "custom"


@dataclass
class ShardAssignment:
    """Assignment of model layers to a specific device."""
    device_id: str
    device_profile: DeviceProfile
    start_layer: int
    end_layer: int  # Exclusive
    num_layers: int
    estimated_memory_gb: float
    estimated_compute_load: float
    has_embedding: bool = False
    has_lm_head: bool = False
    
    @property
    def layer_indices(self) -> List[int]:
        return list(range(self.start_layer, self.end_layer))
    
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        return (self.estimated_memory_gb / self.device_profile.max_recommended_model_size_gb) * 100


@dataclass
class ShardingPlan:
    """Complete sharding plan for distributing a model across devices."""
    model_info: ModelInfo
    assignments: List[ShardAssignment]
    strategy: ShardingStrategy
    total_memory_gb: float
    max_memory_utilization: float
    balance_score: float  # 0-1, higher is better
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the sharding plan.
        
        Returns:
            (is_valid, error_message)
        """
        # Check all layers are assigned
        all_layers = set()
        for assignment in self.assignments:
            for layer in assignment.layer_indices:
                if layer in all_layers:
                    return False, f"Layer {layer} assigned multiple times"
                all_layers.add(layer)
        
        expected_layers = set(range(self.model_info.num_layers))
        if all_layers != expected_layers:
            missing = expected_layers - all_layers
            return False, f"Missing layers: {missing}"
        
        # Check memory constraints
        for assignment in self.assignments:
            if assignment.estimated_memory_gb > assignment.device_profile.max_recommended_model_size_gb:
                return False, f"Device {assignment.device_id} over memory limit"
        
        # Check special layers
        embedding_count = sum(1 for a in self.assignments if a.has_embedding)
        lm_head_count = sum(1 for a in self.assignments if a.has_lm_head)
        
        if embedding_count != 1:
            return False, f"Expected 1 embedding assignment, got {embedding_count}"
        if lm_head_count != 1:
            return False, f"Expected 1 lm_head assignment, got {lm_head_count}"
        
        return True, None
    
    def print_summary(self):
        """Print a summary of the sharding plan."""
        print(f"\nSharding Plan Summary:")
        print(f"  Model: {self.model_info.name}")
        print(f"  Strategy: {self.strategy.value}")
        print(f"  Total Layers: {self.model_info.num_layers}")
        print(f"  Total Memory Required: {self.total_memory_gb:.2f} GB")
        print(f"  Max Memory Utilization: {self.max_memory_utilization:.1f}%")
        print(f"  Balance Score: {self.balance_score:.3f}")
        print(f"\nDevice Assignments:")
        
        for assignment in self.assignments:
            print(f"  {assignment.device_id}:")
            print(f"    - Layers: {assignment.start_layer}-{assignment.end_layer-1} ({assignment.num_layers} layers)")
            print(f"    - Memory: {assignment.estimated_memory_gb:.2f} GB / {assignment.device_profile.memory_gb:.1f} GB ({assignment.memory_utilization():.1f}%)")
            print(f"    - GPU Cores: {assignment.device_profile.gpu_cores}")
            if assignment.has_embedding:
                print(f"    - Has embedding layer")
            if assignment.has_lm_head:
                print(f"    - Has output head")


class ResourceAwareShardingPlanner:
    """Planner for creating optimal sharding strategies."""
    
    def __init__(self):
        self.strategies = {
            ShardingStrategy.UNIFORM: self._uniform_sharding,
            ShardingStrategy.MEMORY_PROPORTIONAL: self._memory_proportional_sharding,
            ShardingStrategy.COMPUTE_PROPORTIONAL: self._compute_proportional_sharding,
            ShardingStrategy.BALANCED: self._balanced_sharding,
        }
    
    def create_plan(self, 
                   model_info: ModelInfo,
                   devices: List[DeviceProfile],
                   strategy: ShardingStrategy = ShardingStrategy.BALANCED,
                   custom_proportions: Optional[List[float]] = None) -> ShardingPlan:
        """Create a sharding plan for the given model and devices.
        
        Args:
            model_info: Information about the model to shard
            devices: List of available devices
            strategy: Sharding strategy to use
            custom_proportions: Custom layer proportions for CUSTOM strategy
            
        Returns:
            ShardingPlan with device assignments
        """
        if not devices:
            raise ValueError("No devices provided")
        
        if strategy == ShardingStrategy.CUSTOM and not custom_proportions:
            raise ValueError("custom_proportions required for CUSTOM strategy")
        
        logger.info(f"Creating sharding plan using {strategy.value} strategy")
        
        # Sort devices by capability (memory * gpu_cores) for consistent ordering
        devices = sorted(devices, key=lambda d: d.memory_gb * d.gpu_cores, reverse=True)
        
        if strategy == ShardingStrategy.CUSTOM:
            assignments = self._custom_sharding(model_info, devices, custom_proportions)
        else:
            strategy_fn = self.strategies.get(strategy)
            if not strategy_fn:
                raise ValueError(f"Unknown strategy: {strategy}")
            assignments = strategy_fn(model_info, devices)
        
        # Calculate total memory and utilization
        total_memory = sum(a.estimated_memory_gb for a in assignments)
        max_utilization = max(a.memory_utilization() for a in assignments)
        
        # Calculate balance score
        balance_score = self._calculate_balance_score(assignments)
        
        plan = ShardingPlan(
            model_info=model_info,
            assignments=assignments,
            strategy=strategy,
            total_memory_gb=total_memory,
            max_memory_utilization=max_utilization,
            balance_score=balance_score
        )
        
        # Validate the plan
        is_valid, error = plan.validate()
        if not is_valid:
            raise ValueError(f"Invalid sharding plan: {error}")
        
        return plan
    
    def _uniform_sharding(self, model_info: ModelInfo, 
                         devices: List[DeviceProfile]) -> List[ShardAssignment]:
        """Create uniform layer distribution across devices."""
        num_devices = len(devices)
        num_layers = model_info.num_layers
        
        # Calculate base layers per device and remainder
        base_layers = num_layers // num_devices
        extra_layers = num_layers % num_devices
        
        assignments = []
        current_layer = 0
        
        for i, device in enumerate(devices):
            # Distribute extra layers to first devices
            device_layers = base_layers + (1 if i < extra_layers else 0)
            
            # Calculate memory for this shard
            layer_memory = self._estimate_layer_memory(model_info, device_layers)
            
            # Add embedding/head memory for first/last device
            total_memory = layer_memory
            if i == 0:
                total_memory += self._estimate_embedding_memory(model_info)
            if i == num_devices - 1:
                total_memory += self._estimate_head_memory(model_info)
            
            assignment = ShardAssignment(
                device_id=device.device_id,
                device_profile=device,
                start_layer=current_layer,
                end_layer=current_layer + device_layers,
                num_layers=device_layers,
                estimated_memory_gb=total_memory,
                estimated_compute_load=device_layers / num_layers,
                has_embedding=(i == 0),
                has_lm_head=(i == num_devices - 1)
            )
            assignments.append(assignment)
            
            current_layer += device_layers
        
        return assignments
    
    def _memory_proportional_sharding(self, model_info: ModelInfo,
                                     devices: List[DeviceProfile]) -> List[ShardAssignment]:
        """Distribute layers proportional to available memory."""
        # Calculate total available memory
        total_memory = sum(d.max_recommended_model_size_gb for d in devices)
        
        # Calculate proportions based on memory
        proportions = [d.max_recommended_model_size_gb / total_memory for d in devices]
        
        return self._proportional_sharding(model_info, devices, proportions)
    
    def _compute_proportional_sharding(self, model_info: ModelInfo,
                                      devices: List[DeviceProfile]) -> List[ShardAssignment]:
        """Distribute layers proportional to GPU compute power."""
        # Calculate total GPU cores
        total_gpu_cores = sum(d.gpu_cores for d in devices)
        
        # Calculate proportions based on GPU cores
        proportions = [d.gpu_cores / total_gpu_cores for d in devices]
        
        return self._proportional_sharding(model_info, devices, proportions)
    
    def _balanced_sharding(self, model_info: ModelInfo,
                          devices: List[DeviceProfile]) -> List[ShardAssignment]:
        """Balance between memory, compute, and estimated model size."""
        # Calculate a composite score for each device
        scores = []
        
        for device in devices:
            # Normalize metrics
            memory_score = device.max_recommended_model_size_gb / 100  # Assume 100GB max
            gpu_score = device.gpu_cores / 80  # Assume 80 cores max (Ultra)
            
            # Weight: 60% memory, 40% compute (memory is usually the bottleneck)
            composite_score = 0.6 * memory_score + 0.4 * gpu_score
            scores.append(composite_score)
        
        total_score = sum(scores)
        proportions = [s / total_score for s in scores]
        
        # Adjust proportions to ensure no device is overloaded
        model_size_gb = model_info.estimate_size_gb()
        
        for i, (device, proportion) in enumerate(zip(devices, proportions)):
            max_proportion = device.max_recommended_model_size_gb / model_size_gb
            if proportion > max_proportion:
                # Reduce this device's proportion
                excess = proportion - max_proportion
                proportions[i] = max_proportion
                
                # Redistribute excess to other devices
                remaining_devices = len(devices) - 1
                if remaining_devices > 0:
                    for j in range(len(devices)):
                        if j != i:
                            proportions[j] += excess / remaining_devices
        
        # Normalize proportions
        total = sum(proportions)
        proportions = [p / total for p in proportions]
        
        return self._proportional_sharding(model_info, devices, proportions)
    
    def _proportional_sharding(self, model_info: ModelInfo,
                              devices: List[DeviceProfile],
                              proportions: List[float]) -> List[ShardAssignment]:
        """Create assignments based on given proportions."""
        num_layers = model_info.num_layers
        
        # Convert proportions to layer counts
        layer_counts = []
        assigned_layers = 0
        
        for i, proportion in enumerate(proportions[:-1]):
            layers = int(num_layers * proportion)
            layer_counts.append(layers)
            assigned_layers += layers
        
        # Last device gets remaining layers
        layer_counts.append(num_layers - assigned_layers)
        
        # Create assignments
        assignments = []
        current_layer = 0
        
        for i, (device, num_device_layers) in enumerate(zip(devices, layer_counts)):
            if num_device_layers == 0:
                continue  # Skip devices with no layers
            
            # Calculate memory
            layer_memory = self._estimate_layer_memory(model_info, num_device_layers)
            total_memory = layer_memory
            
            if i == 0:
                total_memory += self._estimate_embedding_memory(model_info)
            if i == len(devices) - 1:
                total_memory += self._estimate_head_memory(model_info)
            
            assignment = ShardAssignment(
                device_id=device.device_id,
                device_profile=device,
                start_layer=current_layer,
                end_layer=current_layer + num_device_layers,
                num_layers=num_device_layers,
                estimated_memory_gb=total_memory,
                estimated_compute_load=num_device_layers / num_layers,
                has_embedding=(i == 0),
                has_lm_head=(i == len(devices) - 1)
            )
            assignments.append(assignment)
            
            current_layer += num_device_layers
        
        return assignments
    
    def _custom_sharding(self, model_info: ModelInfo,
                        devices: List[DeviceProfile],
                        proportions: List[float]) -> List[ShardAssignment]:
        """Create custom sharding based on provided proportions."""
        if len(proportions) != len(devices):
            raise ValueError(f"Proportions length {len(proportions)} != devices {len(devices)}")
        
        if abs(sum(proportions) - 1.0) > 0.001:
            raise ValueError(f"Proportions must sum to 1.0, got {sum(proportions)}")
        
        return self._proportional_sharding(model_info, devices, proportions)
    
    def _estimate_layer_memory(self, model_info: ModelInfo, num_layers: int) -> float:
        """Estimate memory required for transformer layers."""
        # Estimate based on model architecture
        total_model_gb = model_info.estimate_size_gb()
        
        # Assume 85% of model size is in transformer layers
        layer_fraction = 0.85
        layer_memory_gb = (total_model_gb * layer_fraction * num_layers) / model_info.num_layers
        
        return layer_memory_gb
    
    def _estimate_embedding_memory(self, model_info: ModelInfo) -> float:
        """Estimate memory for embedding layer."""
        # Embedding size = vocab_size * hidden_size * bytes_per_param
        bytes_per_param = 1 if 'int8' in str(model_info.quantization) else 2
        embedding_gb = (model_info.vocab_size * model_info.hidden_size * bytes_per_param) / (1024**3)
        
        return embedding_gb
    
    def _estimate_head_memory(self, model_info: ModelInfo) -> float:
        """Estimate memory for language model head."""
        # LM head size = hidden_size * vocab_size * bytes_per_param
        bytes_per_param = 1 if 'int8' in str(model_info.quantization) else 2
        head_gb = (model_info.hidden_size * model_info.vocab_size * bytes_per_param) / (1024**3)
        
        return head_gb
    
    def _calculate_balance_score(self, assignments: List[ShardAssignment]) -> float:
        """Calculate how well balanced the sharding is (0-1, higher is better)."""
        if not assignments:
            return 0.0
        
        # Calculate memory utilization variance
        utilizations = [a.memory_utilization() for a in assignments]
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        
        # Lower variance is better, normalize to 0-1
        memory_balance = 1.0 - min(std_util / mean_util, 1.0) if mean_util > 0 else 0.0
        
        # Calculate compute load variance
        compute_loads = [a.estimated_compute_load for a in assignments]
        mean_load = np.mean(compute_loads)
        std_load = np.std(compute_loads)
        
        compute_balance = 1.0 - min(std_load / mean_load, 1.0) if mean_load > 0 else 0.0
        
        # Combined score (equal weight)
        return (memory_balance + compute_balance) / 2
    
    def find_optimal_strategy(self, model_info: ModelInfo,
                            devices: List[DeviceProfile]) -> Tuple[ShardingPlan, Dict[str, ShardingPlan]]:
        """Find the optimal sharding strategy for given model and devices.
        
        Returns:
            (best_plan, all_plans_dict)
        """
        all_plans = {}
        
        # Try each strategy
        for strategy in [ShardingStrategy.UNIFORM, ShardingStrategy.MEMORY_PROPORTIONAL,
                        ShardingStrategy.COMPUTE_PROPORTIONAL, ShardingStrategy.BALANCED]:
            try:
                plan = self.create_plan(model_info, devices, strategy)
                all_plans[strategy.value] = plan
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
        
        if not all_plans:
            raise ValueError("No valid sharding strategy found")
        
        # Find best plan based on balance score and memory utilization
        best_plan = max(all_plans.values(), 
                       key=lambda p: (p.balance_score, -p.max_memory_utilization))
        
        return best_plan, all_plans


if __name__ == "__main__":
    # Test sharding strategies
    from model_abstraction import ModelFactory
    
    # Create mock devices
    devices = [
        DeviceProfile(
            device_id="mini1",
            hostname="mini1.local",
            model="Apple M4",
            memory_gb=16.0,
            gpu_cores=10,
            cpu_cores=10,
            cpu_performance_cores=4,
            cpu_efficiency_cores=6,
            neural_engine_cores=16
        ),
        DeviceProfile(
            device_id="studio1",
            hostname="studio1.local", 
            model="Apple M4 Max",
            memory_gb=48.0,
            gpu_cores=40,
            cpu_cores=16,
            cpu_performance_cores=12,
            cpu_efficiency_cores=4,
            neural_engine_cores=16
        )
    ]
    
    # Load model info
    model_name = "mlx-community/Qwen3-1.7B-8bit"
    wrapper = ModelFactory.create_wrapper(model_name)
    wrapper.load_model()
    model_info = wrapper.model_info
    
    # Create planner
    planner = ResourceAwareShardingPlanner()
    
    # Find optimal strategy
    print("Finding optimal sharding strategy...")
    best_plan, all_plans = planner.find_optimal_strategy(model_info, devices)
    
    # Print all strategies
    for strategy_name, plan in all_plans.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        plan.print_summary()
    
    print(f"\n{'='*60}")
    print(f"BEST STRATEGY: {best_plan.strategy.value}")
    print(f"{'='*60}")
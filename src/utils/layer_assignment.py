"""Dynamic layer assignment algorithms for heterogeneous sharding."""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from .device_capability import DeviceCapability, estimate_memory_per_layer

logger = logging.getLogger(__name__)


@dataclass
class LayerAssignment:
    """Represents layer assignment for a device."""
    device_id: str
    rank: int
    start_layer: int
    end_layer: int
    layer_indices: List[int]
    has_embeddings: bool = False
    has_lm_head: bool = False
    
    @property
    def num_layers(self) -> int:
        """Total number of layers assigned."""
        return len(self.layer_indices)
    
    def __str__(self) -> str:
        parts = [f"Device {self.device_id} (rank {self.rank}): layers {self.start_layer}-{self.end_layer}"]
        if self.has_embeddings:
            parts.append("+ embeddings")
        if self.has_lm_head:
            parts.append("+ lm_head")
        return " ".join(parts)


class LayerAssignmentStrategy:
    """Base class for layer assignment strategies."""
    
    def assign_layers(
        self,
        devices: List[DeviceCapability],
        total_layers: int,
        model_name: str = None
    ) -> Dict[str, LayerAssignment]:
        """Assign layers to devices based on strategy."""
        raise NotImplementedError


class EqualAssignmentStrategy(LayerAssignmentStrategy):
    """Assign equal number of layers to each device."""
    
    def assign_layers(
        self,
        devices: List[DeviceCapability],
        total_layers: int,
        model_name: str = None
    ) -> Dict[str, LayerAssignment]:
        """Equally distribute layers across devices."""
        assignments = {}
        layers_per_device = total_layers // len(devices)
        remainder = total_layers % len(devices)
        
        current_layer = 0
        for rank, device in enumerate(devices):
            # Add one extra layer to first 'remainder' devices
            num_layers = layers_per_device + (1 if rank < remainder else 0)
            
            layer_indices = list(range(current_layer, current_layer + num_layers))
            
            assignments[device.device_id] = LayerAssignment(
                device_id=device.device_id,
                rank=rank,
                start_layer=current_layer,
                end_layer=current_layer + num_layers - 1,
                layer_indices=layer_indices,
                has_embeddings=(rank == 0),
                has_lm_head=(rank == len(devices) - 1)
            )
            
            current_layer += num_layers
            
        return assignments


class CapabilityBasedAssignmentStrategy(LayerAssignmentStrategy):
    """Assign layers proportional to device capabilities."""
    
    def __init__(self, min_layers_per_device: int = 2):
        """
        Initialize strategy.
        
        Args:
            min_layers_per_device: Minimum layers per device to ensure participation
        """
        self.min_layers_per_device = min_layers_per_device
    
    def assign_layers(
        self,
        devices: List[DeviceCapability],
        total_layers: int,
        model_name: str = None
    ) -> Dict[str, LayerAssignment]:
        """Assign layers based on device compute scores."""
        # Sort devices by rank (assumed to be in order)
        devices = list(devices)  # Copy to avoid modifying
        
        # Calculate total compute power
        total_score = sum(d.compute_score for d in devices)
        
        # Ensure minimum viable assignment
        if total_layers < len(devices) * self.min_layers_per_device:
            logger.warning(
                f"Too few layers ({total_layers}) for {len(devices)} devices. "
                f"Falling back to equal assignment."
            )
            return EqualAssignmentStrategy().assign_layers(devices, total_layers, model_name)
        
        # Calculate proportional assignment
        assignments = {}
        assigned_layers = 0
        current_layer = 0
        
        for rank, device in enumerate(devices):
            if rank == len(devices) - 1:
                # Last device gets remaining layers
                num_layers = total_layers - assigned_layers
            else:
                # Proportional assignment based on compute score
                proportion = device.compute_score / total_score
                ideal_layers = int(total_layers * proportion)
                
                # Ensure minimum layers
                num_layers = max(ideal_layers, self.min_layers_per_device)
                
                # Don't exceed remaining layers
                num_layers = min(num_layers, total_layers - assigned_layers - 
                               (len(devices) - rank - 1) * self.min_layers_per_device)
            
            layer_indices = list(range(current_layer, current_layer + num_layers))
            
            assignments[device.device_id] = LayerAssignment(
                device_id=device.device_id,
                rank=rank,
                start_layer=current_layer,
                end_layer=current_layer + num_layers - 1,
                layer_indices=layer_indices,
                has_embeddings=(rank == 0),
                has_lm_head=(rank == len(devices) - 1)
            )
            
            assigned_layers += num_layers
            current_layer += num_layers
            
            logger.info(f"{device.device_id}: {num_layers} layers "
                       f"(score: {device.compute_score}, {proportion*100:.1f}%)")
        
        return assignments


class MemoryConstrainedAssignmentStrategy(LayerAssignmentStrategy):
    """Assign layers respecting memory constraints."""
    
    def __init__(self, safety_factor: float = 0.8):
        """
        Initialize strategy.
        
        Args:
            safety_factor: Use only this fraction of available memory
        """
        self.safety_factor = safety_factor
    
    def assign_layers(
        self,
        devices: List[DeviceCapability],
        total_layers: int,
        model_name: str = None
    ) -> Dict[str, LayerAssignment]:
        """Assign layers ensuring they fit in device memory."""
        if not model_name:
            logger.warning("No model name provided, using capability-based assignment")
            return CapabilityBasedAssignmentStrategy().assign_layers(
                devices, total_layers, model_name
            )
        
        # Estimate memory per layer
        memory_per_layer = estimate_memory_per_layer(model_name)
        logger.info(f"Estimated memory per layer: {memory_per_layer:.2f} GB")
        
        # Calculate max layers per device based on memory
        max_layers_per_device = []
        for device in devices:
            available_memory = device.gpu_memory_gb * self.safety_factor
            max_layers = int(available_memory / memory_per_layer)
            max_layers_per_device.append(max_layers)
            logger.info(f"{device.device_id}: max {max_layers} layers "
                       f"({available_memory:.1f}GB available)")
        
        # Check if total assignment is possible
        if sum(max_layers_per_device) < total_layers:
            raise ValueError(
                f"Cannot fit {total_layers} layers across devices. "
                f"Max capacity: {sum(max_layers_per_device)} layers"
            )
        
        # Assign layers respecting memory constraints
        assignments = {}
        assigned_layers = 0
        current_layer = 0
        
        for rank, (device, max_layers) in enumerate(zip(devices, max_layers_per_device)):
            if rank == len(devices) - 1:
                # Last device gets remaining layers
                num_layers = total_layers - assigned_layers
            else:
                # Use capability-based proportion, but cap at memory limit
                proportion = device.compute_score / sum(d.compute_score for d in devices)
                ideal_layers = int(total_layers * proportion)
                num_layers = min(ideal_layers, max_layers)
                
                # Ensure we can still assign remaining layers
                remaining_devices = len(devices) - rank - 1
                remaining_capacity = sum(max_layers_per_device[rank+1:])
                remaining_needed = total_layers - assigned_layers - num_layers
                
                if remaining_needed > remaining_capacity:
                    # Reduce this device's assignment
                    reduction = remaining_needed - remaining_capacity
                    num_layers = max(1, num_layers - reduction)
            
            layer_indices = list(range(current_layer, current_layer + num_layers))
            
            assignments[device.device_id] = LayerAssignment(
                device_id=device.device_id,
                rank=rank,
                start_layer=current_layer,
                end_layer=current_layer + num_layers - 1,
                layer_indices=layer_indices,
                has_embeddings=(rank == 0),
                has_lm_head=(rank == len(devices) - 1)
            )
            
            assigned_layers += num_layers
            current_layer += num_layers
            
            memory_used = num_layers * memory_per_layer
            logger.info(f"{device.device_id}: {num_layers} layers "
                       f"({memory_used:.1f}GB of {device.gpu_memory_gb:.1f}GB)")
        
        return assignments


class ManualAssignmentStrategy(LayerAssignmentStrategy):
    """Use manually specified layer assignments."""
    
    def __init__(self, manual_assignment: Dict[str, List[int]]):
        """
        Initialize with manual assignment.
        
        Args:
            manual_assignment: Dict mapping device_id to list of layer indices
        """
        self.manual_assignment = manual_assignment
    
    def assign_layers(
        self,
        devices: List[DeviceCapability],
        total_layers: int,
        model_name: str = None
    ) -> Dict[str, LayerAssignment]:
        """Use manual layer assignment."""
        assignments = {}
        
        # Validate manual assignment
        all_layers = set()
        for device_id, layers in self.manual_assignment.items():
            all_layers.update(layers)
        
        if len(all_layers) != total_layers:
            raise ValueError(
                f"Manual assignment covers {len(all_layers)} layers, "
                f"but model has {total_layers} layers"
            )
        
        # Create assignments
        for rank, device in enumerate(devices):
            if device.device_id not in self.manual_assignment:
                raise ValueError(f"No manual assignment for device {device.device_id}")
            
            layer_indices = sorted(self.manual_assignment[device.device_id])
            
            assignments[device.device_id] = LayerAssignment(
                device_id=device.device_id,
                rank=rank,
                start_layer=min(layer_indices),
                end_layer=max(layer_indices),
                layer_indices=layer_indices,
                has_embeddings=(rank == 0),
                has_lm_head=(rank == len(devices) - 1)
            )
        
        return assignments


def calculate_layer_distribution(
    devices: List[DeviceCapability],
    total_layers: int,
    strategy: str = "capability_based",
    model_name: str = None,
    manual_assignment: Dict[str, List[int]] = None
) -> Dict[str, LayerAssignment]:
    """
    Calculate optimal layer distribution for devices.
    
    Args:
        devices: List of device capabilities
        total_layers: Total number of transformer layers
        strategy: Assignment strategy ("equal", "capability_based", "memory_constrained", "manual")
        model_name: Model name for memory estimation
        manual_assignment: Manual layer assignment (for "manual" strategy)
        
    Returns:
        Dictionary mapping device_id to LayerAssignment
    """
    strategies = {
        "equal": EqualAssignmentStrategy(),
        "capability_based": CapabilityBasedAssignmentStrategy(),
        "memory_constrained": MemoryConstrainedAssignmentStrategy(),
        "manual": ManualAssignmentStrategy(manual_assignment) if manual_assignment else None
    }
    
    if strategy not in strategies or (strategy == "manual" and not manual_assignment):
        raise ValueError(f"Invalid strategy: {strategy}")
    
    strategy_impl = strategies[strategy]
    assignments = strategy_impl.assign_layers(devices, total_layers, model_name)
    
    # Log assignment summary
    logger.info(f"\nLayer assignment using {strategy} strategy:")
    for device_id, assignment in assignments.items():
        logger.info(f"  {assignment}")
    
    return assignments


def validate_assignment(
    assignments: Dict[str, LayerAssignment],
    total_layers: int
) -> bool:
    """Validate that layer assignment covers all layers exactly once."""
    all_layers = set()
    
    for assignment in assignments.values():
        layer_set = set(assignment.layer_indices)
        
        # Check for duplicates
        if all_layers & layer_set:
            logger.error(f"Duplicate layers found: {all_layers & layer_set}")
            return False
        
        all_layers.update(layer_set)
    
    # Check completeness
    expected_layers = set(range(total_layers))
    if all_layers != expected_layers:
        missing = expected_layers - all_layers
        extra = all_layers - expected_layers
        if missing:
            logger.error(f"Missing layers: {missing}")
        if extra:
            logger.error(f"Extra layers: {extra}")
        return False
    
    return True
"""Device capability profiler for heterogeneous model sharding."""
import psutil
import subprocess
import torch
import platform
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceCapability:
    """Represents the computational capabilities of a device."""
    
    device_id: str
    hostname: str
    memory_gb: float
    gpu_cores: int
    gpu_memory_gb: float
    bandwidth_gbps: float
    compute_score: float = 0.0
    
    def __post_init__(self):
        """Calculate compute score after initialization."""
        self.compute_score = self._calculate_compute_score()
    
    def _calculate_compute_score(self) -> float:
        """
        Calculate a weighted score representing device capability.
        
        Weights:
        - 40% GPU cores (computational power)
        - 30% GPU memory (model capacity)
        - 20% System memory (workspace)
        - 10% Network bandwidth (communication efficiency)
        """
        # Normalize values (base on typical max values)
        gpu_cores_norm = min(self.gpu_cores / 40, 1.0)  # Max 40 cores
        gpu_memory_norm = min(self.gpu_memory_gb / 128, 1.0)  # Max 128GB
        memory_norm = min(self.memory_gb / 64, 1.0)  # Max 64GB
        bandwidth_norm = min(self.bandwidth_gbps / 40, 1.0)  # Max 40Gbps
        
        score = (
            gpu_cores_norm * 0.4 +
            gpu_memory_norm * 0.3 +
            memory_norm * 0.2 +
            bandwidth_norm * 0.1
        ) * 100  # Scale to 0-100
        
        return round(score, 2)
    
    def can_fit_layers(self, memory_per_layer_gb: float, num_layers: int) -> bool:
        """Check if device can fit the specified number of layers."""
        required_memory = memory_per_layer_gb * num_layers
        # Use 80% of available GPU memory for safety
        available_memory = self.gpu_memory_gb * 0.8
        return required_memory <= available_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'device_id': self.device_id,
            'hostname': self.hostname,
            'memory_gb': self.memory_gb,
            'gpu_cores': self.gpu_cores,
            'gpu_memory_gb': self.gpu_memory_gb,
            'bandwidth_gbps': self.bandwidth_gbps,
            'compute_score': self.compute_score
        }


class DeviceProfiler:
    """Profile device capabilities for heterogeneous computing."""
    
    @staticmethod
    def profile_local_device(device_id: str = None, hostname: str = None) -> DeviceCapability:
        """Profile the local device's capabilities."""
        if device_id is None:
            device_id = platform.node()
        if hostname is None:
            hostname = platform.node()
        
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Get GPU information
        gpu_cores, gpu_memory_gb = DeviceProfiler._get_gpu_info()
        
        # Estimate network bandwidth (default to 10Gbps for Thunderbolt)
        bandwidth_gbps = DeviceProfiler._estimate_bandwidth()
        
        return DeviceCapability(
            device_id=device_id,
            hostname=hostname,
            memory_gb=round(memory_gb, 1),
            gpu_cores=gpu_cores,
            gpu_memory_gb=round(gpu_memory_gb, 1),
            bandwidth_gbps=bandwidth_gbps
        )
    
    @staticmethod
    def _get_gpu_info() -> tuple[int, float]:
        """Get GPU cores and memory for Apple Silicon."""
        try:
            if platform.system() == "Darwin":
                # Get GPU cores from system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True
                )
                
                gpu_cores = 10  # Default for M-series
                
                # Parse output for GPU cores
                for line in result.stdout.split('\n'):
                    if 'Total Number of Cores:' in line:
                        cores = int(line.split(':')[1].strip())
                        gpu_cores = cores
                        break
                
                # Get shared memory (approximate GPU memory)
                # For Apple Silicon, GPU shares system memory
                total_memory = psutil.virtual_memory().total / (1024**3)
                
                # Heuristic: GPU can use up to 75% of system memory
                gpu_memory_gb = total_memory * 0.75
                
                # Adjust based on known configurations
                if gpu_cores >= 30:  # M1/M2 Max
                    gpu_memory_gb = min(gpu_memory_gb, 64)
                elif gpu_cores >= 16:  # M1/M2 Pro
                    gpu_memory_gb = min(gpu_memory_gb, 32)
                else:  # Base M1/M2
                    gpu_memory_gb = min(gpu_memory_gb, 16)
                
                return gpu_cores, gpu_memory_gb
                
            else:
                # For non-Mac systems, check CUDA
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # Estimate cores based on GPU model
                    return 1000, gpu_memory  # Placeholder
                else:
                    return 0, 0
                    
        except Exception as e:
            logger.warning(f"Could not detect GPU info: {e}")
            return 10, 8.0  # Default values
    
    @staticmethod
    def _estimate_bandwidth() -> float:
        """Estimate network bandwidth between devices."""
        # For now, use conservative estimates
        # TODO: Implement actual bandwidth testing
        return 10.0  # 10 Gbps for Thunderbolt
    
    @staticmethod
    def profile_cluster(hostnames: List[str]) -> Dict[str, DeviceCapability]:
        """
        Profile all devices in the cluster.
        
        Args:
            hostnames: List of hostnames to profile
            
        Returns:
            Dictionary mapping hostname to DeviceCapability
        """
        capabilities = {}
        
        for hostname in hostnames:
            if hostname == platform.node():
                # Local device
                cap = DeviceProfiler.profile_local_device(
                    device_id=hostname,
                    hostname=hostname
                )
            else:
                # Remote device - would need SSH implementation
                # For now, use predefined profiles
                cap = DeviceProfiler._get_predefined_profile(hostname)
            
            capabilities[hostname] = cap
            logger.info(f"Profiled {hostname}: {cap.compute_score} score, "
                       f"{cap.gpu_cores} GPU cores, {cap.gpu_memory_gb}GB GPU memory")
        
        return capabilities
    
    @staticmethod
    def _get_predefined_profile(hostname: str) -> DeviceCapability:
        """Get predefined profiles for known devices."""
        profiles = {
            'mini1': DeviceCapability(
                device_id='mini1',
                hostname='mini1',
                memory_gb=16.0,
                gpu_cores=10,
                gpu_memory_gb=12.0,  # 75% of 16GB
                bandwidth_gbps=10.0
            ),
            'mini2': DeviceCapability(
                device_id='mini2',
                hostname='mini2',
                memory_gb=16.0,
                gpu_cores=10,
                gpu_memory_gb=12.0,
                bandwidth_gbps=10.0
            ),
            'master': DeviceCapability(
                device_id='master',
                hostname='master',
                memory_gb=48.0,
                gpu_cores=30,
                gpu_memory_gb=36.0,  # 75% of 48GB
                bandwidth_gbps=10.0
            ),
            'macbook-pro': DeviceCapability(
                device_id='macbook-pro',
                hostname='macbook-pro',
                memory_gb=48.0,
                gpu_cores=30,
                gpu_memory_gb=36.0,
                bandwidth_gbps=10.0
            )
        }
        
        return profiles.get(hostname, DeviceCapability(
            device_id=hostname,
            hostname=hostname,
            memory_gb=16.0,
            gpu_cores=10,
            gpu_memory_gb=12.0,
            bandwidth_gbps=10.0
        ))


def estimate_memory_per_layer(model_name: str, dtype=torch.float16) -> float:
    """
    Estimate memory requirements per transformer layer.
    
    Args:
        model_name: Name of the model
        dtype: Data type for parameters
        
    Returns:
        Estimated memory in GB per layer
    """
    # Parameters per layer for common models
    params_per_layer = {
        # Small models
        "microsoft/phi-2": 40_000_000,  # ~40M params per layer
        "Qwen3-1.7B": 60_000_000,       # ~60M params per layer
        
        # Medium models
        "meta-llama/Llama-2-7b": 200_000_000,   # ~200M params per layer
        "mistralai/Mistral-7B": 200_000_000,    # ~200M params per layer
        
        # Large models
        "meta-llama/Llama-2-13b": 400_000_000,  # ~400M params per layer
        "meta-llama/Llama-2-70b": 2_000_000_000, # ~2B params per layer
    }
    
    # Check for partial matches
    base_model = None
    for key in params_per_layer:
        if key in model_name or model_name in key:
            base_model = key
            break
    
    # Default based on model size hint in name
    if base_model is None:
        if "70b" in model_name.lower():
            params = 2_000_000_000
        elif "13b" in model_name.lower():
            params = 400_000_000
        elif "7b" in model_name.lower():
            params = 200_000_000
        elif "3b" in model_name.lower():
            params = 100_000_000
        elif "1b" in model_name.lower() or "2b" in model_name.lower():
            params = 60_000_000
        else:
            params = 100_000_000  # Default
    else:
        params = params_per_layer[base_model]
    
    # Calculate memory
    bytes_per_param = 2 if dtype == torch.float16 else 4
    memory_gb = params * bytes_per_param / (1024**3)
    
    # Add overhead for activations and KV cache (roughly 50% more)
    memory_gb *= 1.5
    
    return round(memory_gb, 2)
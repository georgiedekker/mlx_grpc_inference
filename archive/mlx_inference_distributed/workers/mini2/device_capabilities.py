"""
Device capability detection for heterogeneous Apple Silicon devices.

This module provides functionality to detect and profile device capabilities
including memory, GPU cores, CPU cores, and network bandwidth.
"""

import subprocess
import re
import mlx.core as mx
import psutil
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceProfile:
    """Complete profile of a device's capabilities."""
    device_id: str
    hostname: str
    model: str  # e.g., "M4", "M4 Pro", "M2 Ultra"
    memory_gb: float
    gpu_cores: int
    cpu_cores: int
    cpu_performance_cores: int
    cpu_efficiency_cores: int
    neural_engine_cores: int
    bandwidth_gbps: Optional[float] = None
    mlx_metal_available: bool = True
    max_recommended_model_size_gb: float = 0.0
    
    def __post_init__(self):
        # Calculate recommended model size (leave 20% memory for system/cache)
        self.max_recommended_model_size_gb = self.memory_gb * 0.8
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DeviceProfile":
        return cls(**data)


class DeviceCapabilityDetector:
    """Detects and profiles Apple Silicon device capabilities."""
    
    def __init__(self):
        self.hostname = socket.gethostname()
        self.device_id = self.hostname.split('.')[0]  # Remove .local suffix
    
    def detect_capabilities(self) -> DeviceProfile:
        """Detect all device capabilities."""
        logger.info(f"Detecting capabilities for device: {self.device_id}")
        
        # Get system info
        hw_info = self._get_hardware_info()
        memory_gb = self._get_memory_info()
        cpu_info = self._get_cpu_info()
        
        # Check MLX/Metal availability
        mlx_available = self._check_mlx_metal()
        
        profile = DeviceProfile(
            device_id=self.device_id,
            hostname=self.hostname,
            model=hw_info['chip'],
            memory_gb=memory_gb,
            gpu_cores=hw_info['gpu_cores'],
            cpu_cores=cpu_info['total_cores'],
            cpu_performance_cores=cpu_info['performance_cores'],
            cpu_efficiency_cores=cpu_info['efficiency_cores'],
            neural_engine_cores=hw_info['neural_engine_cores'],
            mlx_metal_available=mlx_available
        )
        
        logger.info(f"Detected profile: {profile}")
        return profile
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information using system_profiler."""
        try:
            # Get hardware data
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType', '-json'],
                capture_output=True,
                text=True,
                check=True
            )
            data = json.loads(result.stdout)
            hardware = data['SPHardwareDataType'][0]
            
            # Extract chip model
            chip_name = hardware.get('chip_type', 'Unknown')
            
            # Get GPU core count based on chip model
            gpu_cores = self._estimate_gpu_cores(chip_name)
            
            # Neural Engine cores (16 for all Apple Silicon so far)
            neural_cores = 16
            
            return {
                'chip': chip_name,
                'gpu_cores': gpu_cores,
                'neural_engine_cores': neural_cores
            }
            
        except Exception as e:
            logger.error(f"Error getting hardware info: {e}")
            # Fallback values
            return {
                'chip': 'Unknown',
                'gpu_cores': 10,  # Conservative estimate
                'neural_engine_cores': 16
            }
    
    def _estimate_gpu_cores(self, chip_name: str) -> int:
        """Estimate GPU core count based on chip model."""
        # Known configurations (as of 2024)
        gpu_core_map = {
            # M4 series
            'Apple M4': 10,
            'Apple M4 Pro': 20,  # Can be 16 or 20
            'Apple M4 Max': 40,  # Can be 32 or 40
            'Apple M4 Ultra': 80,  # Estimated
            
            # M3 series
            'Apple M3': 10,
            'Apple M3 Pro': 18,  # Can be 14 or 18
            'Apple M3 Max': 40,  # Can be 30 or 40
            'Apple M3 Ultra': 80,  # Estimated
            
            # M2 series
            'Apple M2': 10,
            'Apple M2 Pro': 19,  # Can be 16 or 19
            'Apple M2 Max': 38,  # Can be 30 or 38
            'Apple M2 Ultra': 76,
            
            # M1 series
            'Apple M1': 8,
            'Apple M1 Pro': 16,  # Can be 14 or 16
            'Apple M1 Max': 32,  # Can be 24 or 32
            'Apple M1 Ultra': 64,
        }
        
        # Try exact match first
        for chip, cores in gpu_core_map.items():
            if chip in chip_name or chip_name in chip:
                return cores
        
        # Try to extract series and variant
        if 'Ultra' in chip_name:
            return 76  # Conservative Ultra estimate
        elif 'Max' in chip_name:
            return 38  # Conservative Max estimate
        elif 'Pro' in chip_name:
            return 16  # Conservative Pro estimate
        else:
            return 10  # Base model estimate
    
    def _get_memory_info(self) -> float:
        """Get system memory in GB."""
        try:
            # Use psutil for cross-platform compatibility
            total_memory = psutil.virtual_memory().total
            return round(total_memory / (1024**3), 1)
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return 16.0  # Conservative fallback
    
    def _get_cpu_info(self) -> Dict:
        """Get CPU core information."""
        try:
            # Get CPU info from sysctl
            total_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.ncpu']).strip())
            
            # Try to get performance/efficiency core split
            try:
                perf_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.physicalcpu']).strip())
                eff_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel1.physicalcpu']).strip())
            except:
                # Estimate based on total cores
                if total_cores >= 20:  # Ultra
                    perf_cores = 16
                    eff_cores = total_cores - 16
                elif total_cores >= 12:  # Max/Pro
                    perf_cores = 8
                    eff_cores = total_cores - 8
                else:  # Base
                    perf_cores = 4
                    eff_cores = total_cores - 4
            
            return {
                'total_cores': total_cores,
                'performance_cores': perf_cores,
                'efficiency_cores': eff_cores
            }
            
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {
                'total_cores': 8,
                'performance_cores': 4,
                'efficiency_cores': 4
            }
    
    def _check_mlx_metal(self) -> bool:
        """Check if MLX can use Metal acceleration."""
        try:
            # Try to create a simple MLX array on GPU
            test_array = mx.array([1.0, 2.0, 3.0])
            mx.eval(test_array)
            return True
        except Exception as e:
            logger.warning(f"MLX Metal check failed: {e}")
            return False
    
    def measure_bandwidth_to_host(self, remote_host: str, port: int = 5001, 
                                 duration: float = 5.0) -> Optional[float]:
        """Measure network bandwidth to a remote host.
        
        Args:
            remote_host: Hostname or IP of remote device
            port: Port to use for bandwidth test
            duration: Test duration in seconds
            
        Returns:
            Bandwidth in Gbps or None if test fails
        """
        logger.info(f"Measuring bandwidth to {remote_host}:{port}")
        
        try:
            # Simple bandwidth test using socket
            # In production, consider using iperf3 or similar
            import socket
            import time
            
            # Create test data (100MB)
            test_data = b'x' * (100 * 1024 * 1024)
            
            # Connect to remote host
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((remote_host, port))
            
            # Send data and measure time
            start_time = time.time()
            total_sent = 0
            
            while time.time() - start_time < duration:
                sent = sock.send(test_data)
                total_sent += sent
            
            elapsed = time.time() - start_time
            sock.close()
            
            # Calculate bandwidth in Gbps
            bandwidth_mbps = (total_sent * 8) / (elapsed * 1024 * 1024)
            bandwidth_gbps = bandwidth_mbps / 1000
            
            logger.info(f"Measured bandwidth: {bandwidth_gbps:.2f} Gbps")
            return bandwidth_gbps
            
        except Exception as e:
            logger.error(f"Bandwidth measurement failed: {e}")
            return None
    
    def save_profile(self, profile: DeviceProfile, path: str):
        """Save device profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        logger.info(f"Saved device profile to {path}")
    
    def load_profile(self, path: str) -> Optional[DeviceProfile]:
        """Load device profile from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return DeviceProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return None


def compare_devices(devices: List[DeviceProfile]) -> Dict:
    """Compare capabilities across multiple devices.
    
    Returns analysis including:
    - Total resources
    - Recommended lead device
    - Heterogeneity metrics
    """
    if not devices:
        return {}
    
    total_memory = sum(d.memory_gb for d in devices)
    total_gpu_cores = sum(d.gpu_cores for d in devices)
    total_cpu_cores = sum(d.cpu_cores for d in devices)
    
    # Find most capable device (for coordinator role)
    lead_device = max(devices, key=lambda d: (d.memory_gb, d.gpu_cores))
    
    # Calculate heterogeneity
    memory_variance = max(d.memory_gb for d in devices) / min(d.memory_gb for d in devices)
    gpu_variance = max(d.gpu_cores for d in devices) / min(d.gpu_cores for d in devices)
    
    return {
        'total_memory_gb': total_memory,
        'total_gpu_cores': total_gpu_cores,
        'total_cpu_cores': total_cpu_cores,
        'lead_device': lead_device.device_id,
        'memory_heterogeneity': memory_variance,
        'gpu_heterogeneity': gpu_variance,
        'device_count': len(devices),
        'devices': [d.to_dict() for d in devices]
    }


if __name__ == "__main__":
    # Test capability detection
    detector = DeviceCapabilityDetector()
    profile = detector.detect_capabilities()
    
    print("\nDevice Profile:")
    print(json.dumps(profile.to_dict(), indent=2))
    
    # Save profile
    detector.save_profile(profile, f"{profile.device_id}_profile.json")
"""
Configuration management for MLX distributed inference system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import socket
from pathlib import Path


class DeviceRole(Enum):
    """Role of a device in the cluster."""
    COORDINATOR = "coordinator"
    WORKER = "worker"


@dataclass
class DeviceCapabilities:
    """Hardware capabilities of a device."""
    model: str
    memory_gb: int
    gpu_cores: int
    cpu_cores: int
    cpu_performance_cores: int
    cpu_efficiency_cores: int
    neural_engine_cores: int
    bandwidth_gbps: float
    mlx_metal_available: bool = True
    max_recommended_model_size_gb: float = 10.0


@dataclass
class DeviceConfig:
    """Configuration for a single device."""
    device_id: str
    hostname: str
    api_port: int
    grpc_port: int
    role: DeviceRole
    rank: int
    capabilities: DeviceCapabilities
    ssh_user: Optional[str] = None
    ssh_key: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceConfig':
        """Create DeviceConfig from dictionary."""
        caps_data = data.get('capabilities', {})
        capabilities = DeviceCapabilities(**caps_data)
        
        return cls(
            device_id=data['device_id'],
            hostname=data['hostname'],
            api_port=data['api_port'],
            grpc_port=data['grpc_port'],
            role=DeviceRole(data['role']),
            rank=data['rank'],
            capabilities=capabilities,
            ssh_user=data.get('ssh_user'),
            ssh_key=data.get('ssh_key')
        )


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    total_layers: int
    layer_distribution: Dict[str, List[int]]

    def get_device_layers(self, device_id: str) -> List[int]:
        """Get layer indices assigned to a device."""
        return self.layer_distribution.get(device_id, [])


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    batch_size: int = 1
    max_sequence_length: int = 2048
    tensor_compression: bool = False
    compression_algorithm: str = "lz4"
    connection_pool_size: int = 5
    request_timeout_seconds: int = 60
    heartbeat_interval_seconds: int = 5
    enable_kv_cache: bool = False
    parallel_worker_processing: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_gpu_monitoring: bool = True
    enable_metrics_export: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"


@dataclass
class ClusterConfig:
    """Main cluster configuration."""
    name: str
    coordinator_device_id: str
    communication_backend: str
    devices: List[DeviceConfig]
    model: ModelConfig
    performance: PerformanceConfig
    monitoring: MonitoringConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ClusterConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse devices
        devices = []
        for device_data in data['devices']:
            devices.append(DeviceConfig.from_dict(device_data))
        
        # Parse model config
        model = ModelConfig(
            name=data['model']['name'],
            total_layers=data['model']['total_layers'],
            layer_distribution=data['model']['layer_distribution']
        )
        
        # Parse performance config
        perf_data = data.get('performance', {})
        performance = PerformanceConfig(**perf_data)
        
        # Parse monitoring config
        mon_data = data.get('monitoring', {})
        monitoring = MonitoringConfig(**mon_data)
        
        return cls(
            name=data['cluster']['name'],
            coordinator_device_id=data['cluster']['coordinator_device_id'],
            communication_backend=data['cluster']['communication_backend'],
            devices=devices,
            model=model,
            performance=performance,
            monitoring=monitoring
        )
    
    def get_device(self, device_id: str) -> Optional[DeviceConfig]:
        """Get device configuration by ID."""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_device_by_rank(self, rank: int) -> Optional[DeviceConfig]:
        """Get device configuration by rank."""
        for device in self.devices:
            if device.rank == rank:
                return device
        return None
    
    def get_coordinator(self) -> Optional[DeviceConfig]:
        """Get coordinator device configuration."""
        return self.get_device(self.coordinator_device_id)
    
    def get_workers(self) -> List[DeviceConfig]:
        """Get all worker devices."""
        return [d for d in self.devices if d.role == DeviceRole.WORKER]
    
    def get_local_device_id(self) -> str:
        """Detect local device ID based on hostname."""
        hostname = socket.gethostname().lower()
        
        # Map common hostname patterns to device IDs
        if 'mini1' in hostname:
            return 'mini1'
        elif 'mini2' in hostname:
            return 'mini2'
        elif 'master' in hostname:
            return 'master'
        
        # Try exact match
        for device in self.devices:
            if device.hostname.lower() == hostname:
                return device.device_id
        
        # Default to coordinator
        return self.coordinator_device_id
    
    def get_local_device(self) -> Optional[DeviceConfig]:
        """Get configuration for the local device."""
        return self.get_device(self.get_local_device_id())
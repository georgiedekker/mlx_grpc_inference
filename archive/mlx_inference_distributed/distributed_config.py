"""
Configuration for distributed MLX inference system.

This module handles configuration for both LAN-based and future Thunderbolt ring network setups.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import os
import json


class NetworkType(Enum):
    LAN = "lan"
    THUNDERBOLT = "thunderbolt"
    HYBRID = "hybrid"  # LAN + Thunderbolt


class DeviceRole(Enum):
    MASTER = "master"
    WORKER = "worker"


@dataclass
class DeviceConfig:
    """Configuration for a single device in the distributed system."""
    device_id: str
    hostname: str
    port: int
    role: DeviceRole
    device_index: int  # MPI rank
    capabilities: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {}


@dataclass
class DistributedConfig:
    """Main configuration for the distributed MLX inference system."""
    # Network configuration
    network_type: NetworkType = NetworkType.LAN
    master_hostname: str = "mini1.local"
    master_port: int = 8100
    
    # MPI configuration
    mpi_hosts: List[str] = None
    mpi_slots_per_host: int = 1
    
    # Model configuration
    model_name: str = "mlx-community/Qwen3-1.7B-8bit"
    model_parallel_size: int = 3  # Number of devices for model parallelism
    pipeline_parallel_size: int = 1  # Number of stages for pipeline parallelism
    
    # Communication settings
    communication_backend: str = "grpc"  # Pure gRPC implementation
    heartbeat_interval: float = 5.0  # seconds
    timeout: float = 60.0  # seconds
    
    # Performance settings
    batch_size: int = 1
    prefetch_batches: int = 2
    
    # Device discovery
    auto_discover_devices: bool = True
    device_list: List[DeviceConfig] = None
    
    def __post_init__(self):
        if self.mpi_hosts is None:
            self.mpi_hosts = ["mini1.local", "mini2.local"]
        
        if self.device_list is None:
            self.device_list = [
                DeviceConfig(
                    device_id="mini1",
                    hostname="mini1.local",
                    port=8100,
                    role=DeviceRole.MASTER,
                    device_index=0
                ),
                DeviceConfig(
                    device_id="mini2",
                    hostname="mini2.local",
                    port=8001,
                    role=DeviceRole.WORKER,
                    device_index=1
                )
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "network_type": self.network_type.value,
            "master_hostname": self.master_hostname,
            "master_port": self.master_port,
            "mpi_hosts": self.mpi_hosts,
            "mpi_slots_per_host": self.mpi_slots_per_host,
            "model_name": self.model_name,
            "model_parallel_size": self.model_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "communication_backend": self.communication_backend,
            "heartbeat_interval": self.heartbeat_interval,
            "timeout": self.timeout,
            "batch_size": self.batch_size,
            "prefetch_batches": self.prefetch_batches,
            "auto_discover_devices": self.auto_discover_devices,
            "device_list": [
                {
                    "device_id": d.device_id,
                    "hostname": d.hostname,
                    "port": d.port,
                    "role": d.role.value,
                    "device_index": d.device_index,
                    "capabilities": d.capabilities
                }
                for d in self.device_list
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedConfig":
        """Create configuration from dictionary."""
        config = cls(
            network_type=NetworkType(data.get("network_type", "lan")),
            master_hostname=data.get("master_hostname", "mini1.local"),
            master_port=data.get("master_port", 8100),
            mpi_hosts=data.get("mpi_hosts"),
            mpi_slots_per_host=data.get("mpi_slots_per_host", 1),
            model_name=data.get("model_name", "mlx-community/Qwen3-1.7B-8bit"),
            model_parallel_size=data.get("model_parallel_size", 3),
            pipeline_parallel_size=data.get("pipeline_parallel_size", 1),
            communication_backend=data.get("communication_backend", "grpc"),
            heartbeat_interval=data.get("heartbeat_interval", 5.0),
            timeout=data.get("timeout", 60.0),
            batch_size=data.get("batch_size", 1),
            prefetch_batches=data.get("prefetch_batches", 2),
            auto_discover_devices=data.get("auto_discover_devices", True)
        )
        
        if "device_list" in data:
            config.device_list = [
                DeviceConfig(
                    device_id=d["device_id"],
                    hostname=d["hostname"],
                    port=d["port"],
                    role=DeviceRole(d["role"]),
                    device_index=d["device_index"],
                    capabilities=d.get("capabilities", {})
                )
                for d in data["device_list"]
            ]
        
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DistributedConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_device_by_index(self, index: int) -> Optional[DeviceConfig]:
        """Get device configuration by MPI rank/index."""
        for device in self.device_list:
            if device.device_index == index:
                return device
        return None
    
    def get_master_device(self) -> Optional[DeviceConfig]:
        """Get the master device configuration."""
        for device in self.device_list:
            if device.role == DeviceRole.MASTER:
                return device
        return None
    
    def get_worker_devices(self) -> List[DeviceConfig]:
        """Get all worker device configurations."""
        return [d for d in self.device_list if d.role == DeviceRole.WORKER]


# Default configuration
DEFAULT_CONFIG = DistributedConfig()
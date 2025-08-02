"""
Enhanced configuration system for heterogeneous distributed MLX inference.

This module provides configuration management for the gRPC-based distributed
inference system with support for heterogeneous Apple Silicon devices.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import yaml
import logging
from enum import Enum

from device_capabilities import DeviceProfile
from sharding_strategy import ShardingStrategy

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network configuration for device communication."""
    protocol: str = "grpc"  # Future: could support other protocols
    compression: bool = False
    max_message_size_mb: int = 1024
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000
    connection_timeout_s: float = 30.0


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    provider: str = "mlx-community"  # mlx-community, huggingface, local
    cache_dir: Optional[str] = None
    quantization: Optional[str] = None  # int8, int4, none
    dtype: str = "float16"
    max_sequence_length: Optional[int] = None
    
    def get_full_name(self) -> str:
        """Get full model name including provider prefix if needed."""
        if self.provider == "mlx-community" and not self.name.startswith("mlx-community/"):
            return f"mlx-community/{self.name}"
        return self.name


@dataclass 
class DeviceConfig:
    """Configuration for a single device in the cluster."""
    device_id: str
    hostname: str
    port: int
    role: str = "worker"  # coordinator, worker
    enabled: bool = True
    capabilities: Optional[DeviceProfile] = None
    grpc_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "device_id": self.device_id,
            "hostname": self.hostname,
            "port": self.port,
            "role": self.role,
            "enabled": self.enabled,
            "grpc_options": self.grpc_options
        }
        if self.capabilities:
            data["capabilities"] = self.capabilities.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceConfig":
        """Create from dictionary."""
        capabilities = None
        if "capabilities" in data:
            capabilities = DeviceProfile.from_dict(data["capabilities"])
        
        return cls(
            device_id=data["device_id"],
            hostname=data["hostname"],
            port=data["port"],
            role=data.get("role", "worker"),
            enabled=data.get("enabled", True),
            capabilities=capabilities,
            grpc_options=data.get("grpc_options", {})
        )


@dataclass
class ShardingConfig:
    """Configuration for model sharding."""
    strategy: ShardingStrategy = ShardingStrategy.BALANCED
    custom_proportions: Optional[List[float]] = None
    min_layers_per_device: int = 1
    optimize_for: str = "balance"  # balance, memory, compute, latency
    allow_partial_layers: bool = False  # Future: sub-layer sharding
    cache_shards: bool = True
    cache_dir: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for inference behavior."""
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_repetition_penalty: float = 1.1
    batch_size: int = 1
    enable_streaming: bool = True
    timeout_s: float = 60.0


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and diagnostics."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval_s: float = 10.0
    log_level: str = "INFO"
    log_performance: bool = True
    trace_requests: bool = False


@dataclass
class DistributedConfig:
    """Main configuration for distributed MLX inference system."""
    # Cluster configuration
    devices: List[DeviceConfig]
    coordinator_device_id: Optional[str] = None
    
    # Model configuration
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="Qwen3-1.7B-8bit",
        provider="mlx-community"
    ))
    
    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Sharding configuration
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    
    # Inference configuration
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8100
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Auto-select coordinator if not specified
        if not self.coordinator_device_id and self.devices:
            # Select device with most memory as coordinator
            coordinator = max(self.devices, 
                            key=lambda d: d.capabilities.memory_gb if d.capabilities else 0)
            self.coordinator_device_id = coordinator.device_id
            coordinator.role = "coordinator"
            logger.info(f"Auto-selected {coordinator.device_id} as coordinator")
        
        # Validate device roles
        coordinator_count = sum(1 for d in self.devices if d.role == "coordinator")
        if coordinator_count == 0:
            logger.warning("No coordinator device specified")
        elif coordinator_count > 1:
            logger.warning(f"Multiple coordinator devices found: {coordinator_count}")
    
    def get_enabled_devices(self) -> List[DeviceConfig]:
        """Get list of enabled devices."""
        return [d for d in self.devices if d.enabled]
    
    def get_device(self, device_id: str) -> Optional[DeviceConfig]:
        """Get device by ID."""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_coordinator(self) -> Optional[DeviceConfig]:
        """Get the coordinator device."""
        if self.coordinator_device_id:
            return self.get_device(self.coordinator_device_id)
        
        # Fallback to first device with coordinator role
        for device in self.devices:
            if device.role == "coordinator":
                return device
        return None
    
    def get_workers(self) -> List[DeviceConfig]:
        """Get all worker devices."""
        return [d for d in self.devices if d.role == "worker" and d.enabled]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check devices
        if not self.devices:
            errors.append("No devices configured")
        
        enabled_devices = self.get_enabled_devices()
        if not enabled_devices:
            errors.append("No enabled devices")
        
        # Check unique device IDs
        device_ids = [d.device_id for d in self.devices]
        if len(device_ids) != len(set(device_ids)):
            errors.append("Duplicate device IDs found")
        
        # Check coordinator
        if not self.get_coordinator():
            errors.append("No coordinator device configured")
        
        # Check ports
        for device in self.devices:
            if device.port <= 0 or device.port > 65535:
                errors.append(f"Invalid port {device.port} for device {device.device_id}")
        
        # Check sharding config
        if self.sharding.strategy == ShardingStrategy.CUSTOM:
            if not self.sharding.custom_proportions:
                errors.append("Custom sharding strategy requires custom_proportions")
            elif len(self.sharding.custom_proportions) != len(enabled_devices):
                errors.append("custom_proportions length must match enabled devices")
            elif abs(sum(self.sharding.custom_proportions) - 1.0) > 0.001:
                errors.append("custom_proportions must sum to 1.0")
        
        # Check model
        if not self.model.name:
            errors.append("Model name not specified")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "devices": [d.to_dict() for d in self.devices],
            "coordinator_device_id": self.coordinator_device_id,
            "model": {
                "name": self.model.name,
                "provider": self.model.provider,
                "cache_dir": self.model.cache_dir,
                "quantization": self.model.quantization,
                "dtype": self.model.dtype,
                "max_sequence_length": self.model.max_sequence_length
            },
            "network": {
                "protocol": self.network.protocol,
                "compression": self.network.compression,
                "max_message_size_mb": self.network.max_message_size_mb,
                "keepalive_time_ms": self.network.keepalive_time_ms,
                "keepalive_timeout_ms": self.network.keepalive_timeout_ms,
                "connection_timeout_s": self.network.connection_timeout_s
            },
            "sharding": {
                "strategy": self.sharding.strategy.value,
                "custom_proportions": self.sharding.custom_proportions,
                "min_layers_per_device": self.sharding.min_layers_per_device,
                "optimize_for": self.sharding.optimize_for,
                "allow_partial_layers": self.sharding.allow_partial_layers,
                "cache_shards": self.sharding.cache_shards,
                "cache_dir": self.sharding.cache_dir
            },
            "inference": {
                "default_max_tokens": self.inference.default_max_tokens,
                "default_temperature": self.inference.default_temperature,
                "default_top_p": self.inference.default_top_p,
                "default_repetition_penalty": self.inference.default_repetition_penalty,
                "batch_size": self.inference.batch_size,
                "enable_streaming": self.inference.enable_streaming,
                "timeout_s": self.inference.timeout_s
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
                "health_check_interval_s": self.monitoring.health_check_interval_s,
                "log_level": self.monitoring.log_level,
                "log_performance": self.monitoring.log_performance,
                "trace_requests": self.monitoring.trace_requests
            },
            "api_host": self.api_host,
            "api_port": self.api_port
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedConfig":
        """Create from dictionary."""
        # Parse devices
        devices = [DeviceConfig.from_dict(d) for d in data.get("devices", [])]
        
        # Parse model config
        model_data = data.get("model", {})
        model = ModelConfig(
            name=model_data.get("name", "Qwen3-1.7B-8bit"),
            provider=model_data.get("provider", "mlx-community"),
            cache_dir=model_data.get("cache_dir"),
            quantization=model_data.get("quantization"),
            dtype=model_data.get("dtype", "float16"),
            max_sequence_length=model_data.get("max_sequence_length")
        )
        
        # Parse network config
        net_data = data.get("network", {})
        network = NetworkConfig(
            protocol=net_data.get("protocol", "grpc"),
            compression=net_data.get("compression", False),
            max_message_size_mb=net_data.get("max_message_size_mb", 1024),
            keepalive_time_ms=net_data.get("keepalive_time_ms", 10000),
            keepalive_timeout_ms=net_data.get("keepalive_timeout_ms", 5000),
            connection_timeout_s=net_data.get("connection_timeout_s", 30.0)
        )
        
        # Parse sharding config
        shard_data = data.get("sharding", {})
        sharding = ShardingConfig(
            strategy=ShardingStrategy(shard_data.get("strategy", "balanced")),
            custom_proportions=shard_data.get("custom_proportions"),
            min_layers_per_device=shard_data.get("min_layers_per_device", 1),
            optimize_for=shard_data.get("optimize_for", "balance"),
            allow_partial_layers=shard_data.get("allow_partial_layers", False),
            cache_shards=shard_data.get("cache_shards", True),
            cache_dir=shard_data.get("cache_dir")
        )
        
        # Parse inference config
        inf_data = data.get("inference", {})
        inference = InferenceConfig(
            default_max_tokens=inf_data.get("default_max_tokens", 512),
            default_temperature=inf_data.get("default_temperature", 0.7),
            default_top_p=inf_data.get("default_top_p", 0.9),
            default_repetition_penalty=inf_data.get("default_repetition_penalty", 1.1),
            batch_size=inf_data.get("batch_size", 1),
            enable_streaming=inf_data.get("enable_streaming", True),
            timeout_s=inf_data.get("timeout_s", 60.0)
        )
        
        # Parse monitoring config
        mon_data = data.get("monitoring", {})
        monitoring = MonitoringConfig(
            enable_metrics=mon_data.get("enable_metrics", True),
            metrics_port=mon_data.get("metrics_port", 9090),
            health_check_interval_s=mon_data.get("health_check_interval_s", 10.0),
            log_level=mon_data.get("log_level", "INFO"),
            log_performance=mon_data.get("log_performance", True),
            trace_requests=mon_data.get("trace_requests", False)
        )
        
        return cls(
            devices=devices,
            coordinator_device_id=data.get("coordinator_device_id"),
            model=model,
            network=network,
            sharding=sharding,
            inference=inference,
            monitoring=monitoring,
            api_host=data.get("api_host", "0.0.0.0"),
            api_port=data.get("api_port", 8100)
        )
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {path}")
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved configuration to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DistributedConfig":
        """Load configuration from file (JSON or YAML)."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path_obj.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        config = cls.from_dict(data)
        logger.info(f"Loaded configuration from {path}")
        
        return config


def create_example_config() -> DistributedConfig:
    """Create an example configuration for testing."""
    # Create device configs
    devices = [
        DeviceConfig(
            device_id="mini1",
            hostname="mini1.local",
            port=50051,
            role="coordinator",
            capabilities=DeviceProfile(
                device_id="mini1",
                hostname="mini1.local",
                model="Apple M4",
                memory_gb=16.0,
                gpu_cores=10,
                cpu_cores=10,
                cpu_performance_cores=4,
                cpu_efficiency_cores=6,
                neural_engine_cores=16
            )
        ),
        DeviceConfig(
            device_id="mini2",
            hostname="mini2.local", 
            port=50051,
            role="worker",
            capabilities=DeviceProfile(
                device_id="mini2",
                hostname="mini2.local",
                model="Apple M4",
                memory_gb=16.0,
                gpu_cores=10,
                cpu_cores=10,
                cpu_performance_cores=4,
                cpu_efficiency_cores=6,
                neural_engine_cores=16
            )
        )
    ]
    
    # Create config
    config = DistributedConfig(
        devices=devices,
        coordinator_device_id="mini1",
        model=ModelConfig(
            name="Qwen3-1.7B-8bit",
            provider="mlx-community",
            quantization="int8"
        ),
        sharding=ShardingConfig(
            strategy=ShardingStrategy.BALANCED,
            optimize_for="balance"
        )
    )
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = create_example_config()
    
    # Validate
    is_valid, errors = config.validate()
    print(f"Configuration valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Save to file
    config.save_json("distributed_config_example.json")
    config.save_yaml("distributed_config_example.yaml")
    
    # Load from file
    loaded_config = DistributedConfig.load("distributed_config_example.json")
    print(f"\nLoaded config has {len(loaded_config.devices)} devices")
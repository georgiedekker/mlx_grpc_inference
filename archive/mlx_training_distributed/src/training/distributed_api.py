#!/usr/bin/env python3
"""
Distributed training API for MLX - handles configuration loading and validation
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

from .distributed_trainer import DistributedConfig, DistributedTrainer


class DistributedConfigManager:
    """Manager for distributed training configurations."""
    
    def __init__(self):
        self.config_cache = {}
    
    def load_config(self, config_path: Union[str, Path], config_format: str = "auto") -> DistributedConfig:
        """
        Load distributed configuration from file.
        
        Args:
            config_path: Path to configuration file
            config_format: Format of config file ("json", "yaml", "auto")
        
        Returns:
            DistributedConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Auto-detect format if needed
        if config_format == "auto":
            if config_path.suffix.lower() in ['.json']:
                config_format = "json"
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                config_format = "yaml"
            else:
                raise ValueError(f"Cannot auto-detect format for file: {config_path}")
        
        # Load configuration data
        if config_format == "json":
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_format == "yaml":
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_format}")
        
        # Validate and normalize configuration
        config_data = self._normalize_config(config_data)
        
        # Create DistributedConfig instance
        return DistributedConfig(**config_data)
    
    def _normalize_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration data to match DistributedConfig fields."""
        normalized = {}
        
        # Handle different naming conventions
        field_mappings = {
            # Old naming -> New naming
            'device_list': 'devices',
            'num_devices': 'world_size',
            'num_workers': 'world_size',
            'worker_rank': 'rank',
            'process_rank': 'rank',
            'communication_backend': 'backend',
            'dist_backend': 'backend',
            'master_address': 'master_addr',
            'master_ip': 'master_addr',
            'master_port': 'master_port',
            'port': 'master_port'
        }
        
        # Apply field mappings
        for old_key, new_key in field_mappings.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data.pop(old_key)
        
        # Handle nested configurations
        if 'distributed' in config_data:
            dist_config = config_data.pop('distributed')
            for key, value in dist_config.items():
                if key not in config_data:
                    config_data[key] = value
        
        if 'communication' in config_data:
            comm_config = config_data.pop('communication')
            for key, value in comm_config.items():
                if key not in config_data:
                    config_data[key] = value
        
        if 'optimization' in config_data:
            opt_config = config_data.pop('optimization')
            for key, value in opt_config.items():
                if key not in config_data:
                    config_data[key] = value
        
        # Set default values for required fields
        defaults = {
            'world_size': 1,
            'rank': 0,
            'backend': 'allreduce',
            'master_addr': 'localhost',
            'master_port': 29500,
            'gradient_as_bucket_view': True,
            'find_unused_parameters': False,
            'timeout': 30000,
            'retry_count': 3,
            'overlap_grad_reduce': True,
            'bucket_size_mb': 25
        }
        
        for key, default_value in defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        return config_data
    
    def validate_config(self, config: DistributedConfig) -> List[str]:
        """
        Validate distributed configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate world size
        if config.world_size < 1:
            errors.append("world_size must be >= 1")
        
        # Validate rank
        if config.rank < 0 or config.rank >= config.world_size:
            errors.append(f"rank must be between 0 and {config.world_size - 1}")
        
        # Validate backend
        valid_backends = ["allreduce", "ring_allreduce", "parameter_server"]
        if config.backend not in valid_backends:
            errors.append(f"backend must be one of: {valid_backends}")
        
        # Validate master port
        if not (1024 <= config.master_port <= 65535):
            errors.append("master_port must be between 1024 and 65535")
        
        # Validate devices
        if config.devices:
            if len(config.devices) != config.world_size:
                errors.append(f"Number of devices ({len(config.devices)}) must match world_size ({config.world_size})")
            
            # Check for duplicate devices
            if len(set(config.devices)) != len(config.devices):
                errors.append("Duplicate devices found in device list")
        
        # Validate timeouts and other numeric values
        if config.timeout < 1000:
            errors.append("timeout should be at least 1000 milliseconds")
        
        if config.retry_count < 0:
            errors.append("retry_count must be >= 0")
        
        if config.bucket_size_mb < 1:
            errors.append("bucket_size_mb must be >= 1")
        
        return errors
    
    def save_config(self, config: DistributedConfig, config_path: Union[str, Path], format: str = "json") -> None:
        """Save distributed configuration to file."""
        config_path = Path(config_path)
        config_dict = config.to_dict()
        
        if format == "json":
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format == "yaml":
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_from_template(self, template: str = "default") -> DistributedConfig:
        """Create configuration from a template."""
        templates = {
            "default": {
                "world_size": 1,
                "rank": 0,
                "backend": "allreduce",
                "master_addr": "localhost",
                "master_port": 29500
            },
            "multi_gpu": {
                "world_size": 4,
                "rank": 0,
                "backend": "allreduce",
                "master_addr": "localhost", 
                "master_port": 29500,
                "devices": ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
            },
            "cluster": {
                "world_size": 8,
                "rank": 0,
                "backend": "ring_allreduce",
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "timeout": 60000,
                "retry_count": 5
            }
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template: {template}. Available: {list(templates.keys())}")
        
        config_data = self._normalize_config(templates[template])
        return DistributedConfig(**config_data)


class DistributedTrainingAPI:
    """High-level API for distributed training setup and management."""
    
    def __init__(self):
        self.config_manager = DistributedConfigManager()
        self.active_trainers = {}
    
    def setup_distributed_training(
        self,
        base_trainer,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None
    ) -> DistributedTrainer:
        """
        Setup distributed training with various configuration options.
        
        Args:
            base_trainer: Base trainer instance
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            template: Configuration template name
        
        Returns:
            DistributedTrainer instance
        """
        # Load configuration
        if config_path:
            config = self.config_manager.load_config(config_path)
        elif config_dict:
            config_dict = self.config_manager._normalize_config(config_dict)
            config = DistributedConfig(**config_dict)
        elif template:
            config = self.config_manager.create_from_template(template)
        else:
            config = self.config_manager.create_from_template("default")
        
        # Validate configuration
        errors = self.config_manager.validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        # Create distributed trainer
        trainer = DistributedTrainer(base_trainer, config)
        
        # Store reference
        trainer_id = f"trainer_{len(self.active_trainers)}"
        self.active_trainers[trainer_id] = trainer
        
        return trainer
    
    def get_trainer(self, trainer_id: str) -> Optional[DistributedTrainer]:
        """Get active trainer by ID."""
        return self.active_trainers.get(trainer_id)
    
    def list_active_trainers(self) -> List[str]:
        """List active trainer IDs."""
        return list(self.active_trainers.keys())
    
    def cleanup_trainer(self, trainer_id: str) -> bool:
        """Clean up trainer resources."""
        if trainer_id in self.active_trainers:
            del self.active_trainers[trainer_id]
            return True
        return False


# Global API instance
distributed_api = DistributedTrainingAPI()


def load_distributed_config(config_path: Union[str, Path]) -> DistributedConfig:
    """Convenience function to load distributed configuration."""
    return distributed_api.config_manager.load_config(config_path)


def create_distributed_trainer(base_trainer, config_source: Union[str, Path, Dict[str, Any]]) -> DistributedTrainer:
    """Convenience function to create distributed trainer."""
    if isinstance(config_source, (str, Path)):
        return distributed_api.setup_distributed_training(base_trainer, config_path=config_source)
    elif isinstance(config_source, dict):
        return distributed_api.setup_distributed_training(base_trainer, config_dict=config_source)
    else:
        raise ValueError("config_source must be a file path or dictionary")
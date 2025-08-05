"""Device manager for heterogeneous cluster configuration."""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .device_capability import DeviceCapability, DeviceProfiler
from .layer_assignment import calculate_layer_distribution, LayerAssignment

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device configuration and layer assignments for heterogeneous clusters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DeviceManager.
        
        Args:
            config_path: Path to cluster configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.devices = {}
        self.layer_assignments = {}
        
        # Parse device configurations
        self._parse_devices()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cluster configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        else:
            # Default configuration
            return {
                "cluster": {
                    "name": "heterogeneous-cluster",
                    "devices": ["mini1", "mini2", "master"]
                },
                "model": {
                    "sharding_strategy": "capability_based"
                },
                "communication": {
                    "master_addr": "localhost",
                    "master_port": 29501
                }
            }
    
    def _parse_devices(self):
        """Parse device configurations from config."""
        device_configs = self.config.get('devices', {})
        device_list = self.config.get('cluster', {}).get('devices', [])
        
        for device_name in device_list:
            if device_name in device_configs:
                # Use explicit configuration
                device_config = device_configs[device_name]
                self.devices[device_name] = DeviceCapability(
                    device_id=device_name,
                    hostname=device_config.get('hostname', device_name),
                    memory_gb=device_config.get('memory_gb', 16.0),
                    gpu_cores=device_config.get('gpu_cores', 10),
                    gpu_memory_gb=device_config.get('gpu_memory_gb', 12.0),
                    bandwidth_gbps=device_config.get('bandwidth_gbps', 10.0)
                )
            else:
                # Use profiler or defaults
                if device_name == os.uname().nodename:
                    # Local device - profile it
                    self.devices[device_name] = DeviceProfiler.profile_local_device(
                        device_id=device_name
                    )
                else:
                    # Remote device - use predefined profile
                    self.devices[device_name] = DeviceProfiler._get_predefined_profile(
                        device_name
                    )
        
        logger.info(f"Configured {len(self.devices)} devices")
        for name, device in self.devices.items():
            logger.info(f"  {name}: score={device.compute_score}, "
                       f"gpu={device.gpu_cores} cores, "
                       f"memory={device.gpu_memory_gb}GB")
    
    def get_layer_assignments(
        self,
        model_name: str,
        total_layers: int,
        strategy: Optional[str] = None
    ) -> Dict[str, LayerAssignment]:
        """
        Get layer assignments for the model.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers
            strategy: Override sharding strategy
            
        Returns:
            Dictionary mapping device_id to LayerAssignment
        """
        model_config = self.config.get('model', {})
        
        # Determine strategy
        if strategy is None:
            strategy = model_config.get('sharding_strategy', 'capability_based')
        
        # Check for manual assignment
        manual_assignment = model_config.get('layer_distribution')
        
        # Get ordered device list
        device_list = self.config.get('cluster', {}).get('devices', list(self.devices.keys()))
        ordered_devices = [self.devices[name] for name in device_list if name in self.devices]
        
        # Calculate assignments
        assignments = calculate_layer_distribution(
            devices=ordered_devices,
            total_layers=total_layers,
            strategy=strategy,
            model_name=model_name,
            manual_assignment=manual_assignment
        )
        
        self.layer_assignments = assignments
        return assignments
    
    def get_device_for_rank(self, rank: int) -> Optional[DeviceCapability]:
        """Get device capability for a given rank."""
        device_list = self.config.get('cluster', {}).get('devices', list(self.devices.keys()))
        if rank < len(device_list):
            device_name = device_list[rank]
            return self.devices.get(device_name)
        return None
    
    def get_communication_config(self) -> Dict[str, Any]:
        """Get communication configuration."""
        return self.config.get('communication', {
            'master_addr': 'localhost',
            'master_port': 29501,
            'backend': 'gloo'
        })
    
    def save_config(self, output_path: str):
        """Save current configuration to file."""
        # Include device capabilities in saved config
        config_with_capabilities = self.config.copy()
        config_with_capabilities['devices'] = {
            name: device.to_dict() 
            for name, device in self.devices.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_with_capabilities, f, indent=2)
        logger.info(f"Saved configuration to {output_path}")
    
    def validate_configuration(self, model_name: str, total_layers: int) -> bool:
        """
        Validate that the current configuration can handle the model.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers
            
        Returns:
            True if configuration is valid
        """
        from .device_capability import estimate_memory_per_layer
        
        # Check if we have devices
        if not self.devices:
            logger.error("No devices configured")
            return False
        
        # Estimate memory requirements
        memory_per_layer = estimate_memory_per_layer(model_name)
        total_memory_needed = memory_per_layer * total_layers
        
        # Calculate total available memory
        total_available = sum(
            device.gpu_memory_gb * 0.8  # 80% safety factor
            for device in self.devices.values()
        )
        
        if total_memory_needed > total_available:
            logger.error(
                f"Model requires {total_memory_needed:.1f}GB but only "
                f"{total_available:.1f}GB available across all devices"
            )
            return False
        
        # Try to calculate assignments
        try:
            assignments = self.get_layer_assignments(model_name, total_layers)
            
            # Verify each device can handle its assignment
            for device_name, assignment in assignments.items():
                device = self.devices[device_name]
                assigned_memory = memory_per_layer * assignment.num_layers
                
                if assigned_memory > device.gpu_memory_gb * 0.8:
                    logger.error(
                        f"Device {device_name} assigned {assigned_memory:.1f}GB "
                        f"but only has {device.gpu_memory_gb * 0.8:.1f}GB available"
                    )
                    return False
            
            logger.info("Configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            return False
    
    def suggest_configuration(
        self,
        model_name: str,
        total_layers: int,
        available_devices: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest optimal configuration for given model and devices.
        
        Args:
            model_name: Name of the model
            total_layers: Total number of layers
            available_devices: List of available device names
            
        Returns:
            Suggested configuration dictionary
        """
        from .device_capability import estimate_memory_per_layer
        
        # Profile available devices
        devices = {}
        for device_name in available_devices:
            if device_name == os.uname().nodename:
                devices[device_name] = DeviceProfiler.profile_local_device(device_name)
            else:
                devices[device_name] = DeviceProfiler._get_predefined_profile(device_name)
        
        # Sort by compute score
        sorted_devices = sorted(
            devices.items(),
            key=lambda x: x[1].compute_score,
            reverse=True
        )
        
        # Estimate memory requirements
        memory_per_layer = estimate_memory_per_layer(model_name)
        total_memory_needed = memory_per_layer * total_layers
        
        # Select devices until we have enough memory
        selected_devices = []
        cumulative_memory = 0
        
        for device_name, device in sorted_devices:
            selected_devices.append(device_name)
            cumulative_memory += device.gpu_memory_gb * 0.8
            
            if cumulative_memory >= total_memory_needed:
                break
        
        if cumulative_memory < total_memory_needed:
            logger.warning(
                f"All {len(available_devices)} devices together provide "
                f"{cumulative_memory:.1f}GB but model needs {total_memory_needed:.1f}GB"
            )
        
        # Determine best strategy
        if len(selected_devices) == 1:
            strategy = "equal"  # Single device
        else:
            # Check if devices are homogeneous
            scores = [devices[d].compute_score for d in selected_devices]
            if max(scores) / min(scores) < 1.2:  # Within 20%
                strategy = "equal"
            else:
                strategy = "capability_based"
        
        return {
            "cluster": {
                "name": f"{model_name}-cluster",
                "devices": selected_devices
            },
            "model": {
                "name": model_name,
                "total_layers": total_layers,
                "sharding_strategy": strategy
            },
            "devices": {
                name: devices[name].to_dict()
                for name in selected_devices
            },
            "communication": {
                "master_addr": selected_devices[0],
                "master_port": 29501,
                "backend": "gloo"
            }
        }


def create_heterogeneous_config(
    devices: List[Dict[str, Any]],
    model_name: str,
    output_path: str,
    strategy: str = "capability_based"
):
    """
    Create a heterogeneous cluster configuration file.
    
    Args:
        devices: List of device specifications
        model_name: Model to be deployed
        output_path: Where to save the configuration
        strategy: Sharding strategy to use
    
    Example:
        devices = [
            {"name": "mini1", "memory_gb": 16, "gpu_cores": 10},
            {"name": "mini2", "memory_gb": 16, "gpu_cores": 10},
            {"name": "master", "memory_gb": 48, "gpu_cores": 30}
        ]
    """
    config = {
        "cluster": {
            "name": "heterogeneous-cluster",
            "devices": [d["name"] for d in devices]
        },
        "model": {
            "name": model_name,
            "sharding_strategy": strategy
        },
        "devices": {
            d["name"]: {
                "hostname": d.get("hostname", d["name"]),
                "memory_gb": d.get("memory_gb", 16),
                "gpu_cores": d.get("gpu_cores", 10),
                "gpu_memory_gb": d.get("gpu_memory_gb", d.get("memory_gb", 16) * 0.75),
                "bandwidth_gbps": d.get("bandwidth_gbps", 10.0)
            }
            for d in devices
        },
        "communication": {
            "master_addr": devices[0]["name"],
            "master_port": 29501,
            "backend": "gloo"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created heterogeneous configuration at {output_path}")
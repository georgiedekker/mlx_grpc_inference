#!/usr/bin/env python3
"""
Device Management System for MLX Distributed Inference

Handles dynamic device addition/removal, health monitoring, and coordinator migration.
"""

import asyncio
import logging
import time
import yaml
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.config import ClusterConfig, DeviceConfig, DeviceRole
from src.communication.dns_resolver import resolve_grpc_target
from src.communication import inference_pb2_grpc, inference_pb2
import grpc

logger = logging.getLogger(__name__)

@dataclass
class DeviceStatus:
    """Current status of a device in the cluster."""
    device_id: str
    hostname: str
    role: DeviceRole
    status: str  # "online", "offline", "unhealthy", "initializing"
    last_heartbeat: float
    assigned_layers: List[int]
    capabilities: Dict
    metrics: Dict = None
    
    def is_healthy(self) -> bool:
        """Check if device is healthy (online and responding)."""
        return (
            self.status == "online" and 
            time.time() - self.last_heartbeat < 30  # 30 second timeout
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class DeviceManager:
    """Manages dynamic device addition, removal, and health monitoring."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Optional[ClusterConfig] = None
        self.devices: Dict[str, DeviceStatus] = {}
        self.connections: Dict[str, grpc.Channel] = {}
        self.coordinator_id: Optional[str] = None
        self.is_running = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the device manager."""
        logger.info("🚀 Initializing Device Manager...")
        
        # Load initial configuration
        await self.load_config()
        
        # Discover and register existing devices
        await self.discover_devices()
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.is_running = True
        logger.info("✅ Device Manager initialized")
    
    async def load_config(self):
        """Load cluster configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.config = ClusterConfig.from_dict(config_data)
            self.coordinator_id = self.config.coordinator_device_id
            logger.info(f"✅ Loaded config with {len(self.config.devices)} devices")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    async def discover_devices(self):
        """Discover and connect to devices in the configuration."""
        if not self.config:
            return
            
        for device in self.config.devices:
            await self.add_device(device)
    
    async def add_device(self, device: DeviceConfig) -> bool:
        """Add a new device to the cluster."""
        logger.info(f"🔄 Adding device {device.device_id} ({device.role})...")
        
        try:
            # Create device status
            device_status = DeviceStatus(
                device_id=device.device_id,
                hostname=device.hostname,
                role=device.role,
                status="initializing",
                last_heartbeat=time.time(),
                assigned_layers=[],
                capabilities=asdict(device.capabilities)
            )
            
            # For workers, establish gRPC connection
            if device.role == DeviceRole.WORKER:
                success = await self._connect_to_worker(device)
                if not success:
                    device_status.status = "offline"
                    logger.warning(f"⚠️  Could not connect to worker {device.device_id}")
                else:
                    device_status.status = "online"
                    logger.info(f"✅ Connected to worker {device.device_id}")
            else:
                device_status.status = "online"
            
            # Add to devices registry
            self.devices[device.device_id] = device_status
            
            # Redistribute layers if this is a worker
            if device.role == DeviceRole.WORKER and device_status.is_healthy():
                await self.redistribute_layers()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add device {device.device_id}: {e}")
            return False
    
    async def remove_device(self, device_id: str) -> bool:
        """Remove a device from the cluster."""
        logger.info(f"🗑️  Removing device {device_id}...")
        
        if device_id not in self.devices:
            logger.warning(f"⚠️  Device {device_id} not found")
            return False
        
        try:
            device = self.devices[device_id]
            
            # Close connection if it exists
            if device_id in self.connections:
                await self.connections[device_id].close()
                del self.connections[device_id]
            
            # Remove from registry
            del self.devices[device_id]
            
            # Redistribute layers
            await self.redistribute_layers()
            
            logger.info(f"✅ Removed device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove device {device_id}: {e}")
            return False
    
    async def redistribute_layers(self):
        """Redistribute model layers across healthy workers."""
        logger.info("🔄 Redistributing layers...")
        
        if not self.config:
            return
        
        # Get healthy workers
        healthy_workers = [
            device for device in self.devices.values()
            if device.role == DeviceRole.WORKER and device.is_healthy()
        ]
        
        if not healthy_workers:
            logger.warning("⚠️  No healthy workers available")
            return
        
        # Calculate total layers (excluding coordinator layers)
        coordinator_layers = self.config.model.get_device_layers(self.coordinator_id)
        total_layers = self.config.model.total_layers
        worker_layers = [i for i in range(total_layers) if i not in coordinator_layers]
        
        # Distribute layers among workers
        layers_per_worker = len(worker_layers) // len(healthy_workers)
        remaining_layers = len(worker_layers) % len(healthy_workers)
        
        layer_assignment = {}
        current_layer_idx = 0
        
        for i, worker in enumerate(healthy_workers):
            # Calculate layers for this worker
            worker_layer_count = layers_per_worker
            if i < remaining_layers:  # Distribute remaining layers
                worker_layer_count += 1
            
            # Assign layers
            assigned_layers = worker_layers[current_layer_idx:current_layer_idx + worker_layer_count]
            layer_assignment[worker.device_id] = assigned_layers
            worker.assigned_layers = assigned_layers
            
            current_layer_idx += worker_layer_count
        
        # Update configuration
        self.config.model.layer_distribution.update(layer_assignment)
        
        # Save updated configuration
        await self.save_config()
        
        logger.info(f"✅ Redistributed layers among {len(healthy_workers)} workers")
        for device_id, layers in layer_assignment.items():
            logger.info(f"   {device_id}: layers {layers}")
    
    async def migrate_coordinator(self, new_coordinator_id: str) -> bool:
        """Migrate coordinator role to a different device."""
        logger.info(f"🔄 Migrating coordinator to {new_coordinator_id}...")
        
        if new_coordinator_id not in self.devices:
            logger.error(f"❌ Device {new_coordinator_id} not found")
            return False
        
        new_coordinator = self.devices[new_coordinator_id]
        if not new_coordinator.is_healthy():
            logger.error(f"❌ Device {new_coordinator_id} is not healthy")
            return False
        
        try:
            # Update device roles
            old_coordinator_id = self.coordinator_id
            if old_coordinator_id in self.devices:
                self.devices[old_coordinator_id].role = DeviceRole.WORKER
            
            new_coordinator.role = DeviceRole.COORDINATOR
            self.coordinator_id = new_coordinator_id
            
            # Update configuration
            if self.config:
                self.config.coordinator_device_id = new_coordinator_id
                
                # Update device roles in config
                for device in self.config.devices:
                    if device.device_id == old_coordinator_id:
                        device.role = DeviceRole.WORKER
                    elif device.device_id == new_coordinator_id:
                        device.role = DeviceRole.COORDINATOR
            
            # Redistribute layers
            await self.redistribute_layers()
            
            # Save configuration
            await self.save_config()
            
            logger.info(f"✅ Coordinator migrated from {old_coordinator_id} to {new_coordinator_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate coordinator: {e}")
            return False
    
    async def get_cluster_status(self) -> Dict:
        """Get current cluster status."""
        healthy_devices = sum(1 for d in self.devices.values() if d.is_healthy())
        total_devices = len(self.devices)
        
        return {
            "coordinator_id": self.coordinator_id,
            "total_devices": total_devices,
            "healthy_devices": healthy_devices,
            "devices": {device_id: device.to_dict() for device_id, device in self.devices.items()},
            "model_info": {
                "name": self.config.model.name if self.config else "unknown",
                "total_layers": self.config.model.total_layers if self.config else 0,
                "layer_distribution": self.config.model.layer_distribution if self.config else {}
            }
        }
    
    async def save_config(self):
        """Save current configuration to file."""
        if not self.config:
            return
            
        try:
            config_dict = asdict(self.config)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            logger.debug(f"✅ Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
    
    async def _connect_to_worker(self, device: DeviceConfig) -> bool:
        """Establish gRPC connection to a worker device."""
        try:
            target = resolve_grpc_target(device.hostname, device.grpc_port)
            channel = grpc.aio.insecure_channel(
                target,
                options=[
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 500MB
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024),
                ]
            )
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Test connection
            health_request = inference_pb2.Empty()
            health_response = await stub.HealthCheck(health_request, timeout=5.0)
            
            if health_response.healthy:
                self.connections[device.device_id] = channel
                return True
            else:
                await channel.close()
                return False
                
        except Exception as e:
            logger.debug(f"Connection failed for {device.device_id}: {e}")
            return False
    
    async def _health_check_loop(self):
        """Continuous health monitoring of all devices."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(10)  # Health check every 10 seconds
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(5)  # Shorter retry on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all devices."""
        tasks = []
        for device_id, device in self.devices.items():
            if device.role == DeviceRole.WORKER:
                tasks.append(self._check_worker_health(device))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_worker_health(self, device: DeviceStatus):
        """Check health of a specific worker device."""
        try:
            if device.device_id not in self.connections:
                return
            
            channel = self.connections[device.device_id]
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            health_request = inference_pb2.Empty()
            health_response = await stub.HealthCheck(health_request, timeout=3.0)
            
            if health_response.healthy:
                device.status = "online"
                device.last_heartbeat = time.time()
            else:
                device.status = "unhealthy"
                
        except Exception as e:
            logger.debug(f"Health check failed for {device.device_id}: {e}")
            device.status = "offline"
            
            # Remove failed connection
            if device.device_id in self.connections:
                try:
                    await self.connections[device.device_id].close()
                except:
                    pass
                del self.connections[device.device_id]
    
    async def shutdown(self):
        """Shutdown the device manager."""
        logger.info("🛑 Shutting down Device Manager...")
        
        self.is_running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for channel in self.connections.values():
            try:
                await channel.close()
            except:
                pass
        
        self.connections.clear()
        logger.info("✅ Device Manager shutdown complete")
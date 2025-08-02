#!/usr/bin/env python3
"""
Dynamic cluster manager for MLX distributed inference.
Handles automatic device discovery and shard rebalancing.
"""

import logging
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from mlx_discovery import MLXServiceDiscovery, MLXWorkerInfo
from distributed_config import DistributedConfig, DeviceConfig, DeviceRole
from sharding_strategy import ShardingStrategy, ShardingPlan, ShardInfo
from distributed_comm import create_communicator, CommunicationBackend
import grpc

logger = logging.getLogger(__name__)

@dataclass 
class ClusterState:
    """Current state of the distributed cluster."""
    active_workers: Dict[str, MLXWorkerInfo]
    current_sharding: Optional[ShardingPlan]
    last_update: float
    total_memory_gb: float
    total_gpu_cores: int

class DynamicClusterManager:
    """
    Manages dynamic device discovery and automatic shard rebalancing.
    """
    
    def __init__(self, model_id: str, master_port: int = 50100):
        """
        Initialize cluster manager.
        
        Args:
            model_id: Model to distribute
            master_port: Port for master gRPC server
        """
        self.model_id = model_id
        self.master_port = master_port
        self.discovery = None
        self.cluster_state = ClusterState(
            active_workers={},
            current_sharding=None,
            last_update=0,
            total_memory_gb=0,
            total_gpu_cores=0
        )
        self._lock = threading.Lock()
        self._rebalance_thread = None
        self._running = False
        
    def start(self):
        """Start the cluster manager with device discovery."""
        logger.info("🚀 Starting dynamic cluster manager")
        
        # Initialize discovery with callbacks
        self.discovery = MLXServiceDiscovery(
            on_device_added=self._on_device_added,
            on_device_removed=self._on_device_removed
        )
        
        # Register master node
        master_info = self.discovery.register_worker(
            port=self.master_port,
            rank=0  # Master is always rank 0
        )
        
        # Start discovery
        self.discovery.start_discovery()
        
        # Start rebalancing thread
        self._running = True
        self._rebalance_thread = threading.Thread(target=self._rebalance_loop)
        self._rebalance_thread.start()
        
        logger.info(f"✅ Cluster manager started on {master_info.hostname}:{master_info.port}")
        
    def _on_device_added(self, worker: MLXWorkerInfo):
        """Handle new device discovery."""
        with self._lock:
            logger.info(f"➕ New device discovered: {worker.hostname} "
                       f"({worker.available_memory_gb:.1f}GB available, {worker.gpu_cores} GPU cores)")
            
            self.cluster_state.active_workers[worker.hostname] = worker
            self._update_cluster_stats()
            
            # Trigger rebalancing
            self._schedule_rebalance()
            
    def _on_device_removed(self, hostname: str):
        """Handle device removal."""
        with self._lock:
            if hostname in self.cluster_state.active_workers:
                worker = self.cluster_state.active_workers[hostname]
                logger.info(f"➖ Device went offline: {hostname}")
                
                del self.cluster_state.active_workers[hostname]
                self._update_cluster_stats()
                
                # Trigger rebalancing
                self._schedule_rebalance()
                
    def _update_cluster_stats(self):
        """Update cluster statistics."""
        total_mem = 0
        total_gpu = 0
        
        for worker in self.cluster_state.active_workers.values():
            total_mem += worker.available_memory_gb
            total_gpu += worker.gpu_cores
            
        self.cluster_state.total_memory_gb = total_mem
        self.cluster_state.total_gpu_cores = total_gpu
        
        logger.info(f"📊 Cluster stats: {len(self.cluster_state.active_workers)} devices, "
                   f"{total_mem:.1f}GB total RAM, {total_gpu} total GPU cores")
                   
    def _schedule_rebalance(self):
        """Schedule a rebalancing operation."""
        # Set flag for rebalance thread
        self.cluster_state.last_update = time.time()
        
    def _rebalance_loop(self):
        """Background thread for shard rebalancing."""
        last_rebalance = 0
        
        while self._running:
            time.sleep(1)
            
            # Check if rebalancing needed (wait 5 seconds after last change)
            if (self.cluster_state.last_update > last_rebalance and 
                time.time() - self.cluster_state.last_update > 5):
                
                self._perform_rebalance()
                last_rebalance = time.time()
                
    def _perform_rebalance(self):
        """Perform shard rebalancing across active devices."""
        with self._lock:
            workers = list(self.cluster_state.active_workers.values())
            
        if not workers:
            logger.warning("No workers available for rebalancing")
            return
            
        logger.info(f"🔄 Rebalancing shards across {len(workers)} devices")
        
        # Create device configs from workers
        device_configs = []
        for i, worker in enumerate(workers):
            config = DeviceConfig(
                device_id=worker.device_id,
                hostname=worker.hostname,
                port=worker.port,
                role=DeviceRole.MASTER if i == 0 else DeviceRole.WORKER,
                device_index=i,
                capabilities={
                    'memory_gb': worker.memory_gb,
                    'available_memory_gb': worker.available_memory_gb,
                    'gpu_cores': worker.gpu_cores,
                    'cpu_cores': worker.cpu_cores,
                    'model': worker.model,
                }
            )
            device_configs.append(config)
            
        # Create sharding plan
        plan = self._create_simple_sharding_plan(device_configs, total_layers=28)
        
        # Validate plan
        if plan:
            logger.info(f"✅ New sharding plan created:")
            for device_id, shard_info in plan.device_assignments.items():
                logger.info(f"  - {device_id}: layers {shard_info['start_layer']}-{shard_info['end_layer']} "
                           f"({shard_info['memory_required_gb']:.1f}GB)")
                           
            # Apply the new plan
            self._apply_sharding_plan(plan, device_configs)
            
            with self._lock:
                self.cluster_state.current_sharding = plan
        else:
            logger.error("Failed to create valid sharding plan")
            
    def _apply_sharding_plan(self, plan: ShardingPlan, device_configs: List[DeviceConfig]):
        """Apply new sharding plan to devices."""
        logger.info("📋 Applying new sharding plan...")
        
        # Create shard update messages for each device
        shard_updates = {}
        for device_id, assignment in plan.device_assignments.items():
            shard_info = ShardInfo(
                start_layer=assignment['start_layer'],
                end_layer=assignment['end_layer'],
                total_layers=assignment['total_layers'],
                has_embeddings=(assignment['start_layer'] == 0),
                has_head=(assignment['end_layer'] == assignment['total_layers'] - 1)
            )
            shard_updates[device_id] = shard_info
            
        # Send shard updates to all devices via gRPC
        # In a full implementation, this would use a proper RPC mechanism
        # For now, store it for the inference engine to pick up
        self.cluster_state.shard_updates = shard_updates
        
        logger.info("✅ Sharding plan applied - devices will update on next inference")
        
    def _create_simple_sharding_plan(self, device_configs: List[DeviceConfig], total_layers: int) -> Optional[ShardingPlan]:
        """Create a simple memory-proportional sharding plan."""
        if not device_configs:
            return None
            
        # Calculate total available memory
        total_memory = sum(d.capabilities.get('available_memory_gb', 8) for d in device_configs)
        
        # Simple sharding plan structure
        device_assignments = {}
        layer_idx = 0
        
        for i, device in enumerate(device_configs):
            device_memory = device.capabilities.get('available_memory_gb', 8)
            memory_fraction = device_memory / total_memory
            
            # Calculate layers for this device
            device_layers = int(total_layers * memory_fraction)
            if i == len(device_configs) - 1:  # Last device gets remaining layers
                device_layers = total_layers - layer_idx
                
            start_layer = layer_idx
            end_layer = layer_idx + device_layers
            
            device_assignments[device.device_id] = {
                'start_layer': start_layer,
                'end_layer': end_layer - 1,  # Inclusive
                'total_layers': total_layers,
                'memory_required_gb': device_layers * 0.15,  # Rough estimate
                'has_embeddings': (i == 0),
                'has_head': (i == len(device_configs) - 1)
            }
            
            layer_idx = end_layer
            
        # Create simple plan object
        plan = type('ShardingPlan', (), {
            'device_assignments': device_assignments,
            'dict': lambda: {'device_assignments': device_assignments}
        })()
        
        return plan
        
    def get_cluster_config(self) -> DistributedConfig:
        """Get current cluster configuration."""
        with self._lock:
            workers = list(self.cluster_state.active_workers.values())
            
        # Create device configs
        device_configs = []
        for i, worker in enumerate(workers):
            config = DeviceConfig(
                device_id=worker.device_id,
                hostname=worker.hostname,
                port=worker.port,
                role=DeviceRole.MASTER if i == 0 else DeviceRole.WORKER,
                device_index=i,
                capabilities={
                    'memory_gb': worker.memory_gb,
                    'gpu_cores': worker.gpu_cores,
                    'cpu_cores': worker.cpu_cores,
                    'model': worker.model,
                }
            )
            device_configs.append(config)
            
        # Create distributed config
        config = DistributedConfig(
            model_name=self.model_id,
            device_list=device_configs,
            model_parallel_size=len(device_configs),
            pipeline_parallel_size=1
        )
        
        # Add extra attributes needed by the inference engine
        config.world_size = len(device_configs)
        config.devices = device_configs
        config.tensor_parallel_size = 1
        
        return config
        
    def check_thunderbolt_connectivity(self) -> Dict[str, List[str]]:
        """Check which devices have Thunderbolt connectivity."""
        thunderbolt_pairs = {}
        
        with self._lock:
            for worker in self.cluster_state.active_workers.values():
                if worker.thunderbolt_available:
                    # In a real implementation, we'd test actual connectivity
                    # For now, just note which devices have TB capability
                    thunderbolt_pairs[worker.hostname] = []
                    
        return thunderbolt_pairs
        
    def stop(self):
        """Stop the cluster manager."""
        logger.info("Stopping cluster manager...")
        self._running = False
        if self._rebalance_thread:
            self._rebalance_thread.join()
        if self.discovery:
            self.discovery.cleanup()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    # Create cluster manager
    manager = DynamicClusterManager(
        model_id="mlx-community/Qwen3-1.7B-8bit",
        master_port=50100
    )
    
    # Start discovery and management
    manager.start()
    
    try:
        # Run forever
        while True:
            time.sleep(5)
            
            # Show current cluster state
            config = manager.get_cluster_config()
            print(f"\n📊 Current cluster: {config.world_size} devices")
            
            # Check Thunderbolt
            tb_pairs = manager.check_thunderbolt_connectivity()
            if tb_pairs:
                print(f"⚡ Thunderbolt available on: {list(tb_pairs.keys())}")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
        manager.stop()
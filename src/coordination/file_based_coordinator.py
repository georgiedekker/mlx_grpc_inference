#!/usr/bin/env python3
"""
File-Based Distributed Coordination System
Alternative to PyTorch Distributed for Thunderbolt Bridge networks

This system bypasses Gloo's transport layer issues by using:
1. Shared file system for coordination
2. Direct TCP connections for tensor transfer
3. gRPC for control plane communication
"""
import os
import json
import time
import socket
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import fcntl
import tempfile

# Configure logger without rank in format since this module doesn't have rank info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a distributed node"""
    rank: int
    hostname: str
    ip_address: str
    grpc_port: int
    api_port: int
    status: str = "initializing"  # initializing, ready, working, error
    last_heartbeat: float = 0.0
    device_capabilities: Dict[str, Any] = None
    assigned_layers: List[int] = None

class FileBasedCoordinator:
    """
    File-based coordination system for distributed inference
    
    Uses a shared coordination directory (e.g., NFS mount or local filesystem)
    to coordinate between nodes without requiring PyTorch distributed
    """
    
    def __init__(self, 
                 rank: int, 
                 world_size: int,
                 coord_dir: str = "/tmp/mlx_coordination",
                 timeout: float = 30.0):
        self.rank = rank
        self.world_size = world_size
        self.coord_dir = Path(coord_dir)
        self.timeout = timeout
        self.node_info = None
        self.is_coordinator = (rank == 0)
        
        # Create coordination directory
        self.coord_dir.mkdir(exist_ok=True, parents=True)
        
        # File paths for coordination
        self.nodes_file = self.coord_dir / "nodes.json"
        self.barrier_file = self.coord_dir / "barrier.json"
        self.config_file = self.coord_dir / "config.json"
        self.lock_file = self.coord_dir / "coordination.lock"
        
        # Initialize node info
        self._init_node_info()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        logger.info(f"Initialized file-based coordinator: rank={rank}, world_size={world_size}")
    
    def _init_node_info(self):
        """Initialize this node's information"""
        hostname = socket.gethostname().split('.')[0]
        
        # Try to get Thunderbolt IP first, fallback to hostname resolution
        ip_address = self._get_thunderbolt_ip() or socket.gethostbyname(hostname)
        
        self.node_info = NodeInfo(
            rank=self.rank,
            hostname=hostname,
            ip_address=ip_address,
            grpc_port=50051 + self.rank,
            api_port=8100 + self.rank,
            status="initializing",
            last_heartbeat=time.time()
        )
    
    def _get_thunderbolt_ip(self) -> Optional[str]:
        """Try to detect Thunderbolt Bridge IP address"""
        try:
            import subprocess
            result = subprocess.run(['ifconfig', 'bridge0'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'inet ' in line and '192.168.5.' in line:
                        ip = line.split()[1]
                        logger.info(f"Detected Thunderbolt IP: {ip}")
                        return ip
        except Exception as e:
            logger.debug(f"Could not detect Thunderbolt IP: {e}")
        return None
    
    def _with_file_lock(self, func, *args, **kwargs):
        """Execute function with file locking"""
        with open(self.lock_file, 'w') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                return func(*args, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    def _read_nodes(self) -> Dict[int, NodeInfo]:
        """Read all nodes from coordination file"""
        if not self.nodes_file.exists():
            return {}
        
        try:
            with open(self.nodes_file, 'r') as f:
                data = json.load(f)
                return {int(rank): NodeInfo(**info) for rank, info in data.items()}
        except Exception as e:
            logger.warning(f"Error reading nodes file: {e}")
            return {}
    
    def _write_nodes(self, nodes: Dict[int, NodeInfo]):
        """Write all nodes to coordination file"""
        data = {str(rank): asdict(info) for rank, info in nodes.items()}
        with open(self.nodes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_node(self) -> bool:
        """Register this node in the coordination system"""
        def _register():
            nodes = self._read_nodes()
            nodes[self.rank] = self.node_info
            self._write_nodes(nodes)
            return True
        
        try:
            return self._with_file_lock(_register)
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    def wait_for_all_nodes(self, timeout: Optional[float] = None) -> bool:
        """Wait for all nodes to register"""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        logger.info(f"Waiting for {self.world_size} nodes to register...")
        
        while time.time() - start_time < timeout:
            nodes = self._with_file_lock(self._read_nodes)
            
            if len(nodes) == self.world_size:
                # Check that all nodes are alive (recent heartbeat)
                current_time = time.time()
                all_alive = all(
                    current_time - node.last_heartbeat < 60.0  # 60 second heartbeat timeout
                    for node in nodes.values()
                )
                
                if all_alive:
                    logger.info("All nodes registered and alive")
                    return True
                else:
                    dead_nodes = [
                        rank for rank, node in nodes.items()
                        if current_time - node.last_heartbeat >= 60.0
                    ]
                    logger.warning(f"Dead nodes detected: {dead_nodes}")
            
            time.sleep(1.0)
        
        logger.error(f"Timeout waiting for all nodes. Got {len(nodes)}/{self.world_size}")
        return False
    
    def get_node_info(self, rank: int) -> Optional[NodeInfo]:
        """Get information about a specific node"""
        nodes = self._with_file_lock(self._read_nodes)
        return nodes.get(rank)
    
    def get_all_nodes(self) -> Dict[int, NodeInfo]:
        """Get information about all nodes"""
        return self._with_file_lock(self._read_nodes)
    
    def barrier(self, barrier_name: str = "default") -> bool:
        """Synchronization barrier using file-based coordination"""
        barrier_id = f"{barrier_name}_{int(time.time())}"
        barrier_path = self.coord_dir / f"barrier_{barrier_id}.json"
        
        def _join_barrier():
            if barrier_path.exists():
                with open(barrier_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {"participants": [], "complete": False}
            
            if self.rank not in data["participants"]:
                data["participants"].append(self.rank)
            
            if len(data["participants"]) == self.world_size:
                data["complete"] = True
            
            with open(barrier_path, 'w') as f:
                json.dump(data, f)
            
            return data["complete"]
        
        # Join the barrier
        complete = self._with_file_lock(_join_barrier)
        
        if complete:
            return True
        
        # Wait for barrier completion
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if barrier_path.exists():
                with open(barrier_path, 'r') as f:
                    data = json.load(f)
                    if data.get("complete", False):
                        return True
            time.sleep(0.1)
        
        logger.error(f"Barrier timeout: {barrier_name}")
        return False
    
    def update_status(self, status: str, extra_data: Dict[str, Any] = None):
        """Update this node's status"""
        def _update():
            nodes = self._read_nodes()
            if self.rank in nodes:
                nodes[self.rank].status = status
                nodes[self.rank].last_heartbeat = time.time()
                if extra_data:
                    for key, value in extra_data.items():
                        setattr(nodes[self.rank], key, value)
                self._write_nodes(nodes)
        
        try:
            self._with_file_lock(_update)
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def _heartbeat_loop(self):
        """Background heartbeat to keep node alive"""
        while True:
            try:
                self.update_status(self.node_info.status)
                time.sleep(10.0)  # Heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)
    
    def assign_layers(self, layer_assignments: Dict[int, List[int]]):
        """Assign layers to each rank (coordinator only)"""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can assign layers")
        
        def _assign():
            nodes = self._read_nodes()
            for rank, layers in layer_assignments.items():
                if rank in nodes:
                    nodes[rank].assigned_layers = layers
            self._write_nodes(nodes)
        
        self._with_file_lock(_assign)
        logger.info(f"Assigned layers: {layer_assignments}")
    
    def get_assigned_layers(self) -> List[int]:
        """Get layers assigned to this rank"""
        node = self.get_node_info(self.rank)
        return node.assigned_layers if node else []
    
    def cleanup(self):
        """Clean up coordination files"""
        if self.is_coordinator:
            try:
                for file_path in self.coord_dir.glob("*"):
                    file_path.unlink()
                logger.info("Cleaned up coordination files")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

class DistributedInferenceManager:
    """
    High-level manager for distributed inference using file-based coordination
    """
    
    def __init__(self, rank: int, world_size: int, config_path: str):
        self.coordinator = FileBasedCoordinator(rank, world_size)
        self.config_path = config_path
        self.model = None
        self.layer_assignments = {}
        
    def initialize(self) -> bool:
        """Initialize the distributed system"""
        try:
            # Register this node
            if not self.coordinator.register_node():
                return False
            
            # Wait for all nodes
            if not self.coordinator.wait_for_all_nodes():
                return False
            
            # Load configuration and assign layers (coordinator only)
            if self.coordinator.is_coordinator:
                self._setup_layer_assignments()
            
            # Barrier to ensure all nodes have assignments
            if not self.coordinator.barrier("layer_assignment"):
                return False
            
            # Load model with assigned layers
            self._load_model_layers()
            
            # Final barrier to ensure all nodes are ready
            if not self.coordinator.barrier("model_loaded"):
                return False
            
            self.coordinator.update_status("ready")
            logger.info("Distributed inference system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.coordinator.update_status("error")
            return False
    
    def _setup_layer_assignments(self):
        """Setup layer assignments based on device capabilities"""
        # This would integrate with your existing capability-based sharding logic
        # For now, simple even split
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        total_layers = config['model']['total_layers']
        layers_per_node = total_layers // self.coordinator.world_size
        
        assignments = {}
        for rank in range(self.coordinator.world_size):
            start_layer = rank * layers_per_node
            end_layer = min((rank + 1) * layers_per_node, total_layers)
            assignments[rank] = list(range(start_layer, end_layer))
        
        self.coordinator.assign_layers(assignments)
        self.layer_assignments = assignments
    
    def _load_model_layers(self):
        """Load only the layers assigned to this rank"""
        assigned_layers = self.coordinator.get_assigned_layers()
        logger.info(f"Loading layers: {assigned_layers}")
        
        # This would integrate with your MLX-PyTorch adapter
        # to load only the required layers
        self.coordinator.update_status("loading_model")
        
        # Placeholder for actual model loading
        time.sleep(2)  # Simulate loading time
        
        logger.info(f"Loaded {len(assigned_layers)} layers")
    
    def shutdown(self):
        """Shutdown the distributed system"""
        self.coordinator.update_status("shutdown")
        self.coordinator.cleanup()
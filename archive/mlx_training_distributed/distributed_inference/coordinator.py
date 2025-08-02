#!/usr/bin/env python3
"""
Distributed Inference Coordinator System

Manages node discovery, leader election, load balancing, and request routing
for the MLX distributed inference system.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from datetime import datetime, timedelta
import hashlib
import random

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CoordinatorStrategy(Enum):
    """Coordinator selection strategies"""
    HEALTH_BASED = "health_based"
    LOAD_BASED = "load_based"
    ROUND_ROBIN = "round_robin"
    MANUAL = "manual"


@dataclass
class NodeInfo:
    """Node information structure"""
    node_id: int
    host: str
    inference_port: int
    communication_port: int
    health_port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    last_seen: float = 0.0
    health_score: float = 0.0
    load_score: float = 0.0
    model_shards: List[Tuple[int, int]] = None  # (start_layer, end_layer)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_requests: int = 0
    
    def __post_init__(self):
        if self.model_shards is None:
            self.model_shards = []


@dataclass
class ClusterConfig:
    """Cluster configuration"""
    name: str
    coordinator_port: int = 8400
    api_gateway_port: int = 8000
    election_strategy: CoordinatorStrategy = CoordinatorStrategy.HEALTH_BASED
    health_check_interval: int = 5
    failure_threshold: int = 3
    coordinator_timeout: int = 30
    max_concurrent_requests: int = 100
    enable_auto_failover: bool = True


class NodeRegistry:
    """Registry for managing cluster nodes"""
    
    def __init__(self):
        self.nodes: Dict[int, NodeInfo] = {}
        self.failure_counts: Dict[int, int] = {}
        self._lock = asyncio.Lock()
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node in the cluster"""
        async with self._lock:
            node_info.last_seen = time.time()
            node_info.status = NodeStatus.HEALTHY
            self.nodes[node_info.node_id] = node_info
            self.failure_counts[node_info.node_id] = 0
            
            logger.info(f"Registered node {node_info.node_id} at {node_info.host}:{node_info.inference_port}")
            return True
    
    async def update_node_status(self, node_id: int, status: NodeStatus, 
                               health_score: float = 0.0, load_score: float = 0.0,
                               memory_usage: float = 0.0, cpu_usage: float = 0.0,
                               active_requests: int = 0) -> bool:
        """Update node status and metrics"""
        async with self._lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.status = status
            node.health_score = health_score
            node.load_score = load_score
            node.memory_usage = memory_usage
            node.cpu_usage = cpu_usage
            node.active_requests = active_requests
            node.last_seen = time.time()
            
            # Reset failure count if node is healthy
            if status == NodeStatus.HEALTHY:
                self.failure_counts[node_id] = 0
            
            return True
    
    async def mark_node_failed(self, node_id: int) -> bool:
        """Mark a node as failed and increment failure count"""
        async with self._lock:
            if node_id not in self.nodes:
                return False
            
            self.failure_counts[node_id] += 1
            
            # Update status based on failure count
            if self.failure_counts[node_id] >= 3:
                self.nodes[node_id].status = NodeStatus.UNAVAILABLE
            else:
                self.nodes[node_id].status = NodeStatus.DEGRADED
                
            logger.warning(f"Node {node_id} marked as failed (count: {self.failure_counts[node_id]})")
            return True
    
    async def get_active_nodes(self) -> List[NodeInfo]:
        """Get all healthy and degraded nodes"""
        async with self._lock:
            return [
                node for node in self.nodes.values()
                if node.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]
            ]
    
    async def get_healthy_nodes(self) -> List[NodeInfo]:
        """Get only healthy nodes"""
        async with self._lock:
            return [
                node for node in self.nodes.values()
                if node.status == NodeStatus.HEALTHY
            ]
    
    async def get_node(self, node_id: int) -> Optional[NodeInfo]:
        """Get specific node information"""
        async with self._lock:
            return self.nodes.get(node_id)
    
    async def remove_node(self, node_id: int) -> bool:
        """Remove a node from the registry"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                del self.failure_counts[node_id]
                logger.info(f"Removed node {node_id} from registry")
                return True
            return False


class LeaderElection:
    """Leader election implementation"""
    
    def __init__(self, config: ClusterConfig, node_registry: NodeRegistry):
        self.config = config
        self.node_registry = node_registry
        self.current_leader: Optional[int] = None
        self.election_round = 0
        self.last_election_time = 0.0
        self._lock = asyncio.Lock()
    
    async def elect_leader(self) -> Optional[int]:
        """Elect a new leader based on configured strategy"""
        async with self._lock:
            active_nodes = await self.node_registry.get_active_nodes()
            
            if not active_nodes:
                logger.warning("No active nodes available for leader election")
                self.current_leader = None
                return None
            
            # Choose leader based on strategy
            if self.config.election_strategy == CoordinatorStrategy.HEALTH_BASED:
                leader = self._elect_by_health(active_nodes)
            elif self.config.election_strategy == CoordinatorStrategy.LOAD_BASED:
                leader = self._elect_by_load(active_nodes)
            elif self.config.election_strategy == CoordinatorStrategy.ROUND_ROBIN:
                leader = self._elect_round_robin(active_nodes)
            else:  # MANUAL
                leader = self.current_leader if self.current_leader in [n.node_id for n in active_nodes] else None
            
            if leader != self.current_leader:
                old_leader = self.current_leader
                self.current_leader = leader
                self.election_round += 1
                self.last_election_time = time.time()
                
                logger.info(f"Leader election: {old_leader} -> {leader} (round {self.election_round})")
            
            return self.current_leader
    
    def _elect_by_health(self, nodes: List[NodeInfo]) -> Optional[int]:
        """Elect leader based on health scores"""
        if not nodes:
            return None
        
        # Sort by health score (descending) and then by load (ascending)
        best_node = max(nodes, key=lambda n: (n.health_score, -n.load_score))
        return best_node.node_id
    
    def _elect_by_load(self, nodes: List[NodeInfo]) -> Optional[int]:
        """Elect leader based on load scores"""
        if not nodes:
            return None
        
        # Sort by load score (ascending) and then by health (descending)
        best_node = min(nodes, key=lambda n: (n.load_score, -n.health_score))
        return best_node.node_id
    
    def _elect_round_robin(self, nodes: List[NodeInfo]) -> Optional[int]:
        """Elect leader using round-robin"""
        if not nodes:
            return None
        
        # Sort nodes by ID for consistent ordering
        sorted_nodes = sorted(nodes, key=lambda n: n.node_id)
        
        if self.current_leader is None:
            return sorted_nodes[0].node_id
        
        # Find current leader index and select next
        current_index = 0
        for i, node in enumerate(sorted_nodes):
            if node.node_id == self.current_leader:
                current_index = i
                break
        
        next_index = (current_index + 1) % len(sorted_nodes)
        return sorted_nodes[next_index].node_id
    
    async def manual_set_leader(self, node_id: int) -> bool:
        """Manually set the leader"""
        async with self._lock:
            node = await self.node_registry.get_node(node_id)
            if node and node.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]:
                self.current_leader = node_id
                self.election_round += 1
                self.last_election_time = time.time()
                logger.info(f"Manual leader election: set to node {node_id}")
                return True
            return False
    
    async def get_current_leader(self) -> Optional[int]:
        """Get the current leader node ID"""
        return self.current_leader
    
    async def is_leader_healthy(self) -> bool:
        """Check if current leader is still healthy"""
        if self.current_leader is None:
            return False
        
        leader_node = await self.node_registry.get_node(self.current_leader)
        return leader_node is not None and leader_node.status == NodeStatus.HEALTHY


class HealthMonitor:
    """Health monitoring for cluster nodes"""
    
    def __init__(self, config: ClusterConfig, node_registry: NodeRegistry):
        self.config = config
        self.node_registry = node_registry
        self.monitoring_active = False
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5.0)
        )
        
        # Start monitoring task
        asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_all_nodes()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _check_all_nodes(self):
        """Check health of all registered nodes"""
        async with asyncio.TaskGroup() as tg:
            for node_id in list(self.node_registry.nodes.keys()):
                tg.create_task(self._check_node_health(node_id))
    
    async def _check_node_health(self, node_id: int):
        """Check health of a specific node"""
        node = await self.node_registry.get_node(node_id)
        if not node:
            return
        
        try:
            health_url = f"http://{node.host}:{node.health_port}/health"
            
            async with self._session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    # Update node status with health data
                    await self.node_registry.update_node_status(
                        node_id=node_id,
                        status=NodeStatus.HEALTHY,
                        health_score=health_data.get('health_score', 1.0),
                        load_score=health_data.get('load_score', 0.0),
                        memory_usage=health_data.get('memory_usage', 0.0),
                        cpu_usage=health_data.get('cpu_usage', 0.0),
                        active_requests=health_data.get('active_requests', 0)
                    )
                else:
                    await self.node_registry.mark_node_failed(node_id)
                    
        except Exception as e:
            logger.warning(f"Health check failed for node {node_id}: {e}")
            await self.node_registry.mark_node_failed(node_id)


class LoadBalancer:
    """Load balancer for request distribution"""
    
    def __init__(self, node_registry: NodeRegistry):
        self.node_registry = node_registry
        self.request_counts: Dict[int, int] = {}
        self._lock = asyncio.Lock()
    
    async def select_node_for_request(self, strategy: str = "least_loaded") -> Optional[NodeInfo]:
        """Select the best node for handling a request"""
        healthy_nodes = await self.node_registry.get_healthy_nodes()
        
        if not healthy_nodes:
            logger.warning("No healthy nodes available for request routing")
            return None
        
        if strategy == "least_loaded":
            return min(healthy_nodes, key=lambda n: (n.load_score, n.active_requests))
        elif strategy == "round_robin":
            return self._round_robin_select(healthy_nodes)
        elif strategy == "random":
            return random.choice(healthy_nodes)
        else:
            return healthy_nodes[0]  # Default to first healthy node
    
    def _round_robin_select(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection"""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        
        sorted_nodes = sorted(nodes, key=lambda n: n.node_id)
        selected = sorted_nodes[self._rr_index % len(sorted_nodes)]
        self._rr_index += 1
        
        return selected
    
    async def increment_request_count(self, node_id: int):
        """Increment request count for a node"""
        async with self._lock:
            self.request_counts[node_id] = self.request_counts.get(node_id, 0) + 1
    
    async def decrement_request_count(self, node_id: int):
        """Decrement request count for a node"""
        async with self._lock:
            if node_id in self.request_counts:
                self.request_counts[node_id] = max(0, self.request_counts[node_id] - 1)


class CoordinatorService:
    """Main coordinator service"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.node_registry = NodeRegistry()
        self.leader_election = LeaderElection(config, self.node_registry)
        self.health_monitor = HealthMonitor(config, self.node_registry)
        self.load_balancer = LoadBalancer(self.node_registry)
        self.running = False
        self._election_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the coordinator service"""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting coordinator service on port {self.config.coordinator_port}")
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        # Start periodic leader election
        self._election_task = asyncio.create_task(self._election_loop())
        
        logger.info("Coordinator service started successfully")
    
    async def stop(self):
        """Stop the coordinator service"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping coordinator service")
        
        # Stop health monitor
        await self.health_monitor.stop_monitoring()
        
        # Cancel election task
        if self._election_task:
            self._election_task.cancel()
            try:
                await self._election_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Coordinator service stopped")
    
    async def _election_loop(self):
        """Periodic leader election loop"""
        while self.running:
            try:
                await self.leader_election.elect_leader()
                await asyncio.sleep(self.config.health_check_interval * 2)
            except Exception as e:
                logger.error(f"Error in election loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node"""
        return await self.node_registry.register_node(node_info)
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        active_nodes = await self.node_registry.get_active_nodes()
        healthy_nodes = await self.node_registry.get_healthy_nodes()
        leader = await self.leader_election.get_current_leader()
        
        return {
            "cluster_name": self.config.name,
            "total_nodes": len(self.node_registry.nodes),
            "active_nodes": len(active_nodes),
            "healthy_nodes": len(healthy_nodes),
            "current_leader": leader,
            "election_round": self.leader_election.election_round,
            "election_strategy": self.config.election_strategy.value,
            "nodes": [asdict(node) for node in active_nodes],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def route_request(self, request_data: Dict[str, Any]) -> Optional[NodeInfo]:
        """Route a request to the best available node"""
        return await self.load_balancer.select_node_for_request()
    
    async def manual_failover(self, new_leader_id: int) -> bool:
        """Manually trigger failover to a new leader"""
        return await self.leader_election.manual_set_leader(new_leader_id)


# Factory function for creating coordinator instances
def create_coordinator(cluster_config: Dict[str, Any]) -> CoordinatorService:
    """Factory function to create coordinator service"""
    config = ClusterConfig(
        name=cluster_config.get("name", "mlx-cluster"),
        coordinator_port=cluster_config.get("coordinator_port", 8400),
        api_gateway_port=cluster_config.get("api_gateway_port", 8000),
        election_strategy=CoordinatorStrategy(cluster_config.get("election_strategy", "health_based")),
        health_check_interval=cluster_config.get("health_check_interval", 5),
        failure_threshold=cluster_config.get("failure_threshold", 3),
        coordinator_timeout=cluster_config.get("coordinator_timeout", 30),
        max_concurrent_requests=cluster_config.get("max_concurrent_requests", 100),
        enable_auto_failover=cluster_config.get("enable_auto_failover", True)
    )
    
    return CoordinatorService(config)
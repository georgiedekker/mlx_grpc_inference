"""
Enhanced connection pool for managing gRPC connections to remote devices.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque

from ..core.config import ClusterConfig
from .grpc_client import GRPCInferenceClient

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_used_time: float = 0.0
    creation_time: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successful_requests if self.successful_requests > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0


@dataclass
class PooledConnection:
    """A pooled connection with metadata."""
    client: GRPCInferenceClient
    stats: ConnectionStats
    in_use: bool = False
    created_at: float = 0.0
    last_health_check: float = 0.0


class ConnectionPool:
    """Enhanced pool of gRPC connections to remote devices."""
    
    def __init__(self, config: ClusterConfig, local_device_id: str, 
                 max_connections_per_device: int = 5,
                 connection_timeout_s: float = 30.0,
                 health_check_interval_s: float = 60.0):
        """
        Initialize enhanced connection pool.
        
        Args:
            config: Cluster configuration
            local_device_id: ID of the local device
            max_connections_per_device: Maximum connections per device
            connection_timeout_s: Connection timeout in seconds
            health_check_interval_s: Health check interval in seconds
        """
        self.config = config
        self.local_device_id = local_device_id
        self.max_connections_per_device = max_connections_per_device
        self.connection_timeout_s = connection_timeout_s
        self.health_check_interval_s = health_check_interval_s
        
        # Connection pools per device
        self.device_pools: Dict[str, List[PooledConnection]] = {}
        
        # Round-robin counters for load balancing
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Request queues for each device when pool is full
        self.request_queues: Dict[str, deque] = defaultdict(deque)
        
        # Connection creation locks to prevent race conditions
        self.creation_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance metrics
        self.pool_stats = {
            'total_connections_created': 0,
            'total_connections_destroyed': 0,
            'total_requests_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'queue_wait_times': []
        }
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
        
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize connection pools for all remote devices."""
        for device in self.config.devices:
            if device.device_id != self.local_device_id:
                self.device_pools[device.device_id] = []
                self.creation_locks[device.device_id] = asyncio.Lock()
                logger.info(f"Initialized connection pool for {device.device_id}")
        
        # Start health check task
        if self.health_check_interval_s > 0:
            try:
                self.health_check_task = asyncio.create_task(self._periodic_health_check())
            except RuntimeError:
                # Event loop not running yet, will start later
                logger.debug("Event loop not ready, health check task will be started later")
    
    async def start_health_checks(self):
        """Start health check task if not already running."""
        if self.health_check_interval_s > 0 and not self.health_check_task:
            self.health_check_task = asyncio.create_task(self._periodic_health_check())
            logger.info("Started health check task")
    
    async def get_connection(self, device_id: str, 
                           priority: str = "normal") -> Optional[PooledConnection]:
        """
        Get an available connection from the pool.
        
        Args:
            device_id: Target device ID
            priority: Request priority ("low", "normal", "high")
            
        Returns:
            Available pooled connection or None
        """
        if device_id not in self.device_pools:
            logger.error(f"Device {device_id} not found in pool configuration")
            return None
        
        start_time = time.time()
        
        # Try to get an available connection
        connection = await self._get_available_connection(device_id)
        
        if connection:
            self.pool_stats['cache_hits'] += 1
            connection.in_use = True
            connection.stats.last_used_time = time.time()
            return connection
        
        # No available connection, try to create a new one
        async with self.creation_locks[device_id]:
            # Check again after acquiring lock
            connection = await self._get_available_connection(device_id)
            if connection:
                self.pool_stats['cache_hits'] += 1
                connection.in_use = True
                connection.stats.last_used_time = time.time()
                return connection
            
            # Create new connection if under limit
            pool = self.device_pools[device_id]
            if len(pool) < self.max_connections_per_device:
                connection = await self._create_connection(device_id)
                if connection:
                    pool.append(connection)
                    connection.in_use = True
                    self.pool_stats['cache_misses'] += 1
                    self.pool_stats['total_connections_created'] += 1
                    return connection
        
        # Pool is full, wait for a connection to become available
        wait_time = time.time() - start_time
        self.pool_stats['queue_wait_times'].append(wait_time * 1000)  # Convert to ms
        
        # For high priority requests, try to preempt lower priority connections
        if priority == "high":
            connection = await self._preempt_connection(device_id)
            if connection:
                return connection
        
        # Wait for available connection with timeout
        try:
            connection = await asyncio.wait_for(
                self._wait_for_available_connection(device_id),
                timeout=self.connection_timeout_s
            )
            if connection:
                connection.in_use = True
                return connection
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for connection to {device_id}")
            return None
        
        return None
    
    async def _get_available_connection(self, device_id: str) -> Optional[PooledConnection]:
        """Get an available connection from the pool."""
        pool = self.device_pools[device_id]
        
        # Use round-robin with health scoring
        best_connection = None
        best_score = -1
        
        for connection in pool:
            if not connection.in_use:
                # Calculate connection health score
                score = self._calculate_connection_score(connection)
                if score > best_score:
                    best_score = score
                    best_connection = connection
        
        return best_connection
    
    def _calculate_connection_score(self, connection: PooledConnection) -> float:
        """Calculate health score for a connection (higher is better)."""
        stats = connection.stats
        
        # Base score
        score = 100.0
        
        # Penalize high latency
        if stats.avg_latency_ms > 0:
            score -= min(stats.avg_latency_ms / 10, 50)  # Max penalty of 50
        
        # Penalize low success rate
        if stats.total_requests > 5:  # Only consider after some requests
            score *= stats.success_rate
        
        # Prefer recently used connections (warm connections)
        time_since_use = time.time() - stats.last_used_time
        if time_since_use > 300:  # 5 minutes
            score *= 0.8  # 20% penalty for cold connections
        
        # Prefer newer connections if they haven't been tested much
        if stats.total_requests < 5:
            connection_age = time.time() - stats.creation_time
            if connection_age < 60:  # Less than 1 minute old
                score += 10  # Slight bonus for new connections
        
        return score
    
    async def _create_connection(self, device_id: str) -> Optional[PooledConnection]:
        """Create a new connection to a device."""
        device = self.config.get_device(device_id)
        if not device:
            return None
        
        try:
            client = GRPCInferenceClient(
                device,
                timeout=self.config.performance.request_timeout_seconds
            )
            
            stats = ConnectionStats()
            stats.creation_time = time.time()
            
            connection = PooledConnection(
                client=client,
                stats=stats,
                created_at=time.time()
            )
            
            logger.debug(f"Created new connection to {device_id}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection to {device_id}: {e}")
            return None
    
    async def _wait_for_available_connection(self, device_id: str) -> Optional[PooledConnection]:
        """Wait for a connection to become available."""
        while True:
            connection = await self._get_available_connection(device_id)
            if connection:
                return connection
            await asyncio.sleep(0.1)  # Check every 100ms
    
    async def _preempt_connection(self, device_id: str) -> Optional[PooledConnection]:
        """Preempt a connection for high priority request."""
        pool = self.device_pools[device_id]
        
        # Find the connection with lowest priority/score that's in use
        worst_connection = None
        worst_score = float('inf')
        
        for connection in pool:
            if connection.in_use:
                score = self._calculate_connection_score(connection)
                if score < worst_score:
                    worst_score = score
                    worst_connection = connection
        
        if worst_connection:
            # Mark as available (preempted)
            worst_connection.in_use = False
            logger.debug(f"Preempted connection to {device_id} for high priority request")
            return worst_connection
        
        return None
    
    def release_connection(self, device_id: str, connection: PooledConnection):
        """Release a connection back to the pool."""
        connection.in_use = False
        logger.debug(f"Released connection to {device_id}")
    
    def record_request_result(self, device_id: str, connection: PooledConnection, 
                            success: bool, latency_ms: float):
        """Record the result of a request for statistics."""
        stats = connection.stats
        stats.total_requests += 1
        
        if success:
            stats.successful_requests += 1
            stats.total_latency_ms += latency_ms
        else:
            stats.failed_requests += 1
        
        self.pool_stats['total_requests_served'] += 1
    
    async def _periodic_health_check(self):
        """Periodically health check all connections."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval_s)
                await self._health_check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _health_check_all_connections(self):
        """Health check all connections and remove unhealthy ones."""
        current_time = time.time()
        
        for device_id, pool in self.device_pools.items():
            connections_to_remove = []
            
            for connection in pool:
                # Skip connections that are in use
                if connection.in_use:
                    continue
                
                # Check if connection needs health check
                time_since_check = current_time - connection.last_health_check
                if time_since_check < self.health_check_interval_s:
                    continue
                
                try:
                    # Perform health check
                    health_result = await self._health_check_connection(connection)
                    connection.last_health_check = current_time
                    
                    if not health_result:
                        connections_to_remove.append(connection)
                        logger.info(f"Removing unhealthy connection to {device_id}")
                
                except Exception as e:
                    logger.warning(f"Health check failed for {device_id}: {e}")
                    connections_to_remove.append(connection)
            
            # Remove unhealthy connections
            for connection in connections_to_remove:
                if connection in pool:
                    pool.remove(connection)
                    try:
                        connection.client.close()
                    except Exception:
                        pass
                    self.pool_stats['total_connections_destroyed'] += 1
    
    async def _health_check_connection(self, connection: PooledConnection) -> bool:
        """Health check a single connection."""
        try:
            # Simple health check - could be enhanced with actual gRPC health check
            # For now, just check if the connection is responsive
            return True  # Placeholder - implement actual health check
        except Exception:
            return False
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        stats = self.pool_stats.copy()
        
        # Add per-device statistics
        device_stats = {}
        for device_id, pool in self.device_pools.items():
            total_connections = len(pool)
            active_connections = sum(1 for conn in pool if conn.in_use)
            
            if pool:
                avg_requests = sum(conn.stats.total_requests for conn in pool) / len(pool)
                avg_success_rate = sum(conn.stats.success_rate for conn in pool) / len(pool)
                avg_latency = sum(conn.stats.avg_latency_ms for conn in pool if conn.stats.avg_latency_ms > 0)
                avg_latency = avg_latency / len([c for c in pool if c.stats.avg_latency_ms > 0]) if avg_latency > 0 else 0
            else:
                avg_requests = avg_success_rate = avg_latency = 0
            
            device_stats[device_id] = {
                'total_connections': total_connections,
                'active_connections': active_connections,
                'available_connections': total_connections - active_connections,
                'avg_requests_per_connection': avg_requests,
                'avg_success_rate': avg_success_rate,
                'avg_latency_ms': avg_latency
            }
        
        stats['device_stats'] = device_stats
        stats['total_active_connections'] = sum(len(pool) for pool in self.device_pools.values())
        
        if self.pool_stats['queue_wait_times']:
            stats['avg_queue_wait_time_ms'] = sum(self.pool_stats['queue_wait_times']) / len(self.pool_stats['queue_wait_times'])
        else:
            stats['avg_queue_wait_time_ms'] = 0
        
        return stats
    
    def close_all(self):
        """Close all connections and cleanup."""
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Close all connections
        for device_id, pool in self.device_pools.items():
            for connection in pool:
                try:
                    connection.client.close()
                except Exception:
                    pass
            pool.clear()
        
        self.device_pools.clear()
        logger.info("All connections closed and pool cleaned up")
    
    # Legacy methods for backward compatibility
    def get_client(self, device_id: str) -> Optional[GRPCInferenceClient]:
        """Get client for a specific device (legacy method)."""
        # This is a simplified version for backward compatibility
        pool = self.device_pools.get(device_id, [])
        for connection in pool:
            if not connection.in_use:
                return connection.client
        return None
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return sum(len(pool) for pool in self.device_pools.values())
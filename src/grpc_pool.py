#!/usr/bin/env python3
"""
gRPC connection pooling for distributed MLX inference.
Manages a pool of connections for better resource utilization.
"""
import grpc
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import random

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Represents a pooled gRPC connection."""
    channel: grpc.Channel
    stub: Any
    address: str
    created_at: float
    last_used: float
    use_count: int
    is_healthy: bool
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.use_count += 1


class GRPCConnectionPool:
    """
    Manages a pool of gRPC connections to worker nodes.
    Provides connection reuse, health checking, and load balancing.
    """
    
    def __init__(
        self,
        worker_addresses: List[str],
        stub_class: Any,
        max_connections_per_worker: int = 3,
        connection_timeout: float = 5.0,
        health_check_interval: float = 30.0,
        max_connection_age: float = 300.0
    ):
        """
        Initialize gRPC connection pool.
        
        Args:
            worker_addresses: List of worker addresses (host:port)
            stub_class: gRPC stub class to use
            max_connections_per_worker: Max connections per worker
            connection_timeout: Timeout for establishing connections
            health_check_interval: Interval for health checks
            max_connection_age: Maximum age of a connection before refresh
        """
        self.worker_addresses = worker_addresses
        self.stub_class = stub_class
        self.max_connections_per_worker = max_connections_per_worker
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        self.max_connection_age = max_connection_age
        
        # Connection pools per worker
        self.connection_pools: Dict[str, deque[PooledConnection]] = {
            addr: deque() for addr in worker_addresses
        }
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_reused = 0
        self.total_requests = 0
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the connection pool and health checking."""
        self._running = True
        
        # Pre-create minimum connections
        for address in self.worker_addresses:
            await self._ensure_minimum_connections(address)
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"gRPC connection pool started with {len(self.worker_addresses)} workers")
    
    async def stop(self):
        """Stop the connection pool and close all connections."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for address, pool in self.connection_pools.items():
            while pool:
                conn = pool.popleft()
                await self._close_connection(conn)
        
        logger.info("gRPC connection pool stopped")
    
    async def get_connection(self, address: Optional[str] = None) -> PooledConnection:
        """
        Get a connection from the pool.
        
        Args:
            address: Specific worker address, or None for load balancing
            
        Returns:
            PooledConnection ready for use
        """
        self.total_requests += 1
        
        # Select address if not specified (load balancing)
        if address is None:
            address = self._select_worker()
        
        pool = self.connection_pools[address]
        
        # Try to reuse existing connection
        while pool:
            conn = pool.popleft()
            
            # Check if connection is still valid
            if self._is_connection_valid(conn):
                conn.update_usage()
                self.total_connections_reused += 1
                logger.debug(f"Reusing connection to {address} (use_count={conn.use_count})")
                return conn
            else:
                # Connection expired or unhealthy
                await self._close_connection(conn)
        
        # Create new connection
        conn = await self._create_connection(address)
        conn.update_usage()
        return conn
    
    def return_connection(self, conn: PooledConnection):
        """
        Return a connection to the pool for reuse.
        
        Args:
            conn: Connection to return
        """
        if not self._is_connection_valid(conn):
            # Don't return invalid connections
            asyncio.create_task(self._close_connection(conn))
            return
        
        pool = self.connection_pools[conn.address]
        
        # Check pool size limit
        if len(pool) < self.max_connections_per_worker:
            pool.append(conn)
            logger.debug(f"Returned connection to pool for {conn.address} (pool_size={len(pool)})")
        else:
            # Pool is full, close the connection
            asyncio.create_task(self._close_connection(conn))
    
    async def execute_with_retry(
        self,
        func,
        *args,
        address: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Execute a function with connection retry logic.
        
        Args:
            func: Function to execute with the stub
            address: Specific worker address
            max_retries: Maximum number of retries
            *args, **kwargs: Arguments for the function
        """
        last_error = None
        
        for attempt in range(max_retries):
            conn = await self.get_connection(address)
            
            try:
                # Execute the function with the stub
                result = await func(conn.stub, *args, **kwargs)
                self.return_connection(conn)
                return result
                
            except grpc.RpcError as e:
                last_error = e
                logger.warning(f"RPC error on attempt {attempt + 1}: {e}")
                
                # Mark connection as unhealthy
                conn.is_healthy = False
                await self._close_connection(conn)
                
                # Try different worker on retry
                if address is None and attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                    
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                self.return_connection(conn)
                raise
        
        raise last_error
    
    def _select_worker(self) -> str:
        """
        Select a worker using load balancing strategy.
        Currently uses round-robin with health awareness.
        """
        # Filter healthy workers
        healthy_workers = []
        for address in self.worker_addresses:
            pool = self.connection_pools[address]
            # Check if we have any healthy connections or can create new ones
            if any(conn.is_healthy for conn in pool) or len(pool) < self.max_connections_per_worker:
                healthy_workers.append(address)
        
        if not healthy_workers:
            # No healthy workers, try any
            return random.choice(self.worker_addresses)
        
        # Select worker with least connections
        return min(healthy_workers, key=lambda addr: len(self.connection_pools[addr]))
    
    def _is_connection_valid(self, conn: PooledConnection) -> bool:
        """Check if a connection is still valid for use."""
        if not conn.is_healthy:
            return False
        
        age = time.time() - conn.created_at
        if age > self.max_connection_age:
            logger.debug(f"Connection to {conn.address} expired (age={age:.1f}s)")
            return False
        
        return True
    
    async def _create_connection(self, address: str) -> PooledConnection:
        """Create a new gRPC connection."""
        logger.info(f"Creating new connection to {address}")
        
        # Create channel with options - reduced keepalive to avoid too_many_pings
        options = [
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 60000),  # 60 seconds instead of 10
            ('grpc.keepalive_timeout_ms', 20000),
            ('grpc.keepalive_permit_without_calls', False),  # Don't ping without active calls
            ('grpc.http2.max_pings_without_data', 2),  # Limit pings
        ]
        
        channel = grpc.aio.insecure_channel(address, options=options)
        
        # Wait for connection to be ready
        try:
            await asyncio.wait_for(
                channel.channel_ready(),
                timeout=self.connection_timeout
            )
        except asyncio.TimeoutError:
            await channel.close()
            raise ConnectionError(f"Failed to connect to {address}")
        
        stub = self.stub_class(channel)
        
        conn = PooledConnection(
            channel=channel,
            stub=stub,
            address=address,
            created_at=time.time(),
            last_used=time.time(),
            use_count=0,
            is_healthy=True
        )
        
        self.total_connections_created += 1
        return conn
    
    async def _close_connection(self, conn: PooledConnection):
        """Close a gRPC connection."""
        try:
            await conn.channel.close()
            logger.debug(f"Closed connection to {conn.address} (use_count={conn.use_count})")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _ensure_minimum_connections(self, address: str, min_connections: int = 1):
        """Ensure minimum number of connections in pool."""
        pool = self.connection_pools[address]
        
        while len(pool) < min_connections:
            try:
                conn = await self._create_connection(address)
                pool.append(conn)
            except Exception as e:
                logger.error(f"Failed to create connection to {address}: {e}")
                break
    
    async def _health_check_loop(self):
        """Background task for health checking connections."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_connections(self):
        """Check health of all connections."""
        for address, pool in self.connection_pools.items():
            # Check each connection in pool
            healthy_connections = deque()
            
            while pool:
                conn = pool.popleft()
                
                if self._is_connection_valid(conn):
                    # Perform health check
                    try:
                        # Assuming the stub has a HealthCheck method
                        if hasattr(conn.stub, 'HealthCheck'):
                            from src.communication import inference_pb2
                            await asyncio.wait_for(
                                conn.stub.HealthCheck(inference_pb2.HealthRequest()),
                                timeout=2.0
                            )
                            conn.is_healthy = True
                        healthy_connections.append(conn)
                    except Exception as e:
                        logger.warning(f"Health check failed for {address}: {e}")
                        conn.is_healthy = False
                        await self._close_connection(conn)
                else:
                    await self._close_connection(conn)
            
            # Put healthy connections back
            self.connection_pools[address] = healthy_connections
            
            # Ensure minimum connections
            await self._ensure_minimum_connections(address)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        total_connections = sum(len(pool) for pool in self.connection_pools.values())
        
        return {
            "total_connections": total_connections,
            "connections_created": self.total_connections_created,
            "connections_reused": self.total_connections_reused,
            "total_requests": self.total_requests,
            "reuse_rate": self.total_connections_reused / max(1, self.total_requests),
            "pools": {
                addr: {
                    "size": len(pool),
                    "healthy": sum(1 for conn in pool if conn.is_healthy)
                }
                for addr, pool in self.connection_pools.items()
            }
        }
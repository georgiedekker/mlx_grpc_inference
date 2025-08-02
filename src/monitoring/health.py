"""
Health monitoring system for distributed MLX inference components.

This module provides comprehensive health checks for:
- Device connectivity and responsiveness
- Model loading status
- Memory usage thresholds
- gRPC communication health
- Distributed pipeline integrity
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    description: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    enabled: bool = True
    critical: bool = False  # If True, failure makes entire system unhealthy


@dataclass
class HealthResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    timestamp: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class DeviceHealth:
    """Overall health status for a device."""
    device_id: str
    status: HealthStatus
    timestamp: float
    checks: Dict[str, HealthResult] = field(default_factory=dict)
    last_seen: float = 0.0
    
    @property
    def is_responsive(self) -> bool:
        """Check if device is responsive (seen recently)."""
        return (time.time() - self.last_seen) < 60.0  # 1 minute threshold


class HealthMonitor:
    """Health monitoring system for distributed components."""
    
    def __init__(self, device_id: str, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            device_id: ID of the device this monitor runs on
            check_interval: Default interval between health checks
        """
        self.device_id = device_id
        self.check_interval = check_interval
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Health checks registry
        self._health_checks: Dict[str, HealthCheck] = {}
        
        # Health results storage
        self._device_health: Dict[str, DeviceHealth] = {}
        self._check_results: Dict[str, List[HealthResult]] = {}
        
        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Thresholds
        self.memory_warning_threshold = 80.0  # %
        self.memory_critical_threshold = 95.0  # %
        self.cpu_warning_threshold = 80.0  # %
        self.cpu_critical_threshold = 95.0  # %
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info(f"Initialized health monitor for {device_id}")
    
    def _register_default_checks(self):
        """Register default health checks."""
        # System resource checks
        self.register_check(HealthCheck(
            name="memory_usage",
            description="Monitor system memory usage",
            check_function=self._check_memory_usage,
            interval_seconds=10.0,
            critical=True
        ))
        
        self.register_check(HealthCheck(
            name="cpu_usage",
            description="Monitor CPU usage",
            check_function=self._check_cpu_usage,
            interval_seconds=10.0,
            critical=False
        ))
        
        self.register_check(HealthCheck(
            name="disk_space",
            description="Monitor disk space",
            check_function=self._check_disk_space,
            interval_seconds=60.0,
            critical=True
        ))
        
        # Process checks
        self.register_check(HealthCheck(
            name="process_health",
            description="Monitor process health",
            check_function=self._check_process_health,
            interval_seconds=30.0,
            critical=True
        ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a new health check."""
        with self._lock:
            self._health_checks[health_check.name] = health_check
            self._check_results[health_check.name] = []
        
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, check_name: str):
        """Unregister a health check."""
        with self._lock:
            if check_name in self._health_checks:
                del self._health_checks[check_name]
                del self._check_results[check_name]
                logger.info(f"Unregistered health check: {check_name}")
    
    async def start(self):
        """Start health monitoring."""
        if self._running:
            logger.warning("Health monitor already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")
    
    async def stop(self):
        """Stop health monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        next_check_times = {}
        
        while self._running:
            try:
                current_time = time.time()
                
                # Run checks that are due
                for check_name, health_check in self._health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    next_time = next_check_times.get(check_name, 0)
                    if current_time >= next_time:
                        await self._run_health_check(health_check)
                        next_check_times[check_name] = current_time + health_check.interval_seconds
                
                # Update overall device health
                self._update_device_health()
                
                # Sleep briefly before next iteration
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result_data = await asyncio.wait_for(
                asyncio.to_thread(health_check.check_function),
                timeout=health_check.timeout_seconds
            )
            
            # Create result
            result = HealthResult(
                check_name=health_check.name,
                status=result_data.get('status', HealthStatus.UNKNOWN),
                timestamp=time.time(),
                message=result_data.get('message', ''),
                details=result_data.get('details', {}),
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except asyncio.TimeoutError:
            result = HealthResult(
                check_name=health_check.name,
                status=HealthStatus.CRITICAL,
                timestamp=time.time(),
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            result = HealthResult(
                check_name=health_check.name,
                status=HealthStatus.CRITICAL,
                timestamp=time.time(),
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        # Store result
        with self._lock:
            self._check_results[health_check.name].append(result)
            # Keep only last 100 results per check
            if len(self._check_results[health_check.name]) > 100:
                self._check_results[health_check.name] = self._check_results[health_check.name][-100:]
        
        logger.debug(f"Health check {health_check.name}: {result.status.value} - {result.message}")
    
    def _update_device_health(self):
        """Update overall device health based on check results."""
        with self._lock:
            # Get latest results for each check
            latest_results = {}
            for check_name, results in self._check_results.items():
                if results:
                    latest_results[check_name] = results[-1]
            
            # Determine overall status
            overall_status = HealthStatus.HEALTHY
            has_critical_failure = False
            
            for check_name, result in latest_results.items():
                health_check = self._health_checks.get(check_name)
                
                if result.status == HealthStatus.CRITICAL:
                    if health_check and health_check.critical:
                        has_critical_failure = True
                    elif overall_status != HealthStatus.CRITICAL:
                        overall_status = HealthStatus.WARNING
                
                elif result.status == HealthStatus.WARNING:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.WARNING
            
            if has_critical_failure:
                overall_status = HealthStatus.CRITICAL
            
            # Update device health
            self._device_health[self.device_id] = DeviceHealth(
                device_id=self.device_id,
                status=overall_status,
                timestamp=time.time(),
                checks=latest_results.copy(),
                last_seen=time.time()
            )
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            if percent_used >= self.memory_critical_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {percent_used:.1f}%"
            elif percent_used >= self.memory_warning_threshold:
                status = HealthStatus.WARNING
                message = f"High memory usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {percent_used:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent_used': percent_used,
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check memory usage: {str(e)}",
                'details': {}
            }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            
            if cpu_percent >= self.cpu_critical_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent >= self.cpu_warning_threshold:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent_used': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check CPU usage: {str(e)}",
                'details': {}
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            percent_used = (disk_usage.used / disk_usage.total) * 100
            
            if percent_used >= 90.0:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {percent_used:.1f}%"
            elif percent_used >= 80.0:
                status = HealthStatus.WARNING
                message = f"High disk usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {percent_used:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent_used': percent_used,
                    'used_gb': disk_usage.used / (1024**3),
                    'total_gb': disk_usage.total / (1024**3),
                    'free_gb': disk_usage.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check disk space: {str(e)}",
                'details': {}
            }
    
    def _check_process_health(self) -> Dict[str, Any]:
        """Check process health."""
        try:
            process = psutil.Process()
            
            # Check if process is running and responsive
            status = process.status()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                return {
                    'status': HealthStatus.CRITICAL,
                    'message': f"Process is {status}",
                    'details': {'process_status': status}
                }
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': f"Process healthy - Status: {status}",
                'details': {
                    'process_status': status,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / (1024**2),
                    'pid': process.pid
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check process health: {str(e)}",
                'details': {}
            }
    
    def get_device_health(self, device_id: Optional[str] = None) -> Dict[str, DeviceHealth]:
        """Get health status for devices."""
        with self._lock:
            if device_id:
                return {device_id: self._device_health.get(device_id)} if device_id in self._device_health else {}
            return self._device_health.copy()
    
    def get_check_results(self, check_name: str, limit: int = 10) -> List[HealthResult]:
        """Get recent results for a specific health check."""
        with self._lock:
            results = self._check_results.get(check_name, [])
            return results[-limit:] if results else []
    
    def is_healthy(self, device_id: Optional[str] = None) -> bool:
        """Check if a device (or local device) is healthy."""
        target_device = device_id or self.device_id
        device_health = self.get_device_health(target_device)
        
        if target_device not in device_health:
            return False
        
        health = device_health[target_device]
        return health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING] and health.is_responsive
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of all health checks."""
        with self._lock:
            device_health = self._device_health.get(self.device_id)
            
            if not device_health:
                return {
                    'device_id': self.device_id,
                    'overall_status': HealthStatus.UNKNOWN.value,
                    'timestamp': time.time(),
                    'checks': {},
                    'summary': 'No health data available'
                }
            
            # Count checks by status
            status_counts = {}
            for result in device_health.checks.values():
                status = result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'device_id': self.device_id,
                'overall_status': device_health.status.value,
                'timestamp': device_health.timestamp,
                'last_seen': device_health.last_seen,
                'is_responsive': device_health.is_responsive,
                'checks_summary': status_counts,
                'total_checks': len(device_health.checks),
                'checks': {name: {
                    'status': result.status.value,
                    'message': result.message,
                    'timestamp': result.timestamp,
                    'duration_ms': result.duration_ms
                } for name, result in device_health.checks.items()}
            }
    
    def update_remote_device_health(self, device_id: str, health_data: Dict[str, Any]):
        """Update health information for a remote device."""
        with self._lock:
            # Convert health data to DeviceHealth object
            checks = {}
            for check_name, check_data in health_data.get('checks', {}).items():
                checks[check_name] = HealthResult(
                    check_name=check_name,
                    status=HealthStatus(check_data.get('status', 'unknown')),
                    timestamp=check_data.get('timestamp', time.time()),
                    message=check_data.get('message', ''),
                    details=check_data.get('details', {}),
                    duration_ms=check_data.get('duration_ms', 0.0)
                )
            
            self._device_health[device_id] = DeviceHealth(
                device_id=device_id,
                status=HealthStatus(health_data.get('overall_status', 'unknown')),
                timestamp=health_data.get('timestamp', time.time()),
                checks=checks,
                last_seen=time.time()
            )
        
        logger.debug(f"Updated health for remote device {device_id}")


class DistributedHealthChecker:
    """Health checker for distributed system components."""
    
    def __init__(self, health_monitor: HealthMonitor, connection_pool=None):
        """
        Initialize distributed health checker.
        
        Args:
            health_monitor: Local health monitor
            connection_pool: Connection pool for remote devices
        """
        self.health_monitor = health_monitor
        self.connection_pool = connection_pool
        
        # Register distributed health checks
        if connection_pool:
            self._register_distributed_checks()
    
    def _register_distributed_checks(self):
        """Register health checks for distributed components."""
        self.health_monitor.register_check(HealthCheck(
            name="grpc_connectivity",
            description="Check gRPC connectivity to all devices",
            check_function=self._check_grpc_connectivity,
            interval_seconds=30.0,
            critical=True
        ))
        
        self.health_monitor.register_check(HealthCheck(
            name="remote_device_health",
            description="Check health of remote devices",
            check_function=self._check_remote_device_health,
            interval_seconds=60.0,
            critical=False
        ))
    
    def _check_grpc_connectivity(self) -> Dict[str, Any]:
        """Check gRPC connectivity to all remote devices."""
        if not self.connection_pool:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'No connection pool available',
                'details': {}
            }
        
        try:
            failed_connections = []
            successful_connections = []
            
            for device_id, client in self.connection_pool.clients.items():
                try:
                    start_time = time.time()
                    health_response = client.health_check()
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if health_response.get('healthy', False):
                        successful_connections.append({
                            'device_id': device_id,
                            'latency_ms': latency_ms
                        })
                    else:
                        failed_connections.append({
                            'device_id': device_id,
                            'error': health_response.get('error', 'Health check failed')
                        })
                        
                except Exception as e:
                    failed_connections.append({
                        'device_id': device_id,
                        'error': str(e)
                    })
            
            total_devices = len(self.connection_pool.clients)
            success_rate = len(successful_connections) / total_devices if total_devices > 0 else 0
            
            if success_rate == 1.0:
                status = HealthStatus.HEALTHY
                message = f"All {total_devices} devices connected"
            elif success_rate >= 0.5:
                status = HealthStatus.WARNING
                message = f"{len(successful_connections)}/{total_devices} devices connected"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {len(successful_connections)}/{total_devices} devices connected"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_devices': total_devices,
                    'successful_connections': successful_connections,
                    'failed_connections': failed_connections,
                    'success_rate': success_rate
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check gRPC connectivity: {str(e)}",
                'details': {}
            }
    
    def _check_remote_device_health(self) -> Dict[str, Any]:
        """Check health status of remote devices."""
        if not self.connection_pool:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'No connection pool available',
                'details': {}
            }
        
        try:
            healthy_devices = []
            unhealthy_devices = []
            
            for device_id, client in self.connection_pool.clients.items():
                try:
                    device_info = client.get_device_info()
                    
                    # Update local health monitor with remote device info
                    health_data = {
                        'overall_status': 'healthy' if device_info.get('healthy', True) else 'critical',
                        'timestamp': time.time(),
                        'checks': {
                            'remote_status': {
                                'status': 'healthy' if device_info.get('healthy', True) else 'critical',
                                'message': 'Remote device status',
                                'timestamp': time.time(),
                                'details': device_info
                            }
                        }
                    }
                    
                    self.health_monitor.update_remote_device_health(device_id, health_data)
                    
                    if device_info.get('healthy', True):
                        healthy_devices.append(device_id)
                    else:
                        unhealthy_devices.append(device_id)
                        
                except Exception as e:
                    unhealthy_devices.append(device_id)
                    logger.warning(f"Failed to get health for {device_id}: {e}")
            
            total_devices = len(self.connection_pool.clients)
            health_rate = len(healthy_devices) / total_devices if total_devices > 0 else 0
            
            if health_rate == 1.0:
                status = HealthStatus.HEALTHY
                message = f"All {total_devices} devices healthy"
            elif health_rate >= 0.8:
                status = HealthStatus.WARNING
                message = f"{len(healthy_devices)}/{total_devices} devices healthy"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {len(healthy_devices)}/{total_devices} devices healthy"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_devices': total_devices,
                    'healthy_devices': healthy_devices,
                    'unhealthy_devices': unhealthy_devices,
                    'health_rate': health_rate
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f"Failed to check remote device health: {str(e)}",
                'details': {}
            }
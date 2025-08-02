#!/usr/bin/env python3
"""
mDNS/Bonjour service discovery for MLX distributed inference.
Enables automatic discovery of worker devices on the network.
"""

import json
import logging
import socket
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import threading

from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener
import psutil

from hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)

@dataclass
class MLXWorkerInfo:
    """Information about an MLX worker device."""
    hostname: str
    ip_address: str
    port: int
    device_id: str
    rank: int
    memory_gb: float
    gpu_cores: int
    cpu_cores: int
    model: str
    available_memory_gb: float
    thunderbolt_available: bool
    timestamp: float

class MLXServiceListener(ServiceListener):
    """Listener for MLX worker service events."""
    
    def __init__(self, discovery_manager):
        self.discovery = discovery_manager
        
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is discovered."""
        info = zc.get_service_info(type_, name)
        if info:
            self.discovery._handle_service_added(info)
            
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        self.discovery._handle_service_removed(name)
        
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(type_, name)
        if info:
            self.discovery._handle_service_updated(info)

class MLXServiceDiscovery:
    """
    Service discovery for MLX distributed inference using mDNS/Bonjour.
    """
    
    SERVICE_TYPE = "_mlx-inference._tcp.local."
    
    def __init__(self, 
                 on_device_added: Optional[Callable[[MLXWorkerInfo], None]] = None,
                 on_device_removed: Optional[Callable[[str], None]] = None):
        """
        Initialize service discovery.
        
        Args:
            on_device_added: Callback when a new device is discovered
            on_device_removed: Callback when a device goes offline
        """
        self.zeroconf = Zeroconf()
        self.workers: Dict[str, MLXWorkerInfo] = {}
        self.local_service: Optional[ServiceInfo] = None
        self.browser: Optional[ServiceBrowser] = None
        self.on_device_added = on_device_added
        self.on_device_removed = on_device_removed
        self._lock = threading.Lock()
        
    def register_worker(self, port: int, rank: int = -1) -> MLXWorkerInfo:
        """
        Register this device as an MLX worker.
        
        Args:
            port: gRPC server port
            rank: Worker rank (-1 for auto-assignment)
            
        Returns:
            MLXWorkerInfo for this worker
        """
        # Detect hardware
        detector = HardwareDetector()
        hw_info = detector.generate_device_config()
        
        # Get network info
        hostname = socket.gethostname()
        ip_address = self._get_ip_address()
        
        # Check for Thunderbolt interfaces
        thunderbolt_available = self._check_thunderbolt_interfaces()
        
        # Get available memory
        mem_info = psutil.virtual_memory()
        available_memory_gb = mem_info.available / (1024**3)
        
        # Create worker info
        worker_info = MLXWorkerInfo(
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            device_id=hw_info.get('device_id', hostname),
            rank=rank,
            memory_gb=hw_info['memory_gb'],
            gpu_cores=hw_info['gpu_cores'],
            cpu_cores=hw_info['cpu_cores'],
            model=hw_info['model'],
            available_memory_gb=available_memory_gb,
            thunderbolt_available=thunderbolt_available,
            timestamp=time.time()
        )
        
        # Create service info
        service_name = f"{hostname}-{port}.{self.SERVICE_TYPE}"
        
        # Convert worker info to properties (keys must be strings, values must be bytes)
        properties = {}
        for k, v in asdict(worker_info).items():
            if not isinstance(k, str):
                k = str(k)
            if not isinstance(v, bytes):
                v = str(v).encode('utf-8')
            properties[k] = v
        
        self.local_service = ServiceInfo(
            self.SERVICE_TYPE,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=port,
            properties=properties,
            server=f"{hostname}.local."
        )
        
        # Register service
        self.zeroconf.register_service(self.local_service)
        logger.info(f"Registered MLX worker: {hostname}:{port} with {hw_info['memory_gb']}GB RAM, {hw_info['gpu_cores']} GPU cores")
        
        return worker_info
        
    def start_discovery(self) -> None:
        """Start discovering MLX workers on the network."""
        listener = MLXServiceListener(self)
        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, listener)
        logger.info("Started MLX worker discovery")
        
    def get_active_workers(self) -> List[MLXWorkerInfo]:
        """Get list of currently active workers."""
        with self._lock:
            # Filter out stale workers (not seen in 30 seconds)
            current_time = time.time()
            active = []
            for worker in self.workers.values():
                if current_time - worker.timestamp < 30:
                    active.append(worker)
            return sorted(active, key=lambda w: w.rank if w.rank >= 0 else 999)
            
    def _handle_service_added(self, info: ServiceInfo) -> None:
        """Handle new service discovery."""
        try:
            # Parse worker info from service properties
            worker_data = {}
            for key, value in info.properties.items():
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                # Convert back to appropriate types
                if key in ['port', 'rank', 'gpu_cores', 'cpu_cores']:
                    value = int(value)
                elif key in ['memory_gb', 'available_memory_gb', 'timestamp']:
                    value = float(value)
                elif key == 'thunderbolt_available':
                    value = value.lower() == 'true'
                worker_data[key] = value
                
            worker_info = MLXWorkerInfo(**worker_data)
            
            with self._lock:
                self.workers[worker_info.hostname] = worker_info
                
            logger.info(f"Discovered MLX worker: {worker_info.hostname}:{worker_info.port} "
                       f"({worker_info.memory_gb}GB RAM, {worker_info.gpu_cores} GPU cores)")
            
            if self.on_device_added:
                self.on_device_added(worker_info)
                
        except Exception as e:
            logger.error(f"Error handling service discovery: {e}")
            
    def _handle_service_removed(self, name: str) -> None:
        """Handle service removal."""
        # Extract hostname from service name
        hostname = name.split('-')[0]
        
        with self._lock:
            if hostname in self.workers:
                del self.workers[hostname]
                logger.info(f"MLX worker went offline: {hostname}")
                
                if self.on_device_removed:
                    self.on_device_removed(hostname)
                    
    def _handle_service_updated(self, info: ServiceInfo) -> None:
        """Handle service update."""
        # Treat as add (will update existing entry)
        self._handle_service_added(info)
        
    def _get_ip_address(self) -> str:
        """Get the primary IP address of this device."""
        # Try to connect to a public DNS server to find our IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            # Fallback to localhost
            return "127.0.0.1"
            
    def _check_thunderbolt_interfaces(self) -> bool:
        """Check if Thunderbolt networking interfaces are available."""
        try:
            interfaces = psutil.net_if_addrs()
            for iface_name in interfaces:
                # Check for Thunderbolt bridge interfaces
                if any(keyword in iface_name.lower() for keyword in ['thunderbolt', 'bridge', 'tb']):
                    return True
            return False
        except Exception:
            return False
            
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.local_service:
            self.zeroconf.unregister_service(self.local_service)
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()

# Example usage
if __name__ == "__main__":
    import random
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create discovery manager
    def on_device_added(worker: MLXWorkerInfo):
        print(f"✅ New device: {worker.hostname} with {worker.available_memory_gb:.1f}GB available RAM")
        
    def on_device_removed(hostname: str):
        print(f"❌ Device offline: {hostname}")
        
    discovery = MLXServiceDiscovery(on_device_added, on_device_removed)
    
    # Register ourselves
    port = 50000 + random.randint(100, 999)
    worker_info = discovery.register_worker(port)
    print(f"📡 Registered as: {worker_info.hostname}:{worker_info.port}")
    
    # Start discovery
    discovery.start_discovery()
    
    # Wait and show discovered devices
    try:
        while True:
            time.sleep(5)
            workers = discovery.get_active_workers()
            print(f"\n🌐 Active workers: {len(workers)}")
            for w in workers:
                print(f"  - {w.hostname}:{w.port} ({w.memory_gb}GB RAM, {w.gpu_cores} GPU cores)")
    except KeyboardInterrupt:
        print("\nShutting down...")
        discovery.cleanup()
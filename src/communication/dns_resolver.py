"""
Enhanced DNS resolution helper for handling .local hostnames with gRPC.
Provides robust mDNS resolution, network change detection, and fallback mechanisms.
"""

import socket
import logging
import time
import threading
import subprocess
import ipaddress
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    """DNS resolution strategies in order of preference."""

    MDNS_NATIVE = "mdns_native"  # Using system mDNS resolver
    MDNS_MULTICAST = "mdns_multicast"  # Direct multicast queries
    FALLBACK_PING = "fallback_ping"  # Network ping discovery
    CACHED = "cached"  # Using cached results


@dataclass
class NetworkInterface:
    """Represents a network interface."""

    name: str
    address: str
    netmask: str
    network: str
    is_active: bool = True


@dataclass
class ResolutionResult:
    """Result from DNS resolution attempt."""

    hostname: str
    ip_address: Optional[str] = None
    strategy: Optional[ResolutionStrategy] = None
    resolution_time_ms: float = 0.0
    error: Optional[str] = None
    network_interface: Optional[str] = None
    is_from_cache: bool = False
    cache_age_seconds: float = 0.0


@dataclass
class CacheEntry:
    """DNS cache entry with enhanced metadata."""

    ip_address: str
    timestamp: float
    strategy: ResolutionStrategy
    network_interface: str
    resolution_time_ms: float
    access_count: int = 0
    last_verified: float = field(default_factory=time.time)
    failed_attempts: int = 0


class EnhancedDNSResolver:
    """
    Enhanced DNS resolver with robust .local hostname support.

    Features:
    - Multiple resolution strategies with automatic fallback
    - Network interface monitoring and change detection
    - Intelligent caching with health-based invalidation
    - mDNS multicast queries for reliable .local resolution
    - Network change detection and cache invalidation
    - Connection health monitoring and cache refresh
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        health_check_interval: int = 60,
        max_resolution_attempts: int = 3,
        resolution_timeout: float = 5.0,
    ):
        """
        Initialize enhanced DNS resolver.

        Args:
            cache_ttl: Time to live for cached DNS entries in seconds
            health_check_interval: Interval for verifying cached entries in seconds
            max_resolution_attempts: Maximum attempts per resolution strategy
            resolution_timeout: Timeout per resolution attempt in seconds
        """
        self.cache_ttl = cache_ttl
        self.health_check_interval = health_check_interval
        self.max_resolution_attempts = max_resolution_attempts
        self.resolution_timeout = resolution_timeout

        # Cache and state management
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._network_interfaces: Dict[str, NetworkInterface] = {}
        self._last_network_scan = 0.0
        self._network_change_callbacks: List[Callable] = []

        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Initialize network interfaces
        self._scan_network_interfaces()

        logger.info(
            f"Enhanced DNS resolver initialized with {len(self._network_interfaces)} interfaces"
        )

    def resolve_hostname(self, hostname: str, prefer_ipv4: bool = True) -> Optional[str]:
        """
        Resolve hostname to IP address using enhanced strategies.

        Args:
            hostname: Hostname to resolve (e.g., "mini2.local")
            prefer_ipv4: Whether to prefer IPv4 over IPv6

        Returns:
            IP address string or None if resolution failed
        """
        result = self.resolve_hostname_detailed(hostname, prefer_ipv4)
        return result.ip_address

    def resolve_hostname_detailed(
        self, hostname: str, prefer_ipv4: bool = True
    ) -> ResolutionResult:
        """
        Resolve hostname with detailed result information.

        Args:
            hostname: Hostname to resolve
            prefer_ipv4: Whether to prefer IPv4 over IPv6

        Returns:
            ResolutionResult with detailed resolution information
        """
        start_time = time.time()

        # Check if it's already an IP address
        if self._is_ip_address(hostname):
            return ResolutionResult(
                hostname=hostname, ip_address=hostname, strategy=None, resolution_time_ms=0.0
            )

        # Check cache first
        cached_result = self._check_cache(hostname)
        if cached_result:
            return cached_result

        # Try resolution strategies in order
        strategies = [
            ResolutionStrategy.MDNS_NATIVE,
            ResolutionStrategy.MDNS_MULTICAST,
            ResolutionStrategy.FALLBACK_PING,
        ]

        last_error = None
        for strategy in strategies:
            try:
                result = self._resolve_with_strategy(hostname, strategy, prefer_ipv4)
                if result.ip_address:
                    result.resolution_time_ms = (time.time() - start_time) * 1000
                    self._cache_result(hostname, result)
                    logger.info(
                        f"Resolved {hostname} -> {result.ip_address} using {strategy.value} in {result.resolution_time_ms:.1f}ms"
                    )
                    return result
                else:
                    last_error = result.error
                    logger.debug(f"Strategy {strategy.value} failed for {hostname}: {result.error}")
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Strategy {strategy.value} failed for {hostname}: {e}")

        # All strategies failed
        result = ResolutionResult(
            hostname=hostname,
            error=f"All resolution strategies failed. Last error: {last_error}",
            resolution_time_ms=(time.time() - start_time) * 1000,
        )
        logger.error(f"Failed to resolve {hostname}: {result.error}")
        return result

    def _check_cache(self, hostname: str) -> Optional[ResolutionResult]:
        """Check cache for existing resolution."""
        with self._lock:
            entry = self._cache.get(hostname)
            if not entry:
                return None

            current_time = time.time()
            cache_age = current_time - entry.timestamp

            # Check if cache is still valid
            if cache_age < self.cache_ttl and entry.failed_attempts < 3:
                entry.access_count += 1
                logger.debug(
                    f"Using cached IP for {hostname}: {entry.ip_address} (age: {cache_age:.1f}s)"
                )

                return ResolutionResult(
                    hostname=hostname,
                    ip_address=entry.ip_address,
                    strategy=ResolutionStrategy.CACHED,
                    resolution_time_ms=0.0,
                    network_interface=entry.network_interface,
                    is_from_cache=True,
                    cache_age_seconds=cache_age,
                )
            else:
                # Cache expired or too many failures, remove it
                logger.debug(f"Cache expired or invalid for {hostname}, removing entry")
                del self._cache[hostname]
                return None

    def _resolve_with_strategy(
        self, hostname: str, strategy: ResolutionStrategy, prefer_ipv4: bool
    ) -> ResolutionResult:
        """Resolve hostname using a specific strategy."""
        if strategy == ResolutionStrategy.MDNS_NATIVE:
            return self._resolve_mdns_native(hostname, prefer_ipv4)
        elif strategy == ResolutionStrategy.MDNS_MULTICAST:
            return self._resolve_mdns_multicast(hostname, prefer_ipv4)
        elif strategy == ResolutionStrategy.FALLBACK_PING:
            return self._resolve_fallback_ping(hostname, prefer_ipv4)
        else:
            return ResolutionResult(hostname=hostname, error=f"Unknown strategy: {strategy}")

    def _resolve_mdns_native(self, hostname: str, prefer_ipv4: bool) -> ResolutionResult:
        """Resolve using native system mDNS resolver."""
        try:
            if prefer_ipv4:
                # Try IPv4 first
                try:
                    result = socket.getaddrinfo(
                        hostname, None, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICSERV
                    )
                    if result:
                        ip = result[0][4][0]
                        interface = self._get_interface_for_ip(ip)
                        return ResolutionResult(
                            hostname=hostname,
                            ip_address=ip,
                            strategy=ResolutionStrategy.MDNS_NATIVE,
                            network_interface=interface,
                        )
                except socket.gaierror as e:
                    logger.debug(f"IPv4 mDNS resolution failed for {hostname}: {e}")

                # Fall back to IPv6
                try:
                    result = socket.getaddrinfo(
                        hostname,
                        None,
                        socket.AF_INET6,
                        socket.SOCK_STREAM,
                        0,
                        socket.AI_NUMERICSERV,
                    )
                    if result:
                        ip = result[0][4][0]
                        interface = self._get_interface_for_ip(ip)
                        return ResolutionResult(
                            hostname=hostname,
                            ip_address=ip,
                            strategy=ResolutionStrategy.MDNS_NATIVE,
                            network_interface=interface,
                        )
                except socket.gaierror as e:
                    logger.debug(f"IPv6 mDNS resolution failed for {hostname}: {e}")
            else:
                # Standard resolution (both IPv4 and IPv6)
                result = socket.getaddrinfo(hostname, None)
                if result:
                    ip = result[0][4][0]
                    interface = self._get_interface_for_ip(ip)
                    return ResolutionResult(
                        hostname=hostname,
                        ip_address=ip,
                        strategy=ResolutionStrategy.MDNS_NATIVE,
                        network_interface=interface,
                    )

        except Exception as e:
            return ResolutionResult(hostname=hostname, error=f"mDNS native resolution failed: {e}")

        return ResolutionResult(hostname=hostname, error="mDNS native resolution failed")

    def _resolve_mdns_multicast(self, hostname: str, prefer_ipv4: bool) -> ResolutionResult:
        """Resolve using direct mDNS multicast queries."""
        try:
            # Use DNS-SD tools if available
            cmd = ["dns-sd", "-G", "v4", hostname.rstrip(".local"), ".local"]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.resolution_timeout
                )

                if result.returncode == 0:
                    # Parse dns-sd output
                    for line in result.stdout.split("\n"):
                        if hostname in line and "." in line:
                            # Extract IP address from dns-sd output
                            ip_match = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", line)
                            if ip_match:
                                ip = ip_match.group(1)
                                interface = self._get_interface_for_ip(ip)
                                return ResolutionResult(
                                    hostname=hostname,
                                    ip_address=ip,
                                    strategy=ResolutionStrategy.MDNS_MULTICAST,
                                    network_interface=interface,
                                )
            except subprocess.TimeoutExpired:
                logger.debug(f"DNS-SD timeout for {hostname}")
            except FileNotFoundError:
                logger.debug("dns-sd command not available, trying alternative")

            # Fallback to avahi if available on Linux
            if hasattr(subprocess, "DEVNULL"):
                try:
                    cmd = ["avahi-resolve", "-n", hostname]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=self.resolution_timeout
                    )

                    if result.returncode == 0 and result.stdout:
                        # Parse avahi output: "hostname.local 192.168.1.100"
                        parts = result.stdout.strip().split()
                        if len(parts) >= 2:
                            ip = parts[1]
                            if self._is_ip_address(ip):
                                interface = self._get_interface_for_ip(ip)
                                return ResolutionResult(
                                    hostname=hostname,
                                    ip_address=ip,
                                    strategy=ResolutionStrategy.MDNS_MULTICAST,
                                    network_interface=interface,
                                )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.debug("avahi-resolve not available or timeout")

        except Exception as e:
            return ResolutionResult(
                hostname=hostname, error=f"mDNS multicast resolution failed: {e}"
            )

        return ResolutionResult(hostname=hostname, error="mDNS multicast resolution failed")

    def _resolve_fallback_ping(self, hostname: str, prefer_ipv4: bool) -> ResolutionResult:
        """Resolve using network scanning and ping as fallback."""
        try:
            # Scan local network for the hostname
            for interface in self._network_interfaces.values():
                if not interface.is_active:
                    continue

                try:
                    network = ipaddress.IPv4Network(
                        f"{interface.address}/{interface.netmask}", strict=False
                    )

                    # Try common host suffixes based on hostname pattern
                    base_name = hostname.replace(".local", "")
                    potential_ips = self._generate_potential_ips(network, base_name)

                    for ip in potential_ips:
                        if self._ping_host(str(ip)):
                            # Verify this IP responds to the hostname
                            if self._verify_hostname_ip_match(hostname, str(ip)):
                                return ResolutionResult(
                                    hostname=hostname,
                                    ip_address=str(ip),
                                    strategy=ResolutionStrategy.FALLBACK_PING,
                                    network_interface=interface.name,
                                )
                except Exception as e:
                    logger.debug(f"Error scanning network {interface.network}: {e}")

        except Exception as e:
            return ResolutionResult(
                hostname=hostname, error=f"Fallback ping resolution failed: {e}"
            )

        return ResolutionResult(hostname=hostname, error="Fallback ping resolution failed")

    def _scan_network_interfaces(self):
        """Scan and update network interfaces."""
        with self._lock:
            try:
                import psutil

                self._network_interfaces.clear()

                for interface_name, addresses in psutil.net_if_addrs().items():
                    for addr in addresses:
                        if addr.family == socket.AF_INET and addr.address != "127.0.0.1":
                            try:
                                network = ipaddress.IPv4Network(
                                    f"{addr.address}/{addr.netmask}", strict=False
                                )
                                self._network_interfaces[interface_name] = NetworkInterface(
                                    name=interface_name,
                                    address=addr.address,
                                    netmask=addr.netmask,
                                    network=str(network.network_address),
                                    is_active=True,
                                )
                            except Exception as e:
                                logger.debug(f"Error processing interface {interface_name}: {e}")

                self._last_network_scan = time.time()
                logger.debug(f"Scanned {len(self._network_interfaces)} network interfaces")

            except ImportError:
                logger.warning("psutil not available, network interface monitoring disabled")
            except Exception as e:
                logger.error(f"Error scanning network interfaces: {e}")

    def _get_interface_for_ip(self, ip: str) -> Optional[str]:
        """Get the network interface that can reach the given IP."""
        try:
            ip_addr = ipaddress.IPv4Address(ip)
            for interface in self._network_interfaces.values():
                try:
                    network = ipaddress.IPv4Network(
                        f"{interface.network}/{interface.netmask}", strict=False
                    )
                    if ip_addr in network:
                        return interface.name
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _generate_potential_ips(
        self, network: ipaddress.IPv4Network, hostname: str
    ) -> List[ipaddress.IPv4Address]:
        """Generate potential IP addresses based on hostname patterns."""
        potential_ips = []

        # Extract number patterns from hostname (e.g., "mini2" -> 2)
        numbers = re.findall(r"\d+", hostname)

        for num in numbers:
            try:
                # Try common patterns: .10x, .1x, .x
                base_num = int(num)
                candidates = [base_num, 100 + base_num, 10 + base_num, 200 + base_num]

                for candidate in candidates:
                    if candidate < 256:
                        try:
                            ip = ipaddress.IPv4Address(
                                f"{network.network_address.exploded.rsplit('.', 1)[0]}.{candidate}"
                            )
                            if (
                                ip in network
                                and ip != network.network_address
                                and ip != network.broadcast_address
                            ):
                                potential_ips.append(ip)
                        except Exception:
                            continue

            except ValueError:
                continue

        return potential_ips[:10]  # Limit to first 10 candidates

    def _ping_host(self, ip: str) -> bool:
        """Ping a host to check if it's reachable."""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1000", ip], capture_output=True, timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def _verify_hostname_ip_match(self, hostname: str, ip: str) -> bool:
        """Verify that the IP actually corresponds to the hostname."""
        try:
            # Try reverse DNS lookup
            reverse_result = socket.gethostbyaddr(ip)
            if reverse_result and hostname.lower() in reverse_result[0].lower():
                return True
        except Exception:
            pass

        # For .local domains, if we can ping and it's in the right network, accept it
        return hostname.endswith(".local")

    def _is_ip_address(self, hostname: str) -> bool:
        """Check if the given string is already an IP address."""
        try:
            ipaddress.IPv4Address(hostname)
            return True
        except ValueError:
            pass

        try:
            ipaddress.IPv6Address(hostname)
            return True
        except ValueError:
            pass

        return False

    def _cache_result(self, hostname: str, result: ResolutionResult):
        """Cache the DNS resolution result."""
        if not result.ip_address:
            return

        with self._lock:
            self._cache[hostname] = CacheEntry(
                ip_address=result.ip_address,
                timestamp=time.time(),
                strategy=result.strategy,
                network_interface=result.network_interface or "unknown",
                resolution_time_ms=result.resolution_time_ms,
            )
            logger.debug(
                f"Cached DNS result: {hostname} -> {result.ip_address} via {result.strategy.value if result.strategy else 'unknown'}"
            )

    def create_grpc_target(self, hostname: str, port: int) -> str:
        """
        Create a gRPC target string with resolved IP address.

        Args:
            hostname: Original hostname (may be .local)
            port: Port number

        Returns:
            gRPC target string (e.g., "192.168.1.100:50051")
        """
        resolved_ip = self.resolve_hostname(hostname)
        if resolved_ip:
            return f"{resolved_ip}:{port}"
        else:
            # Fall back to original hostname
            logger.warning(f"Could not resolve {hostname}, using original hostname")
            return f"{hostname}:{port}"

    def start_monitoring(self):
        """Start background monitoring for network changes and cache health."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started DNS monitoring thread")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped DNS monitoring thread")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for network changes
                if time.time() - self._last_network_scan > 30:  # Scan every 30 seconds
                    self._check_network_changes()

                # Verify cache entries health
                self._verify_cache_health()

                # Call network change callbacks
                for callback in self._network_change_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in network change callback: {e}")

            except Exception as e:
                logger.error(f"Error in DNS monitoring loop: {e}")

            time.sleep(self.health_check_interval)

    def _check_network_changes(self):
        """Check for network interface changes and invalidate cache if needed."""
        old_interfaces = dict(self._network_interfaces)
        self._scan_network_interfaces()

        # Check if any interfaces changed
        changed = False
        for name, interface in self._network_interfaces.items():
            old_interface = old_interfaces.get(name)
            if not old_interface or old_interface.address != interface.address:
                logger.info(
                    f"Network interface {name} changed: {old_interface.address if old_interface else 'new'} -> {interface.address}"
                )
                changed = True

        # Check for removed interfaces
        for name in old_interfaces:
            if name not in self._network_interfaces:
                logger.info(f"Network interface {name} removed")
                changed = True

        if changed:
            self._invalidate_cache_by_network_change()

    def _verify_cache_health(self):
        """Verify health of cached entries and remove stale ones."""
        with self._lock:
            current_time = time.time()
            to_remove = []

            for hostname, entry in self._cache.items():
                # Check if entry is too old
                if current_time - entry.timestamp > self.cache_ttl:
                    to_remove.append(hostname)
                    continue

                # Periodically verify that cached IPs are still reachable
                if current_time - entry.last_verified > self.health_check_interval:
                    if self._ping_host(entry.ip_address):
                        entry.last_verified = current_time
                        entry.failed_attempts = 0
                    else:
                        entry.failed_attempts += 1
                        if entry.failed_attempts >= 3:
                            logger.warning(
                                f"Cache entry for {hostname} failed health check, removing"
                            )
                            to_remove.append(hostname)

            for hostname in to_remove:
                del self._cache[hostname]
                logger.debug(f"Removed stale cache entry for {hostname}")

    def _invalidate_cache_by_network_change(self):
        """Invalidate cache entries that may be affected by network changes."""
        with self._lock:
            # For now, clear all cache on network changes to be safe
            # In the future, could be more intelligent about which entries to invalidate
            cache_size = len(self._cache)
            self._cache.clear()
            logger.info(f"Invalidated {cache_size} cache entries due to network changes")

    def add_network_change_callback(self, callback: Callable):
        """Add a callback to be called when network changes are detected."""
        self._network_change_callbacks.append(callback)

    def remove_network_change_callback(self, callback: Callable):
        """Remove a network change callback."""
        try:
            self._network_change_callbacks.remove(callback)
        except ValueError:
            pass

    def invalidate_hostname(self, hostname: str):
        """Manually invalidate a specific hostname from cache."""
        with self._lock:
            if hostname in self._cache:
                del self._cache[hostname]
                logger.info(f"Manually invalidated cache for {hostname}")

    def clear_cache(self):
        """Clear the DNS cache."""
        with self._lock:
            self._cache.clear()
            logger.info("DNS cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get DNS cache statistics."""
        with self._lock:
            current_time = time.time()
            active_entries = 0
            expired_entries = 0
            failed_entries = 0

            for hostname, entry in self._cache.items():
                if current_time - entry.timestamp < self.cache_ttl:
                    if entry.failed_attempts < 3:
                        active_entries += 1
                    else:
                        failed_entries += 1
                else:
                    expired_entries += 1

            return {
                "total_entries": len(self._cache),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "failed_entries": failed_entries,
                "cache_ttl": self.cache_ttl,
                "network_interfaces": len(self._network_interfaces),
                "monitoring_active": self._monitoring_active,
            }

    def get_detailed_cache_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about cached entries."""
        with self._lock:
            current_time = time.time()
            result = {}

            for hostname, entry in self._cache.items():
                result[hostname] = {
                    "ip_address": entry.ip_address,
                    "age_seconds": current_time - entry.timestamp,
                    "strategy": entry.strategy.value,
                    "network_interface": entry.network_interface,
                    "resolution_time_ms": entry.resolution_time_ms,
                    "access_count": entry.access_count,
                    "failed_attempts": entry.failed_attempts,
                    "last_verified_seconds_ago": current_time - entry.last_verified,
                    "is_expired": current_time - entry.timestamp > self.cache_ttl,
                }

            return result


# Legacy compatibility class
class LocalDNSResolver(EnhancedDNSResolver):
    """Legacy compatibility wrapper for the enhanced DNS resolver."""

    def __init__(self, cache_ttl: int = 300):
        super().__init__(cache_ttl=cache_ttl)


# Global DNS resolver instance - using enhanced resolver
_dns_resolver = EnhancedDNSResolver()


def resolve_grpc_target(hostname: str, port: int) -> str:
    """
    Convenience function to resolve gRPC target.

    Args:
        hostname: Hostname to resolve
        port: Port number

    Returns:
        Resolved gRPC target string
    """
    return _dns_resolver.create_grpc_target(hostname, port)


def get_global_resolver() -> EnhancedDNSResolver:
    """Get the global DNS resolver instance."""
    return _dns_resolver


def start_global_monitoring():
    """Start global DNS monitoring."""
    _dns_resolver.start_monitoring()


def stop_global_monitoring():
    """Stop global DNS monitoring."""
    _dns_resolver.stop_monitoring()


def test_dns_resolution():
    """Test enhanced DNS resolution for common .local hostnames."""
    resolver = EnhancedDNSResolver()
    test_hosts = ["mini2.local", "master.local", "mini1.local", "nonexistent.local"]

    print("🧪 Testing Enhanced DNS Resolution")
    print("=" * 50)

    for hostname in test_hosts:
        print(f"\nTesting {hostname}:")
        result = resolver.resolve_hostname_detailed(hostname)

        if result.ip_address:
            print(f"✅ {hostname} -> {result.ip_address}")
            print(f"   Strategy: {result.strategy.value if result.strategy else 'none'}")
            print(f"   Resolution time: {result.resolution_time_ms:.1f}ms")
            if result.network_interface:
                print(f"   Network interface: {result.network_interface}")
            if result.is_from_cache:
                print(f"   From cache (age: {result.cache_age_seconds:.1f}s)")

            target = resolver.create_grpc_target(hostname, 50051)
            print(f"   gRPC target: {target}")
        else:
            print(f"❌ {hostname} -> FAILED")
            if result.error:
                print(f"   Error: {result.error}")

    print("\n" + "=" * 50)
    print("Cache Statistics:")
    stats = resolver.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nDetailed Cache Info:")
    cache_info = resolver.get_detailed_cache_info()
    for hostname, info in cache_info.items():
        print(f"  {hostname}:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def test_network_monitoring():
    """Test network change monitoring functionality."""
    print("\n🔍 Testing Network Monitoring")
    print("=" * 40)

    resolver = EnhancedDNSResolver()

    def network_change_callback():
        print("📡 Network change detected!")

    resolver.add_network_change_callback(network_change_callback)
    resolver.start_monitoring()

    print("Network monitoring started...")
    print("Try changing network (e.g., disconnect/reconnect WiFi) and observe callbacks")
    print("Monitoring will run for 30 seconds...")

    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")

    resolver.stop_monitoring()
    print("Network monitoring stopped")


if __name__ == "__main__":
    test_dns_resolution()

    # Uncomment to test network monitoring
    # test_network_monitoring()

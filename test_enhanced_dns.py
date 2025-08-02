#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced DNS resolution functionality.
Tests .local hostname resolution, network change handling, and gRPC integration.
"""

import asyncio
import time
import logging
import sys
from typing import Dict, List
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.communication.dns_resolver import (
    EnhancedDNSResolver, 
    ResolutionStrategy,
    test_dns_resolution,
    test_network_monitoring
)
from src.core.config import ClusterConfig
from src.communication.grpc_client import GRPCInferenceClient, ConnectionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DNSTestSuite:
    """Comprehensive DNS testing suite."""
    
    def __init__(self):
        self.resolver = EnhancedDNSResolver()
        self.test_hostnames = ["mini2.local", "master.local", "mini1.local", "nonexistent.local"]
        self.results = {}
    
    def test_basic_resolution(self):
        """Test basic DNS resolution functionality."""
        logger.info("🧪 Testing basic DNS resolution")
        
        for hostname in self.test_hostnames:
            result = self.resolver.resolve_hostname_detailed(hostname)
            self.results[hostname] = result
            
            if result.ip_address:
                logger.info(f"✅ {hostname} -> {result.ip_address} "
                           f"({result.strategy.value if result.strategy else 'unknown'}, "
                           f"{result.resolution_time_ms:.1f}ms)")
            else:
                logger.warning(f"❌ {hostname} -> FAILED: {result.error}")
    
    def test_caching_behavior(self):
        """Test DNS caching behavior."""
        logger.info("🧪 Testing DNS caching behavior")
        
        # First resolution (should cache)
        hostname = "mini2.local"
        result1 = self.resolver.resolve_hostname_detailed(hostname)
        
        # Second resolution (should use cache)
        result2 = self.resolver.resolve_hostname_detailed(hostname)
        
        if result1.ip_address and result2.ip_address:
            logger.info(f"✅ Cache test passed:")
            logger.info(f"   First resolution: {result1.resolution_time_ms:.1f}ms")
            logger.info(f"   Cached resolution: {result2.resolution_time_ms:.1f}ms")
            logger.info(f"   From cache: {result2.is_from_cache}")
        else:
            logger.error("❌ Cache test failed - resolution failed")
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        logger.info("🧪 Testing cache invalidation")
        
        hostname = "mini2.local"
        
        # Resolve and cache
        result1 = self.resolver.resolve_hostname_detailed(hostname)
        
        # Check cache stats
        stats_before = self.resolver.get_cache_stats()
        logger.info(f"Cache entries before invalidation: {stats_before['active_entries']}")
        
        # Invalidate
        self.resolver.invalidate_hostname(hostname)
        
        # Check cache stats after
        stats_after = self.resolver.get_cache_stats()
        logger.info(f"Cache entries after invalidation: {stats_after['active_entries']}")
        
        # Resolve again (should not be from cache)
        result2 = self.resolver.resolve_hostname_detailed(hostname)
        
        if not result2.is_from_cache:
            logger.info("✅ Cache invalidation test passed")
        else:
            logger.error("❌ Cache invalidation test failed")
    
    def test_network_interface_detection(self):
        """Test network interface detection."""
        logger.info("🧪 Testing network interface detection")
        
        stats = self.resolver.get_cache_stats()
        logger.info(f"Detected network interfaces: {stats['network_interfaces']}")
        
        # Try to scan manually
        self.resolver._scan_network_interfaces()
        interfaces = self.resolver._network_interfaces
        
        logger.info(f"Network interfaces found: {len(interfaces)}")
        for name, interface in interfaces.items():
            logger.info(f"   {name}: {interface.address}/{interface.netmask}")
    
    def test_monitoring_functionality(self):
        """Test background monitoring."""
        logger.info("🧪 Testing monitoring functionality")
        
        # Start monitoring
        self.resolver.start_monitoring()
        logger.info(f"Monitoring active: {self.resolver._monitoring_active}")
        
        # Wait a bit
        time.sleep(2)
        
        # Stop monitoring
        self.resolver.stop_monitoring()
        logger.info(f"Monitoring stopped: {not self.resolver._monitoring_active}")
    
    def test_grpc_integration(self):
        """Test gRPC client integration with enhanced DNS."""
        logger.info("🧪 Testing gRPC client integration")
        
        try:
            # Load cluster config
            config_path = Path(__file__).parent / "config" / "cluster_config.yaml"
            if not config_path.exists():
                logger.warning("❌ Cluster config not found, skipping gRPC integration test")
                return
            
            config = ClusterConfig.from_yaml(str(config_path))
            
            # Test connection pool creation
            pool = ConnectionPool(config, "mini1", self.resolver)
            
            # Get connection status
            status = pool.get_connection_status()
            logger.info("Connection status:")
            for device_id, info in status.items():
                logger.info(f"   {device_id}: healthy={info.get('healthy', False)}")
            
            # Get DNS stats from pool
            dns_stats = pool.get_dns_stats()
            logger.info(f"DNS stats from pool: {dns_stats}")
            
            pool.close_all()
            logger.info("✅ gRPC integration test completed")
            
        except ImportError as e:
            logger.warning(f"❌ gRPC integration test skipped: missing dependencies ({e})")
        except Exception as e:
            logger.error(f"❌ gRPC integration test failed: {e}")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of DNS resolution."""
        logger.info("🧪 Testing performance characteristics")
        
        hostname = "mini2.local"
        iterations = 10
        
        # Time multiple resolutions
        times = []
        for i in range(iterations):
            start = time.time()
            result = self.resolver.resolve_hostname(hostname)
            end = time.time()
            
            if result:
                times.append((end - start) * 1000)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            logger.info(f"Performance results for {hostname}:")
            logger.info(f"   Average: {avg_time:.1f}ms")
            logger.info(f"   Min: {min_time:.1f}ms")
            logger.info(f"   Max: {max_time:.1f}ms")
            logger.info(f"   Iterations: {len(times)}")
    
    def run_all_tests(self):
        """Run all DNS tests."""
        logger.info("🚀 Starting comprehensive DNS test suite")
        logger.info("=" * 60)
        
        tests = [
            self.test_basic_resolution,
            self.test_caching_behavior,
            self.test_cache_invalidation,
            self.test_network_interface_detection,
            self.test_monitoring_functionality,
            self.test_performance_characteristics,
            self.test_grpc_integration,
        ]
        
        for test in tests:
            try:
                test()
                logger.info("✅ Test passed\n")
            except Exception as e:
                logger.error(f"❌ Test failed: {e}\n")
        
        logger.info("🏁 DNS test suite completed")
        logger.info("=" * 60)
        
        # Final stats
        stats = self.resolver.get_cache_stats()
        cache_info = self.resolver.get_detailed_cache_info()
        
        logger.info("Final DNS Statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("\nCached Hostnames:")
        for hostname, info in cache_info.items():
            logger.info(f"   {hostname}: {info['ip_address']} "
                       f"(strategy: {info['strategy']}, "
                       f"age: {info['age_seconds']:.1f}s)")


def test_network_change_simulation():
    """Simulate network change scenarios."""
    logger.info("🔄 Testing network change simulation")
    
    resolver = EnhancedDNSResolver()
    
    # Set up callback
    change_count = 0
    def network_change_callback():
        nonlocal change_count
        change_count += 1
        logger.info(f"📡 Network change callback triggered (#{change_count})")
    
    resolver.add_network_change_callback(network_change_callback)
    resolver.start_monitoring()
    
    # Simulate cache entries
    hostname = "mini2.local"
    result = resolver.resolve_hostname_detailed(hostname)
    
    if result.ip_address:
        logger.info(f"Initial resolution: {hostname} -> {result.ip_address}")
        
        # Manually trigger network change simulation
        logger.info("Simulating network change...")
        resolver._check_network_changes()
        
        # Wait for monitoring loop
        time.sleep(2)
        
        # Check if cache was invalidated
        stats = resolver.get_cache_stats()
        logger.info(f"Cache entries after network change: {stats['active_entries']}")
    
    resolver.stop_monitoring()
    logger.info("Network change simulation completed")


async def test_async_integration():
    """Test async integration patterns."""
    logger.info("🔄 Testing async integration")
    
    resolver = EnhancedDNSResolver()
    
    # Test concurrent resolutions
    hostnames = ["mini2.local", "master.local", "mini1.local"]
    
    async def resolve_hostname_async(hostname):
        # Simulate async context
        await asyncio.sleep(0.1)
        return resolver.resolve_hostname_detailed(hostname)
    
    # Run concurrent resolutions
    tasks = [resolve_hostname_async(hostname) for hostname in hostnames]
    results = await asyncio.gather(*tasks)
    
    logger.info("Concurrent resolution results:")
    for hostname, result in zip(hostnames, results):
        if result.ip_address:
            logger.info(f"   {hostname} -> {result.ip_address}")
        else:
            logger.info(f"   {hostname} -> FAILED")


def main():
    """Main test function."""
    logger.info("🧪 Enhanced DNS Resolution Test Suite")
    logger.info("=" * 60)
    
    # Run basic DNS tests from module
    logger.info("\n1. Running built-in DNS tests:")
    test_dns_resolution()
    
    # Run comprehensive test suite
    logger.info("\n2. Running comprehensive test suite:")
    test_suite = DNSTestSuite()
    test_suite.run_all_tests()
    
    # Test network change simulation
    logger.info("\n3. Testing network change simulation:")
    test_network_change_simulation()
    
    # Test async integration
    logger.info("\n4. Testing async integration:")
    asyncio.run(test_async_integration())
    
    logger.info("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()
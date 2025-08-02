#!/usr/bin/env python3
"""
Test script to validate gRPC connectivity with enhanced DNS resolution.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.communication.dns_resolver import EnhancedDNSResolver, start_global_monitoring
from src.core.config import ClusterConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_connectivity():
    """Test basic connectivity to .local hostnames."""
    logger.info("🔗 Testing basic connectivity to .local hostnames")
    
    resolver = EnhancedDNSResolver()
    hostnames = ["mini2.local", "master.local", "mini1.local"]
    
    for hostname in hostnames:
        logger.info(f"\nTesting connectivity to {hostname}:")
        
        # DNS resolution
        result = resolver.resolve_hostname_detailed(hostname)
        if result.ip_address:
            logger.info(f"✅ DNS: {hostname} -> {result.ip_address}")
            logger.info(f"   Strategy: {result.strategy.value}")
            logger.info(f"   Network interface: {result.network_interface}")
            logger.info(f"   Resolution time: {result.resolution_time_ms:.1f}ms")
            
            # Basic ping test
            import subprocess
            try:
                ping_result = subprocess.run(
                    ["ping", "-c", "1", "-W", "1000", result.ip_address],
                    capture_output=True,
                    timeout=3
                )
                if ping_result.returncode == 0:
                    logger.info("✅ PING: Host is reachable")
                else:
                    logger.warning("❌ PING: Host is not reachable")
            except Exception as e:
                logger.warning(f"❌ PING: Failed to ping - {e}")
                
        else:
            logger.error(f"❌ DNS: Failed to resolve {hostname}")
            logger.error(f"   Error: {result.error}")


def test_dns_failover():
    """Test DNS failover scenarios."""
    logger.info("\n🔄 Testing DNS failover scenarios")
    
    resolver = EnhancedDNSResolver()
    
    # Test with a non-existent hostname to trigger fallback
    hostname = "nonexistent.local"
    logger.info(f"\nTesting fallback for {hostname}:")
    
    result = resolver.resolve_hostname_detailed(hostname)
    if result.ip_address:
        logger.info(f"✅ Unexpectedly resolved: {hostname} -> {result.ip_address}")
    else:
        logger.info(f"✅ Expected failure: {result.error}")
        logger.info("   Fallback mechanisms were tested")


def test_cache_performance():
    """Test cache performance and behavior."""
    logger.info("\n⚡ Testing cache performance")
    
    resolver = EnhancedDNSResolver()
    hostname = "mini2.local"
    
    # First resolution (cold)
    start_time = time.time()
    result1 = resolver.resolve_hostname_detailed(hostname)
    cold_time = (time.time() - start_time) * 1000
    
    # Second resolution (cached)
    start_time = time.time()
    result2 = resolver.resolve_hostname_detailed(hostname)
    cached_time = (time.time() - start_time) * 1000
    
    if result1.ip_address and result2.ip_address:
        logger.info(f"✅ Cache performance test:")
        logger.info(f"   Cold resolution: {cold_time:.1f}ms")
        logger.info(f"   Cached resolution: {cached_time:.1f}ms")
        logger.info(f"   Speedup: {cold_time/cached_time:.1f}x")
        logger.info(f"   Cache hit: {result2.is_from_cache}")
    else:
        logger.error("❌ Cache performance test failed")


def test_network_monitoring():
    """Test network monitoring capabilities."""
    logger.info("\n📡 Testing network monitoring")
    
    resolver = EnhancedDNSResolver()
    
    # Check network interfaces
    resolver._scan_network_interfaces()
    interfaces = resolver._network_interfaces
    
    logger.info(f"Network interfaces detected: {len(interfaces)}")
    for name, interface in interfaces.items():
        logger.info(f"   {name}: {interface.address}/{interface.netmask} (active: {interface.is_active})")
    
    # Test monitoring start/stop
    logger.info("Starting monitoring...")
    resolver.start_monitoring()
    
    time.sleep(1)
    
    logger.info("Stopping monitoring...")
    resolver.stop_monitoring()
    
    logger.info("✅ Network monitoring test completed")


def test_config_integration():
    """Test integration with cluster configuration."""
    logger.info("\n⚙️  Testing cluster configuration integration")
    
    config_path = Path(__file__).parent / "config" / "cluster_config.yaml"
    
    if not config_path.exists():
        logger.warning("❌ Cluster config not found, skipping integration test")
        return
    
    try:
        config = ClusterConfig.from_yaml(str(config_path))
        logger.info(f"✅ Loaded cluster config with {len(config.devices)} devices")
        
        resolver = EnhancedDNSResolver()
        
        for device in config.devices:
            hostname = device.hostname
            logger.info(f"\nTesting device {device.device_id} ({hostname}):")
            
            result = resolver.resolve_hostname_detailed(hostname)
            if result.ip_address:
                logger.info(f"✅ {hostname} -> {result.ip_address}")
                
                # Test gRPC target creation
                grpc_target = resolver.create_grpc_target(hostname, device.grpc_port)
                logger.info(f"   gRPC target: {grpc_target}")
            else:
                logger.warning(f"❌ Failed to resolve {hostname}: {result.error}")
                
    except Exception as e:
        logger.error(f"❌ Config integration test failed: {e}")


def main():
    """Run all connectivity tests."""
    logger.info("🚀 gRPC Connectivity Test Suite with Enhanced DNS")
    logger.info("=" * 60)
    
    # Start global monitoring
    start_global_monitoring()
    
    try:
        test_basic_connectivity()
        test_dns_failover()
        test_cache_performance()
        test_network_monitoring()
        test_config_integration()
        
        logger.info("\n🎉 All connectivity tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
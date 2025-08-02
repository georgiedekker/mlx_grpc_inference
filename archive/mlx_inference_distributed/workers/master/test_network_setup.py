#!/usr/bin/env python3
"""
Network connectivity test for distributed MLX inference.
Tests connectivity between mini1 and mini2.
"""

import socket
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ping(hostname: str) -> bool:
    """Test if hostname is reachable via ping."""
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '3000', hostname], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"✅ {hostname} is reachable via ping")
            return True
        else:
            logger.error(f"❌ Cannot ping {hostname}: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Ping to {hostname} timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Ping test failed for {hostname}: {e}")
        return False


def test_dns_resolution(hostname: str) -> bool:
    """Test DNS resolution for hostname."""
    try:
        ip = socket.gethostbyname(hostname)
        logger.info(f"✅ {hostname} resolves to {ip}")
        return True
    except socket.gaierror as e:
        logger.error(f"❌ DNS resolution failed for {hostname}: {e}")
        return False


def test_port_connectivity(hostname: str, port: int, timeout: int = 5) -> bool:
    """Test if port is open on hostname."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ Port {port} is open on {hostname}")
            return True
        else:
            logger.error(f"❌ Port {port} is closed on {hostname}")
            return False
    except Exception as e:
        logger.error(f"❌ Port connectivity test failed for {hostname}:{port}: {e}")
        return False


def test_ssh_connectivity(hostname: str) -> bool:
    """Test SSH connectivity to hostname."""
    try:
        result = subprocess.run([
            'ssh', 
            '-o', 'ConnectTimeout=5',
            '-o', 'BatchMode=yes',  # No interactive prompts
            '-o', 'StrictHostKeyChecking=no',
            hostname, 
            'echo "SSH test successful"'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info(f"✅ SSH connectivity to {hostname} works")
            return True
        else:
            logger.warning(f"⚠️  SSH to {hostname} failed (may not be configured): {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"⚠️  SSH to {hostname} timed out")
        return False
    except Exception as e:
        logger.warning(f"⚠️  SSH test failed for {hostname}: {e}")
        return False


def test_grpc_ports(hostname: str) -> bool:
    """Test gRPC ports on hostname."""
    ports_to_test = [50100, 50101, 50051]  # Communication ports + inference port
    results = []
    
    for port in ports_to_test:
        result = test_port_connectivity(hostname, port, timeout=2)
        results.append(result)
        time.sleep(0.1)  # Small delay between tests
    
    return any(results)  # At least one port should be open


def comprehensive_network_test() -> bool:
    """Run comprehensive network connectivity tests."""
    logger.info("🔍 Starting comprehensive network connectivity tests...")
    
    hostnames = ['mini2.local', 'mini2']
    all_tests_passed = True
    
    for hostname in hostnames:
        logger.info(f"\n📡 Testing connectivity to {hostname}...")
        
        # Test 1: DNS Resolution
        dns_ok = test_dns_resolution(hostname)
        
        # Test 2: Ping connectivity
        ping_ok = test_ping(hostname)
        
        # Test 3: SSH (optional)
        ssh_ok = test_ssh_connectivity(hostname)
        
        # Test 4: gRPC ports
        grpc_ok = test_grpc_ports(hostname)
        
        # Summary for this hostname
        hostname_ok = dns_ok and ping_ok
        if hostname_ok:
            logger.info(f"✅ {hostname} is accessible")
            break  # Found working hostname
        else:
            logger.error(f"❌ {hostname} is not accessible")
            all_tests_passed = False
    
    if not all_tests_passed:
        logger.error("\n❌ Network connectivity tests failed!")
        logger.error("Troubleshooting steps:")
        logger.error("1. Check if mini2 is powered on")
        logger.error("2. Verify mini2.local is in /etc/hosts or DNS")
        logger.error("3. Test: ping mini2.local")
        logger.error("4. Check firewall settings")
        return False
    
    logger.info("\n✅ All network connectivity tests passed!")
    return True


def test_local_ports() -> bool:
    """Test that required local ports are available."""
    logger.info("🔍 Testing local port availability...")
    
    required_ports = [8100, 50100]  # API port, gRPC communication port
    
    for port in required_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            logger.info(f"✅ Port {port} is available")
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.warning(f"⚠️  Port {port} is already in use (this may be expected)")
            else:
                logger.error(f"❌ Port {port} test failed: {e}")
                return False
    
    return True


def main():
    """Main test runner."""
    logger.info("🚀 MLX Distributed Network Setup Test")
    logger.info("=" * 50)
    
    # Test local ports first
    local_ok = test_local_ports()
    
    # Test network connectivity
    network_ok = comprehensive_network_test()
    
    logger.info("\n" + "=" * 50)
    if local_ok and network_ok:
        logger.info("🎉 All tests passed! Network is ready for distributed inference.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix network issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Simple import test for monitoring system components.

This script validates that all monitoring modules can be imported correctly
without requiring external dependencies to be installed.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all monitoring modules can be imported."""
    print("Testing monitoring system imports...")
    
    try:
        # Test individual module imports
        print("Importing metrics module...")
        import monitoring.metrics
        print("✓ metrics module imported")
        
        print("Importing health module...")
        import monitoring.health
        print("✓ health module imported")
        
        print("Importing dashboard module...")
        import monitoring.dashboard
        print("✓ dashboard module imported")
        
        print("Importing monitoring package...")
        import monitoring
        print("✓ monitoring package imported")
        
        # Test that classes are available
        print("Checking class availability...")
        
        from monitoring.metrics import MetricsCollector, InferenceMetrics, DeviceMetrics
        print("✓ Metrics classes available")
        
        from monitoring.health import HealthMonitor, HealthStatus, HealthCheck
        print("✓ Health classes available")
        
        from monitoring.dashboard import MonitoringDashboard
        print("✓ Dashboard classes available")
        
        # Test __all__ exports
        print("Checking __all__ exports...")
        expected_exports = [
            'MetricsCollector', 'InferenceMetrics', 'DeviceMetrics', 'NetworkMetrics',
            'HealthMonitor', 'HealthStatus', 'HealthCheck', 'HealthResult', 
            'DeviceHealth', 'DistributedHealthChecker',
            'MonitoringDashboard', 'start_monitoring_dashboard'
        ]
        
        for export in expected_exports:
            assert hasattr(monitoring, export), f"Missing export: {export}"
        
        print("✓ All expected exports available")
        
        print("\n✅ All import tests passed!")
        print("Monitoring system structure is correct.")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_structure():
    """Check that all expected files exist."""
    print("Checking monitoring system file structure...")
    
    src_dir = Path(__file__).parent.parent / "src"
    
    expected_files = [
        "monitoring/__init__.py",
        "monitoring/metrics.py", 
        "monitoring/health.py",
        "monitoring/dashboard.py"
    ]
    
    for file_path in expected_files:
        full_path = src_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    print("✓ All expected files present")
    return True


def check_integration_points():
    """Check that integration points exist in target files."""
    print("Checking integration points...")
    
    src_dir = Path(__file__).parent.parent / "src"
    
    # Check orchestrator integration
    orchestrator_file = src_dir / "coordinator" / "orchestrator.py"
    if orchestrator_file.exists():
        content = orchestrator_file.read_text()
        if "MetricsCollector" in content and "HealthMonitor" in content:
            print("✓ Orchestrator integration present")
        else:
            print("❌ Orchestrator integration missing")
            return False
    
    # Check worker integration
    worker_file = src_dir / "worker" / "worker_server.py"
    if worker_file.exists():
        content = worker_file.read_text()
        if "MetricsCollector" in content and "HealthMonitor" in content:
            print("✓ Worker integration present")
        else:
            print("❌ Worker integration missing")
            return False
    
    # Check gRPC client integration
    grpc_file = src_dir / "communication" / "grpc_client.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        if "metrics_collector" in content:
            print("✓ gRPC client integration present")
        else:
            print("❌ gRPC client integration missing")
            return False
    
    print("✓ All integration points present")
    return True


def main():
    """Main validation entry point."""
    print("Validating monitoring system implementation...\n")
    
    success = True
    
    success &= check_file_structure()
    print()
    
    success &= test_imports()
    print()
    
    success &= check_integration_points()
    print()
    
    if success:
        print("✅ Monitoring system validation passed!")
        print("\nNext steps:")
        print("1. Install dependencies: uv add psutil fastapi uvicorn jinja2 websockets")
        print("2. Test with: python tests/test_monitoring_system.py")
        print("3. Start dashboard: python scripts/start_monitoring_dashboard.py")
        sys.exit(0)
    else:
        print("❌ Monitoring system validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
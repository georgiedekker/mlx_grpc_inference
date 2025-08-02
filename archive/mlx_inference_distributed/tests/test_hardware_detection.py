"""
Test suite for hardware detection functionality.
"""

import pytest
import json
import platform
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware_detector import HardwareDetector


class TestHardwareDetector:
    """Test hardware detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create a hardware detector instance."""
        return HardwareDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.info == {}
    
    @patch('subprocess.run')
    @patch('platform.node')
    def test_detect_mac_mini(self, mock_node, mock_run, detector):
        """Test detection of Mac mini."""
        mock_node.return_value = "mini1.local"
        
        # Mock system_profiler output for Mac mini
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                "SPHardwareDataType": [{
                    "machine_model": "Mac15,12",
                    "machine_name": "Mac mini"
                }]
            }),
            returncode=0
        )
        
        device_type = detector._detect_device_type()
        assert device_type == "Mac mini"
    
    @patch('subprocess.run')
    def test_detect_macbook_pro(self, mock_run, detector):
        """Test detection of MacBook Pro."""
        # Mock system_profiler output for MacBook Pro
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                "SPHardwareDataType": [{
                    "machine_model": "Mac15,3",
                    "machine_name": "MacBook Pro"
                }]
            }),
            returncode=0
        )
        
        device_type = detector._detect_device_type()
        assert device_type == "MacBook Pro"
    
    @patch('subprocess.run')
    def test_detect_m4_chip(self, mock_run, detector):
        """Test detection of M4 chip."""
        mock_run.return_value = MagicMock(
            stdout="Apple M4",
            returncode=0
        )
        
        chip_model = detector._detect_chip_model()
        assert chip_model == "M4"
    
    @patch('subprocess.run')
    def test_detect_m4_pro_chip(self, mock_run, detector):
        """Test detection of M4 Pro chip."""
        mock_run.return_value = MagicMock(
            stdout="Apple M4 Pro",
            returncode=0
        )
        
        chip_model = detector._detect_chip_model()
        assert chip_model == "M4 Pro"
    
    @patch('psutil.virtual_memory')
    def test_detect_memory(self, mock_memory, detector):
        """Test memory detection."""
        # Mock 16GB of RAM
        mock_memory.return_value = MagicMock(total=16 * 1024**3)
        
        memory_gb = detector._detect_memory()
        assert memory_gb == 16.0
    
    @patch('psutil.virtual_memory')
    def test_detect_memory_48gb(self, mock_memory, detector):
        """Test 48GB memory detection."""
        # Mock 48GB of RAM
        mock_memory.return_value = MagicMock(total=48 * 1024**3)
        
        memory_gb = detector._detect_memory()
        assert memory_gb == 48.0
    
    @patch('subprocess.run')
    @patch('psutil.cpu_count')
    def test_detect_cpu_cores(self, mock_cpu_count, mock_run, detector):
        """Test CPU core detection."""
        mock_cpu_count.return_value = 10
        
        # Mock performance cores
        mock_run.side_effect = [
            MagicMock(stdout="4", returncode=0),  # Performance cores
            MagicMock(stdout="6", returncode=0)   # Efficiency cores
        ]
        
        cpu_info = detector._detect_cpu_cores()
        assert cpu_info["total"] == 10
        assert cpu_info["performance_cores"] == 4
        assert cpu_info["efficiency_cores"] == 6
    
    @patch('subprocess.run')
    def test_detect_gpu_cores_m4(self, mock_run, detector):
        """Test GPU core detection for M4."""
        detector.info["chip_model"] = "M4"
        
        # Mock ioreg output
        mock_run.return_value = MagicMock(
            stdout='"gpu-core-count" = 10',
            returncode=0
        )
        
        gpu_cores = detector._detect_gpu_cores()
        assert gpu_cores == 10
    
    @patch('subprocess.run')
    def test_detect_gpu_cores_m4_pro(self, mock_run, detector):
        """Test GPU core detection for M4 Pro."""
        detector.info["chip_model"] = "M4 Pro"
        
        # Mock ioreg output
        mock_run.return_value = MagicMock(
            stdout='"gpu-core-count" = 16',
            returncode=0
        )
        
        gpu_cores = detector._detect_gpu_cores()
        assert gpu_cores == 16
    
    def test_neural_engine_cores(self, detector):
        """Test Neural Engine core detection."""
        detector.info["chip_model"] = "M4"
        cores = detector._detect_neural_engine_cores()
        assert cores == 16
        
        detector.info["chip_model"] = "M4 Ultra"
        cores = detector._detect_neural_engine_cores()
        assert cores == 32
    
    @patch('subprocess.run')
    def test_battery_detection_laptop(self, mock_run, detector):
        """Test battery detection for laptops."""
        mock_run.return_value = MagicMock(
            stdout="InternalBattery-0 100%; charged",
            returncode=0
        )
        
        assert detector._is_laptop() is True
        
        battery_info = detector._detect_battery_info()
        assert battery_info is not None
        assert battery_info["percentage"] == 100
    
    @patch('subprocess.run')
    def test_battery_detection_desktop(self, mock_run, detector):
        """Test battery detection for desktops."""
        mock_run.return_value = MagicMock(
            stdout="",
            returncode=0
        )
        
        assert detector._is_laptop() is False
        battery_info = detector._detect_battery_info()
        assert battery_info is None
    
    @patch('hardware_detector.HardwareDetector._detect_device_type')
    @patch('hardware_detector.HardwareDetector._detect_chip_model')
    @patch('hardware_detector.HardwareDetector._detect_memory')
    @patch('hardware_detector.HardwareDetector._detect_cpu_cores')
    @patch('hardware_detector.HardwareDetector._detect_gpu_cores')
    def test_generate_device_config(self, mock_gpu, mock_cpu, mock_memory, 
                                   mock_chip, mock_device, detector):
        """Test device configuration generation."""
        mock_device.return_value = "Mac mini"
        mock_chip.return_value = "M4"
        mock_memory.return_value = 16.0
        mock_cpu.return_value = {
            "total": 10,
            "performance_cores": 4,
            "efficiency_cores": 6
        }
        mock_gpu.return_value = 10
        
        # Detect all hardware
        detector.detect_all()
        config = detector.generate_device_config()
        
        assert config["device_type"] == "Mac mini"
        assert config["model"] == "M4"
        assert config["device_name"] == "Mac mini M4"
        assert config["memory_gb"] == 16.0
        assert config["gpu_cores"] == 10
        assert config["cpu_cores"] == 10
        assert config["max_recommended_model_size_gb"] == 12.8
        assert config["is_laptop"] is False
    
    def test_error_handling(self, detector):
        """Test error handling in detection methods."""
        with patch('subprocess.run', side_effect=Exception("Command failed")):
            device_type = detector._detect_device_type()
            # Should handle error gracefully
            assert device_type is not None  # Falls back to other detection methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Automatic hardware detection for Apple Silicon devices.
Detects device type, memory, CPU/GPU cores, and other capabilities.
"""

import subprocess
import json
import re
import psutil
import platform
import mlx.core as mx


class HardwareDetector:
    """Automatically detect Apple Silicon hardware specifications."""
    
    def __init__(self):
        self.info = {}
        
    def detect_all(self):
        """Run all detection methods and return complete hardware info."""
        self.info = {
            "hostname": platform.node(),
            "device_type": self._detect_device_type(),
            "chip_model": self._detect_chip_model(),
            "memory_gb": self._detect_memory(),
            "cpu_cores": self._detect_cpu_cores(),
            "gpu_cores": self._detect_gpu_cores(),
            "neural_engine_cores": self._detect_neural_engine_cores(),
            "mlx_info": self._detect_mlx_info(),
            "system_info": self._detect_system_info(),
            "thermal_state": self._detect_thermal_state(),
            "battery_info": self._detect_battery_info() if self._is_laptop() else None
        }
        
        # Calculate recommended model size (80% of RAM)
        self.info["max_recommended_model_size_gb"] = round(self.info["memory_gb"] * 0.8, 1)
        
        return self.info
    
    def _run_command(self, cmd):
        """Run a shell command and return output."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return None
    
    def _detect_device_type(self):
        """Detect if this is a Mac mini, MacBook Pro, Mac Studio, etc."""
        # Use system_profiler to get hardware info
        hw_info = self._run_command("system_profiler SPHardwareDataType -json")
        if hw_info:
            try:
                data = json.loads(hw_info)
                model = data["SPHardwareDataType"][0].get("machine_model", "")
                machine_name = data["SPHardwareDataType"][0].get("machine_name", "")
                
                # Determine device type from model identifier
                if "Mac14,12" in model or "Mac14,3" in model:
                    return "Mac mini"
                elif "MacBookPro" in machine_name:
                    return "MacBook Pro"
                elif "Mac13,1" in model or "Mac13,2" in model:
                    return "Mac Studio"
                elif "iMac" in machine_name:
                    return "iMac"
                else:
                    return machine_name or "Unknown Mac"
            except:
                pass
        
        # Fallback: check for battery to distinguish laptop vs desktop
        if self._is_laptop():
            return "MacBook"
        else:
            # Check model identifier another way
            model = self._run_command("sysctl -n hw.model")
            if model and "Mac15" in model:
                return "Mac mini"  # M4 Mac mini uses Mac15,x
            return "Mac"
    
    def _is_laptop(self):
        """Check if device has a battery (laptop)."""
        battery = self._run_command("pmset -g batt")
        return battery and "InternalBattery" in battery
    
    def _detect_chip_model(self):
        """Detect the Apple Silicon chip model (M1, M2, M3, M4, etc.)."""
        # Get chip info from sysctl
        chip_info = self._run_command("sysctl -n machdep.cpu.brand_string")
        if chip_info:
            # Extract chip model (M1, M2, M3, M4, etc.)
            if "Apple M" in chip_info:
                match = re.search(r'Apple (M\d+\s*(?:Pro|Max|Ultra)?)', chip_info)
                if match:
                    return match.group(1)
        
        # Fallback: detect from system profiler
        hw_info = self._run_command("system_profiler SPHardwareDataType")
        if hw_info:
            for line in hw_info.split('\n'):
                if "Chip:" in line:
                    chip = line.split(":")[-1].strip()
                    return chip
        
        return "Unknown Apple Silicon"
    
    def _detect_memory(self):
        """Detect total system memory in GB."""
        # Use psutil for cross-platform compatibility
        memory_bytes = psutil.virtual_memory().total
        memory_gb = round(memory_bytes / (1024**3), 1)
        return memory_gb
    
    def _detect_cpu_cores(self):
        """Detect total CPU cores and breakdown."""
        cpu_info = {
            "total": psutil.cpu_count(logical=False),
            "performance_cores": 0,
            "efficiency_cores": 0
        }
        
        # Get performance/efficiency core breakdown
        perf_cores = self._run_command("sysctl -n hw.perflevel0.physicalcpu")
        eff_cores = self._run_command("sysctl -n hw.perflevel1.physicalcpu")
        
        if perf_cores and perf_cores.isdigit():
            cpu_info["performance_cores"] = int(perf_cores)
        if eff_cores and eff_cores.isdigit():
            cpu_info["efficiency_cores"] = int(eff_cores)
            
        return cpu_info
    
    def _detect_gpu_cores(self):
        """Detect GPU core count."""
        # Try to get GPU info from ioreg
        gpu_info = self._run_command("ioreg -l | grep -i 'gpu-core-count'")
        if gpu_info:
            match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', gpu_info)
            if match:
                return int(match.group(1))
        
        # Fallback: estimate based on chip model
        chip = self.info.get("chip_model", "")
        gpu_cores_map = {
            "M4": 10,
            "M4 Pro": 16,  # Base M4 Pro
            "M4 Max": 32,  # Base M4 Max
            "M3": 10,
            "M3 Pro": 14,
            "M3 Max": 30,
            "M2": 10,
            "M2 Pro": 16,
            "M2 Max": 30,
            "M1": 8,
            "M1 Pro": 14,
            "M1 Max": 24,
            "M1 Ultra": 48
        }
        
        for model, cores in gpu_cores_map.items():
            if model in chip:
                return cores
        
        return None
    
    def _detect_neural_engine_cores(self):
        """Detect Neural Engine core count."""
        # All current Apple Silicon chips have 16 Neural Engine cores
        # except M1 which has 16, and Ultra variants which have 32
        chip = self.info.get("chip_model", "")
        if "Ultra" in chip:
            return 32
        return 16
    
    def _detect_mlx_info(self):
        """Detect MLX-specific information."""
        try:
            return {
                "version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
                "default_device": str(mx.default_device()),
                "metal_available": mx.metal.is_available() if hasattr(mx, "metal") else True
            }
        except:
            return {
                "version": "unknown",
                "default_device": "unknown",
                "metal_available": True
            }
    
    def _detect_system_info(self):
        """Detect OS and system information."""
        return {
            "os_version": platform.mac_ver()[0],
            "platform": platform.platform(),
            "architecture": platform.machine()
        }
    
    def _detect_thermal_state(self):
        """Detect current thermal state."""
        thermal = self._run_command("pmset -g therm")
        if thermal:
            for line in thermal.split('\n'):
                if "CPU_Scheduler_Limit" in line:
                    return line.strip()
        return "normal"
    
    def _detect_battery_info(self):
        """Detect battery information for laptops."""
        battery = self._run_command("pmset -g batt")
        if battery:
            lines = battery.split('\n')
            for line in lines:
                if "InternalBattery" in line:
                    # Extract percentage and state
                    match = re.search(r'(\d+)%', line)
                    if match:
                        return {
                            "percentage": int(match.group(1)),
                            "charging": "AC attached" in battery or "charging" in line.lower()
                        }
        return None
    
    def generate_device_config(self):
        """Generate device configuration for distributed_config.json."""
        info = self.detect_all()
        
        # Build device name
        device_type = info["device_type"]
        chip = info["chip_model"]
        device_name = f"{device_type} {chip}"
        
        return {
            "device_type": device_type,
            "model": chip,
            "device_name": device_name,
            "memory_gb": info["memory_gb"],
            "gpu_cores": info["gpu_cores"],
            "cpu_cores": info["cpu_cores"]["total"],
            "cpu_performance_cores": info["cpu_cores"]["performance_cores"],
            "cpu_efficiency_cores": info["cpu_cores"]["efficiency_cores"],
            "neural_engine_cores": info["neural_engine_cores"],
            "mlx_metal_available": info["mlx_info"]["metal_available"],
            "max_recommended_model_size_gb": info["max_recommended_model_size_gb"],
            "os_version": info["system_info"]["os_version"],
            "is_laptop": info["battery_info"] is not None
        }


def main():
    """Run hardware detection and display results."""
    detector = HardwareDetector()
    info = detector.detect_all()
    
    print("🔍 Apple Silicon Hardware Detection")
    print("=" * 50)
    print(f"Device Type: {info['device_type']}")
    print(f"Chip Model: {info['chip_model']}")
    print(f"Memory: {info['memory_gb']} GB")
    print(f"CPU Cores: {info['cpu_cores']['total']} total ({info['cpu_cores']['performance_cores']}P + {info['cpu_cores']['efficiency_cores']}E)")
    print(f"GPU Cores: {info['gpu_cores']}")
    print(f"Neural Engine Cores: {info['neural_engine_cores']}")
    print(f"MLX Default Device: {info['mlx_info']['default_device']}")
    print(f"Max Recommended Model Size: {info['max_recommended_model_size_gb']} GB")
    
    if info['battery_info']:
        print(f"Battery: {info['battery_info']['percentage']}% {'(charging)' if info['battery_info']['charging'] else ''}")
    
    print("\n📋 Generated Configuration:")
    config = detector.generate_device_config()
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
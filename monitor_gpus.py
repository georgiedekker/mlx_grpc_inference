#!/usr/bin/env python3
"""
Monitor GPU activity on both Mac minis during inference.
Shows which GPU is actually processing.
"""
import subprocess
import time
import threading
import mlx.core as mx

def monitor_local_gpu():
    """Monitor local GPU activity."""
    while True:
        try:
            # Get GPU memory usage
            memory = mx.get_active_memory() / (1024**3)
            # Get GPU utilization (this is an approximation)
            print(f"mini1 GPU: {memory:.2f} GB active memory")
        except:
            pass
        time.sleep(1)

def monitor_remote_gpu():
    """Monitor mini2 GPU activity via SSH."""
    while True:
        try:
            # SSH to mini2 and check GPU
            cmd = 'ssh 192.168.5.2 "python3 -c \\"import mlx.core as mx; print(f\'mini2 GPU: {mx.get_active_memory()/(1024**3):.2f} GB active memory\')\\""'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                print(result.stdout.strip())
        except:
            pass
        time.sleep(1)

def main():
    print("Monitoring GPU activity on both Mac minis...")
    print("Make an inference request to see activity")
    print("-" * 40)
    
    # Start monitoring threads
    t1 = threading.Thread(target=monitor_local_gpu, daemon=True)
    t2 = threading.Thread(target=monitor_remote_gpu, daemon=True)
    
    t1.start()
    t2.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor")

if __name__ == "__main__":
    main()
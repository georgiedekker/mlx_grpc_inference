#!/usr/bin/env python
"""
Node launcher that ensures proper environment setup before running distributed_api.
"""
import subprocess
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run distributed_api within uv environment
subprocess.run(["uv", "run", "python", "distributed_api.py"] + sys.argv[1:])
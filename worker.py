#!/usr/bin/env python3
"""
Worker node for distributed inference
This is just a wrapper that calls server.py with worker configuration
"""
import os
import sys

# Ensure this runs as a worker (non-coordinator)
if 'RANK' in os.environ and os.environ['RANK'] == '0':
    print("Error: worker.py should not be run on rank 0 (coordinator)")
    sys.exit(1)

# Import and run the server
from server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
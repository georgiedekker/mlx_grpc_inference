#!/usr/bin/env python3
"""
Server wrapper that selects between single-node and distributed inference
"""
import os

# Check if we want distributed mode
if os.environ.get('DISTRIBUTED', 'false').lower() == 'true':
    # Use distributed implementation
    from server_distributed import main
else:
    # Use single-node implementation
    from server_single import main

import asyncio

if __name__ == "__main__":
    asyncio.run(main())
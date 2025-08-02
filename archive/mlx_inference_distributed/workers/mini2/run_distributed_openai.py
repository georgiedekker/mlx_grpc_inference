#!/usr/bin/env python3
"""
Run the distributed OpenAI-compatible API server.
This script launches the distributed MLX inference API on port 8100.
"""

import uvicorn
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Set environment variables
    os.environ["DISTRIBUTED_CONFIG"] = "distributed_config.json"
    os.environ["API_PORT"] = "8100"
    os.environ["LOCAL_RANK"] = "0"  # Master node
    
    # Run the distributed API
    uvicorn.run(
        "distributed_api:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
        reload=False  # Disable reload for production
    )
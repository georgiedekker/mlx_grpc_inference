#!/bin/bash
# Wrapper script to ensure MPI runs Python within uv environment

# Change to the script directory
cd "$(dirname "$0")"

# Run python within uv environment
exec uv run python "$@"
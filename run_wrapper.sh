#!/bin/bash
# Wrapper script to run Python with or without uv depending on environment

# Detect if we're on mini1 (has uv) or mini2 (use python3 directly)
HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == *"mini1"* ]] && command -v uv &> /dev/null; then
    # On mini1 with uv available
    exec uv run python "$@"
elif [ -f "$HOME/.local/bin/uv" ]; then
    # uv installed in user directory
    exec "$HOME/.local/bin/uv" run python "$@"
else
    # Use system Python directly
    exec python3 "$@"
fi
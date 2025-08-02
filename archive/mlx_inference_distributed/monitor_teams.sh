#!/bin/bash

# MLX Distributed Team Activity Monitor
# Tracks file modifications by different teams

echo "🔍 MLX Distributed - Team Activity Monitor"
echo "========================================="
echo ""

while true; do
    clear
    echo "📊 Team Activity Report - $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Team A files (Backend)
    echo "🔧 Team A (Backend Infrastructure):"
    find /Users/mini1/Movies/mlx_distributed -name "distributed_comm.py" -o -name "distributed_api.py" -o -name "grpc_*.py" -o -name "launch_*.sh" | while read file; do
        if [[ -f "$file" ]]; then
            mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$file")
            echo "  - $(basename "$file"): Last modified $mod_time"
        fi
    done | head -5
    
    echo ""
    
    # Team B files (ML Training)  
    echo "🧠 Team B (ML Training):"
    if [[ -d "/Users/mini1/Movies/mlx_distributed/src/mlx_distributed/training" ]]; then
        find /Users/mini1/Movies/mlx_distributed/src/mlx_distributed/training -name "*.py" -not -path "*/rlhf/*" -mmin -60 | while read file; do
            echo "  - ${file#/Users/mini1/Movies/mlx_distributed/}: Recently modified"
        done | head -5
    fi
    
    echo ""
    
    # Team C files (RLHF)
    echo "🎯 Team C (Research/RLHF):"
    if [[ -d "/Users/mini1/Movies/mlx_distributed/src/mlx_distributed/training/rlhf" ]]; then
        find /Users/mini1/Movies/mlx_distributed/src/mlx_distributed/training/rlhf -name "*.py" -mmin -60 | while read file; do
            echo "  - ${file#/Users/mini1/Movies/mlx_distributed/}: Recently modified"
        done | head -5
    fi
    
    echo ""
    
    # Shared files monitoring
    echo "⚠️  Shared Resources:"
    for file in "model_abstraction.py" "sharding_strategy.py" "distributed_config.py" "pyproject.toml"; do
        if [[ -f "/Users/mini1/Movies/mlx_distributed/$file" ]]; then
            mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "/Users/mini1/Movies/mlx_distributed/$file")
            echo "  - $file: $mod_time"
        fi
    done
    
    echo ""
    echo "🔄 Refreshing in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
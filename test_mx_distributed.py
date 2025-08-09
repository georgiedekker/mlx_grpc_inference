#!/usr/bin/env python3
"""
Test what mx.distributed provides
"""
import mlx.core as mx

print("mx.distributed methods:")
print([m for m in dir(mx.distributed) if not m.startswith('_')])

# Try to init
group = mx.distributed.init()
if group:
    print(f"\nGroup initialized!")
    print(f"Rank: {group.rank()}")
    print(f"Size: {group.size()}")
    print(f"Group methods: {[m for m in dir(group) if not m.startswith('_')]}")
else:
    print("\nNo distributed group (single GPU mode)")
    
# Check available collective operations
print("\nAvailable distributed operations:")
for op in ['all_gather', 'all_reduce', 'all_sum', 'broadcast', 'gather', 'reduce', 'scatter', 'send', 'recv']:
    if hasattr(mx.distributed, op):
        print(f"  ✓ {op}")
    else:
        print(f"  ✗ {op}")
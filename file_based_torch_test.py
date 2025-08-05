#!/usr/bin/env python3
"""File-based PyTorch distributed test - more reliable on macOS"""
import os
import torch
import torch.distributed as dist
import socket
import datetime
import tempfile
import time

def test_with_file_store():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"\n{'='*60}")
    print(f"PyTorch Distributed Test - File Store Method")
    print(f"{'='*60}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Rank: {rank}")
    print(f"World Size: {world_size}")
    
    # Use shared file for coordination
    # Both machines need to access the same path
    file_name = os.environ.get('TORCH_DIST_FILE', '/tmp/torch_distributed_init')
    
    if rank == 0:
        # Create the file and remove old one if exists
        try:
            os.remove(file_name)
        except:
            pass
        # Create empty file
        open(file_name, 'a').close()
        print(f"Created coordination file: {file_name}")
    else:
        # Wait for file to exist
        print(f"Waiting for coordination file: {file_name}")
        for i in range(30):
            if os.path.exists(file_name):
                print("File found!")
                break
            time.sleep(1)
        else:
            raise RuntimeError("Coordination file not found after 30 seconds")
    
    # Use file:// init method
    init_method = f'file://{file_name}'
    print(f"Init method: {init_method}")
    print(f"{'='*60}\n")
    
    try:
        print(f"Rank {rank}: Initializing process group...")
        
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        
        print(f"Rank {rank}: ✅ Initialized successfully!")
        
        # Test 1: All-reduce
        tensor = torch.ones(1) * (rank + 1)
        print(f"\nRank {rank}: Test 1 - All-reduce")
        print(f"Rank {rank}: Local value = {tensor.item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        print(f"Rank {rank}: After all-reduce = {tensor.item()} (expected {expected})")
        
        # Test 2: Send/Recv for pipeline
        if world_size == 2:
            print(f"\nRank {rank}: Test 2 - Send/Recv")
            if rank == 0:
                data = torch.tensor([42.0])
                print(f"Rank 0: Sending {data.item()} to rank 1")
                dist.send(data, dst=1)
            else:
                data = torch.zeros(1)
                dist.recv(data, src=0)
                print(f"Rank 1: Received {data.item()} from rank 0")
        
        # Barrier
        dist.barrier()
        print(f"\nRank {rank}: ✅ All tests completed!")
        
        # Cleanup
        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")
        
        # Remove file on rank 0
        if rank == 0:
            try:
                os.remove(file_name)
                print(f"Rank 0: Removed coordination file")
            except:
                pass
        
    except Exception as e:
        print(f"Rank {rank}: ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_with_file_store())
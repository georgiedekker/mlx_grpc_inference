# Team A: Structured Plan for Real gRPC Implementation

## Current Status
✅ Created real GRPCCommunicator with actual functionality
✅ Implemented proper gRPC service with message handling
✅ Added protocol buffer definitions
❌ Missing generated protobuf files
❌ Not integrated with distributed inference
❌ Not tested with multi-device setup

## Critical Issues to Fix

### Issue 1: Missing Generated Protobuf Files
The code imports `distributed_comm_pb2` and `distributed_comm_pb2_grpc` but these don't exist!

**Fix:**
```bash
# Create the proto file first
cat > distributed_comm.proto << 'EOF'
syntax = "proto3";

package distributed_comm;

// Data types
message TensorData {
    bytes data = 1;
    repeated int32 shape = 2;
    string dtype = 3;
}

message CommData {
    string comm_type = 1;
    oneof data {
        TensorData tensor_data = 2;
        bytes numpy_data = 3;
        bytes pickle_data = 4;
    }
    map<string, string> metadata = 5;
}

// Send/Receive
message SendRequest {
    int32 source_rank = 1;
    int32 dest_rank = 2;
    CommData data = 3;
    string tag = 4;
}

message SendResponse {
    bool success = 1;
    string message = 2;
}

message ReceiveRequest {
    int32 receiver_rank = 1;
    int32 source_rank = 2;
    string tag = 3;
    string comm_type = 4;
}

message ReceiveResponse {
    CommData data = 1;
    int32 source_rank = 2;
    string tag = 3;
}

// Broadcast
message BroadcastRequest {
    int32 sender_rank = 1;
    int32 root_rank = 2;
    CommData data = 3;
}

message BroadcastResponse {
    bool success = 1;
    CommData data = 2;
}

// AllReduce
message AllReduceRequest {
    int32 rank = 1;
    TensorData tensor = 2;
    string operation = 3;
}

message AllReduceResponse {
    bool success = 1;
    TensorData result = 2;
}

// Barrier
message BarrierRequest {
    int32 rank = 1;
    string barrier_id = 2;
}

message BarrierResponse {
    bool success = 1;
    int32 participants = 2;
}

// Service definition
service DistributedComm {
    rpc Send(SendRequest) returns (SendResponse);
    rpc Receive(ReceiveRequest) returns (stream ReceiveResponse);
    rpc Broadcast(BroadcastRequest) returns (BroadcastResponse);
    rpc AllReduce(AllReduceRequest) returns (AllReduceResponse);
    rpc Barrier(BarrierRequest) returns (BarrierResponse);
}
EOF

# Generate Python files
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. distributed_comm.proto
```

### Issue 2: Test the Implementation
Create a test script to verify it works:

```python
# test_grpc_comm.py
import mlx.core as mx
import numpy as np
from distributed_comm import GRPCCommunicator, CommunicationBackend, CommunicationType
import threading
import time

def test_basic_communication():
    """Test basic send/receive between two ranks."""
    comm0 = GRPCCommunicator()
    comm1 = GRPCCommunicator()
    
    # Initialize communicators
    comm0.init(rank=0, world_size=2)
    comm1.init(rank=1, world_size=2)
    
    # Give servers time to start
    time.sleep(0.5)
    
    # Test send/receive
    def rank0_work():
        data = {"message": "Hello from rank 0"}
        comm0.send(data, dest=1)
        response = comm0.receive(source=1)
        print(f"Rank 0 received: {response}")
    
    def rank1_work():
        data = comm1.receive(source=0)
        print(f"Rank 1 received: {data}")
        comm1.send({"reply": "Hello back from rank 1"}, dest=0)
    
    # Run in threads
    t0 = threading.Thread(target=rank0_work)
    t1 = threading.Thread(target=rank1_work)
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    
    # Cleanup
    comm0.finalize()
    comm1.finalize()

if __name__ == "__main__":
    test_basic_communication()
```

### Issue 3: Integration with Distributed System

**Step 1: Update distributed_api.py**
The current code tries to use port from device config, but GRPCCommunicator needs its own ports:

```python
# In distributed_api.py, line 183
# Change from:
communicator.init(rank=local_rank, world_size=world_size, port=config.get_device_by_index(local_rank).port)

# To:
communicator.init(rank=local_rank, world_size=world_size)  # Let it use default ports
```

**Step 2: Fix the launch scripts**
Update launch scripts to properly set environment variables:

```bash
# launch_distributed_grpc.sh
#!/bin/bash

# Start coordinator (rank 0)
LOCAL_RANK=0 python distributed_api.py &

# Start workers (rank 1, 2, etc.)
LOCAL_RANK=1 python distributed_api.py &
LOCAL_RANK=2 python distributed_api.py &

wait
```

## Structured Implementation Plan

### Phase 1: Fix Infrastructure (Day 1)
1. **Generate protobuf files** (30 min)
   ```bash
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. distributed_comm.proto
   ```

2. **Create unit tests** (2 hours)
   - test_grpc_comm.py with basic send/receive
   - test_broadcast.py for broadcast operations
   - test_allreduce.py for reduction operations

3. **Fix receive implementation** (1 hour)
   - Current receive() tries to use local servicer
   - Need to properly call remote service

### Phase 2: Multi-Device Testing (Day 2)
1. **Create multi-process test harness** (2 hours)
   ```python
   # test_multidevice.py
   import multiprocessing as mp
   
   def worker(rank, world_size):
       comm = GRPCCommunicator()
       comm.init(rank, world_size)
       # Test operations
       comm.finalize()
   
   def test_multidevice():
       world_size = 4
       processes = []
       for rank in range(world_size):
           p = mp.Process(target=worker, args=(rank, world_size))
           p.start()
           processes.append(p)
       
       for p in processes:
           p.join()
   ```

2. **Performance benchmarks** (2 hours)
   - Latency measurements
   - Throughput tests
   - Scaling tests

### Phase 3: Integration (Day 3)
1. **Update distributed_mlx_inference.py** (3 hours)
   - Replace MPI references with gRPC
   - Test model sharding
   - Verify tensor communication

2. **Fix distributed_api.py** (2 hours)
   - Proper initialization
   - Error handling
   - Health checks

3. **Update launch scripts** (1 hour)
   - Remove MPI dependencies
   - Use environment variables
   - Add proper cleanup

### Phase 4: Production Ready (Day 4-5)
1. **Error handling**
   - Timeout handling
   - Reconnection logic
   - Graceful degradation

2. **Security**
   - TLS support
   - Authentication
   - Encryption

3. **Monitoring**
   - Metrics collection
   - Logging
   - Debugging tools

## Quick Wins for Today

### 1. Generate Proto Files (5 minutes)
```bash
# Save the proto file and generate
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. distributed_comm.proto
```

### 2. Fix Import Issue (10 minutes)
Add to top of distributed_comm.py:
```python
try:
    import distributed_comm_pb2
    import distributed_comm_pb2_grpc
except ImportError:
    print("Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. distributed_comm.proto")
    raise
```

### 3. Create Working Example (30 minutes)
```python
# example_distributed_inference.py
from distributed_comm import GRPCCommunicator
import mlx.core as mx

# Simple test
comm = GRPCCommunicator()
comm.init(rank=0, world_size=1)

# Test array communication
test_array = mx.array([1, 2, 3, 4])
print(f"Test array: {test_array}")

comm.finalize()
```

## Common Pitfalls to Avoid

1. **Port Conflicts**
   - Each rank needs a unique port
   - Default base_port is 50100
   - Avoid using ports already taken by other services

2. **Synchronization Issues**
   - Always wait for servers to start
   - Use proper barriers
   - Handle timeouts gracefully

3. **Memory Management**
   - Clean up channels properly
   - Avoid memory leaks in queues
   - Limit message sizes

## Success Metrics

1. **Functional**: All communication primitives work
2. **Performance**: <1ms latency for small messages
3. **Scalable**: Works with 8+ devices
4. **Reliable**: No dropped messages
5. **Integrated**: Works with distributed inference

## The Path Forward

Team A has made progress but needs to:
1. **Stop making placeholder code**
2. **Test with real multi-device scenarios**
3. **Integrate with the rest of the system**

The current implementation is much better than stubs, but it needs testing and integration to be useful!
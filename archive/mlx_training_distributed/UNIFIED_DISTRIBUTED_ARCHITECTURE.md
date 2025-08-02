# Unified MLX Distributed Inference Architecture

## Overview

This document defines a unified, working architecture for the MLX distributed inference system across 3 M4 Mac devices (mini1, mini2, master) with flexible coordinator selection, proper model sharding, OpenAI-compatible API, and reliable communication.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   mini1 Node    │  │   mini2 Node    │  │ master Node  │ │
│  │ Port: 8100-8199 │  │ Port: 8200-8299 │  │Port:8300-8399│ │
│  │                 │  │                 │  │              │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │┌────────────┐│ │
│  │ │Inference API│ │  │ │Inference API│ │  ││Inference API││ │
│  │ │   :8100     │ │  │ │   :8200     │  │  ││   :8300    ││ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │└────────────┘│ │
│  │                 │  │                 │  │              │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │┌────────────┐│ │
│  │ │Model Shard  │ │  │ │Model Shard  │ │  ││Model Shard ││ │
│  │ │ Layers 0-9  │ │  │ │Layers 10-18 │ │  ││Layers19-27 ││ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │└────────────┘│ │
│  │                 │  │                 │  │              │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │┌────────────┐│ │
│  │ │Communication│ │  │ │Communication│ │  ││Communication││ │
│  │ │   :8101     │ │  │ │   :8201     │  │  ││   :8301    ││ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │└────────────┘│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                │
    ┌───────────────────────────┴───────────────────────────┐
    │              Coordinator System                       │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │     Dynamic Leader Election                      │  │
    │  │  - Health-based selection                       │  │
    │  │  - Load-based selection                         │  │
    │  │  - Manual override capability                   │  │
    │  └─────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────┘
                                │
                                │
    ┌───────────────────────────┴───────────────────────────┐
    │           OpenAI-Compatible API Gateway               │
    │                     Port: 8000                        │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ /v1/chat/completions                            │  │
    │  │ /v1/completions                                 │  │
    │  │ /v1/models                                      │  │
    │  │ /health                                         │  │
    │  └─────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Clear Separation of Concerns

- **API Gateway**: Handles client requests and OpenAI compatibility
- **Coordinator**: Manages distributed state and orchestration
- **Inference Nodes**: Execute model inference on sharded layers
- **Communication Layer**: Handles inter-node messaging
- **Configuration System**: Manages deployment and runtime config

### 2. Consistent Port Assignment

```yaml
Port Allocation:
  API Gateway: 8000
  mini1 (Node 0):
    - Inference API: 8100
    - Communication: 8101
    - Health/Metrics: 8102
  mini2 (Node 1):
    - Inference API: 8200  
    - Communication: 8201
    - Health/Metrics: 8202
  master (Node 2):
    - Inference API: 8300
    - Communication: 8301
    - Health/Metrics: 8302
  Coordinator Service: 8400
```

### 3. Flexible Coordinator Selection

The system supports multiple coordinator selection strategies:

- **Health-based**: Node with best health metrics becomes coordinator
- **Load-based**: Node with lowest load becomes coordinator  
- **Round-robin**: Rotate coordinator periodically
- **Manual**: Administrator can manually assign coordinator
- **Failover**: Automatic failover if coordinator becomes unavailable

### 4. Proper Model Sharding (10-9-9 layers)

```yaml
Model Sharding Strategy:
  Total Layers: 28 (example for Qwen2.5-7B)
  
  mini1 (Node 0):
    - Layers: 0-9 (10 layers)
    - Role: Input processing, initial transformations
    - Memory: ~2.5GB
    
  mini2 (Node 1):  
    - Layers: 10-18 (9 layers)
    - Role: Middle transformations
    - Memory: ~2.2GB
    
  master (Node 2):
    - Layers: 19-27 (9 layers) + Output head
    - Role: Final transformations, output generation
    - Memory: ~2.2GB + output head
```

## Component Specifications

### API Gateway (Port 8000)

**Responsibilities:**
- OpenAI-compatible endpoint exposure
- Request routing to coordinator
- Response aggregation
- Authentication and rate limiting
- Error handling and retry logic

**Endpoints:**
- `POST /v1/chat/completions` - Chat completion requests
- `POST /v1/completions` - Text completion requests  
- `GET /v1/models` - List available models
- `GET /health` - System health check
- `GET /metrics` - System metrics

### Coordinator Service (Port 8400)

**Responsibilities:**
- Node discovery and health monitoring
- Load balancing and request routing
- Model shard coordination
- Failure detection and recovery
- Configuration management

**Key Features:**
- Leader election algorithm
- Health check orchestration
- Request queuing and distribution
- Graceful degradation on node failure

### Inference Nodes (Ports 8100, 8200, 8300)

**Responsibilities:**
- Model shard loading and management
- Layer-wise inference execution
- Inter-node communication
- Local health monitoring
- Memory and resource management

**Communication Protocol:**
- gRPC for high-performance inter-node communication
- Protocol Buffers for message serialization
- Async/await for non-blocking operations
- Automatic retry with exponential backoff

### Configuration System

**Configuration Hierarchy:**
1. Default configurations (embedded)
2. Node-specific configuration files
3. Environment variables
4. Runtime dynamic configuration
5. Manual overrides

**Configuration Files:**
- `config/cluster.yaml` - Cluster-wide settings
- `config/nodes.yaml` - Per-node configurations  
- `config/models.yaml` - Model and sharding configurations
- `config/network.yaml` - Network and communication settings

## Implementation Architecture

### Core Classes

```python
# Coordinator System
class CoordinatorService:
    - node_registry: NodeRegistry
    - leader_election: LeaderElection  
    - load_balancer: LoadBalancer
    - health_monitor: HealthMonitor

class NodeRegistry:
    - register_node()
    - get_active_nodes()
    - update_node_status()

class LeaderElection:
    - elect_leader()
    - handle_leader_failure()
    - manual_override()

# Inference System  
class InferenceNode:
    - model_shard: ModelShard
    - communication: NodeCommunication
    - health_reporter: HealthReporter

class ModelShard:
    - load_layers()
    - forward_pass()
    - get_memory_usage()

class PipelineInference:
    - execute_pipeline()
    - handle_layer_communication()
    - aggregate_results()

# Communication Layer
class NodeCommunication:
    - send_tensor()
    - receive_tensor()
    - broadcast_state()
    - handle_connection_failure()

# API Layer
class APIGateway:
    - route_request()
    - handle_openai_format()
    - aggregate_response()
    - handle_errors()
```

### Error Handling Strategy

1. **Node Failure Recovery:**
   - Automatic detection of node failures
   - Dynamic rebalancing of model shards
   - Graceful degradation to available nodes
   - Coordinator failover mechanisms

2. **Communication Errors:**
   - Automatic retry with exponential backoff
   - Alternative routing paths
   - Circuit breaker pattern
   - Timeout handling

3. **Resource Constraints:**
   - Memory monitoring and alerts
   - Dynamic batch size adjustment
   - Queue management for overload
   - Graceful request rejection

### Performance Optimizations

1. **Memory Management:**
   - Model shard preloading
   - Efficient tensor transfer
   - Memory pool allocation
   - Garbage collection optimization

2. **Network Optimization:**
   - Tensor compression for transfer
   - Pipelined execution
   - Asynchronous communication
   - Connection pooling

3. **Caching:**
   - Model weight caching
   - Intermediate result caching
   - Request deduplication
   - Smart prefetching

## Deployment Configuration

### Environment Setup

```yaml
# cluster.yaml
cluster:
  name: "mlx-distributed-cluster"
  nodes:
    - host: "mini1.local"
      node_id: 0
      ports: [8100, 8101, 8102]
    - host: "mini2.local" 
      node_id: 1
      ports: [8200, 8201, 8202]
    - host: "master.local"
      node_id: 2  
      ports: [8300, 8301, 8302]
      
coordinator:
  port: 8400
  election_strategy: "health_based"
  health_check_interval: 5
  failure_threshold: 3

api_gateway:
  port: 8000
  timeout: 30
  max_concurrent_requests: 100
```

### Model Configuration

```yaml
# models.yaml
models:
  qwen2.5-7b:
    total_layers: 28  
    sharding:
      node_0: 
        layers: [0, 9]
        estimated_memory: "2.5GB"
      node_1:
        layers: [10, 18] 
        estimated_memory: "2.2GB"
      node_2:
        layers: [19, 27]
        estimated_memory: "2.2GB"
        includes_output_head: true
```

## Security Considerations

1. **Authentication:**
   - API key validation
   - JWT token support
   - Node-to-node authentication

2. **Network Security:**
   - TLS encryption for all communication
   - Certificate management
   - Network isolation

3. **Access Control:**
   - Role-based access control
   - Request rate limiting
   - IP whitelisting

## Monitoring and Observability

1. **Health Monitoring:**
   - Node health checks
   - Resource utilization monitoring
   - Model performance metrics

2. **Logging:**
   - Structured logging across all components
   - Centralized log aggregation
   - Request tracing

3. **Metrics:**
   - Inference latency and throughput
   - Resource utilization
   - Error rates and types

## Testing Strategy

1. **Unit Tests:**
   - Individual component testing
   - Mock external dependencies
   - Edge case validation

2. **Integration Tests:**
   - End-to-end pipeline testing
   - Node failure scenarios
   - Load testing

3. **Performance Tests:**
   - Latency benchmarks
   - Throughput measurements
   - Resource utilization tests

This architecture provides a robust, scalable, and maintainable foundation for the MLX distributed inference system while meeting all specified requirements.
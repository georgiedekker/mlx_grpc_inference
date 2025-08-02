# MLX Distributed Inference Architecture Plan

## 🎯 Project Overview

A distributed inference system for MLX models across 3 Apple Silicon devices using gRPC for communication. The system splits large language models (like Qwen3-1.7B) across multiple devices to leverage their combined GPU power.

**Current Status: PARTIALLY IMPLEMENTED - Single device working, distributed components blocked by DNS issues**

## 📋 Requirements

### Hardware Setup
- **Device 1 (Coordinator)**: mini1.local - Apple M4, 16GB RAM
- **Device 2 (Worker)**: mini2.local - Apple M4, 16GB RAM  
- **Device 3 (Worker)**: master.local - Apple M4, 16GB RAM (user: georgedekker)

### Core Goals
1. **Model Sharding**: Split 28-layer Qwen model across 3 devices (10-9-9 distribution) ⚠️ *Implemented but not tested*
2. **GPU Utilization**: All 3 devices should show GPU activity during inference ❌ *Not functional*
3. **OpenAI-Compatible API**: REST endpoint on port 8100 ✅ *Implemented but limited to single device*
4. **High Performance**: Minimal tensor serialization overhead ✅ *Implemented*
5. **Fault Tolerance**: Graceful degradation if workers fail ❌ *Not implemented*
6. **Monitoring**: Real-time GPU utilization tracking ⚠️ *Partially implemented*

## 🏗️ System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                        │
│                    (OpenAI-compatible API calls)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Coordinator (mini1:8100)                       │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   REST API      │  │ Inference    │  │ Model Layers    │    │
│  │   Server        │◄─┤ Orchestrator │──┤    0-9          │    │
│  └─────────────────┘  └──────┬───────┘  └─────────────────┘    │
└───────────────────────────────┼─────────────────────────────────┘
                                │ gRPC (Port 50051)
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Worker 1 (mini2)       │    │   Worker 2 (master)      │
│  ┌─────────────────┐     │    │  ┌─────────────────┐     │
│  │  gRPC Server    │     │    │  │  gRPC Server    │     │
│  ├─────────────────┤     │    │  ├─────────────────┤     │
│  │ Model Layers    │     │    │  │ Model Layers    │     │
│  │    10-18        │     │    │  │    19-27        │     │
│  └─────────────────┘     │    │  └─────────────────┘     │
└──────────────────────────┘    └──────────────────────────┘
```

### Communication Flow

1. **Client → Coordinator**: HTTP POST to `/v1/chat/completions`
2. **Coordinator**: 
   - Tokenizes input
   - Processes layers 0-9
   - Sends intermediate tensors to Worker 1
3. **Worker 1**: 
   - Receives tensors via gRPC
   - Processes layers 10-18
   - Sends results to Worker 2
4. **Worker 2**:
   - Receives tensors via gRPC
   - Processes layers 19-27
   - Returns final output to Coordinator
5. **Coordinator → Client**: Formats and returns response

## 📁 Project Structure

```
mlx_grpc_inference/
│
├── README.md                    # Project overview and setup instructions
├── ARCHITECTURE_PLAN.md         # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # UV project configuration
│
├── config/
│   ├── cluster_config.yaml     # Cluster configuration
│   └── model_config.yaml       # Model-specific settings
│
├── protos/
│   ├── inference.proto         # gRPC service definitions
│   └── generate_protos.sh      # Script to generate Python code
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration classes
│   │   ├── device.py           # Device abstraction
│   │   └── model_info.py       # Model metadata
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── loader.py           # Model loading utilities
│   │   ├── sharding.py         # Layer distribution logic
│   │   └── inference.py        # Local inference execution
│   │
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── grpc_server.py      # gRPC server implementation
│   │   ├── grpc_client.py      # gRPC client for inter-device comm
│   │   ├── tensor_utils.py     # MLX tensor serialization
│   │   └── connection_pool.py  # Connection management
│   │
│   ├── coordinator/
│   │   ├── __init__.py
│   │   ├── api_server.py       # FastAPI REST server
│   │   ├── orchestrator.py     # Inference orchestration
│   │   └── request_handler.py  # Request processing
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── worker_server.py    # Worker node implementation
│   │   ├── layer_processor.py  # Layer computation
│   │   └── health_monitor.py   # Health checks
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── gpu_monitor.py      # GPU utilization tracking
│       ├── metrics.py          # Performance metrics
│       └── dashboard.py        # Terminal dashboard
│
├── scripts/
│   ├── start_cluster.sh        # Start all nodes
│   ├── stop_cluster.sh         # Stop all nodes
│   ├── test_inference.py       # Test distributed inference
│   └── monitor_gpus.py         # GPU monitoring script
│
├── tests/
│   ├── __init__.py
│   ├── test_sharding.py        # Test model sharding
│   ├── test_serialization.py   # Test tensor serialization
│   ├── test_grpc.py           # Test gRPC communication
│   └── test_e2e.py            # End-to-end tests
│
└── logs/                       # Runtime logs (gitignored)
    ├── coordinator.log
    ├── worker_mini2.log
    └── worker_master.log
```

## 🔧 Key Components

### 1. Configuration System (`src/core/config.py`)
- YAML-based configuration
- Device capabilities detection
- Dynamic layer assignment
- Network topology management

### 2. Model Sharding (`src/model/sharding.py`)
- Intelligent layer distribution based on device capabilities
- Support for different model architectures
- Cached shard information
- Memory-efficient loading

### 3. Tensor Serialization (`src/communication/tensor_utils.py`)
- Efficient MLX array serialization
- Compression support (optional)
- Type preservation (bool, float16, etc.)
- Zero-copy where possible

### 4. gRPC Communication
- Protocol Buffers for efficiency
- Streaming support for large tensors
- Connection pooling
- Automatic retry with exponential backoff
- Health checking

### 5. Orchestration (`src/coordinator/orchestrator.py`)
- Request queuing
- Load balancing
- Failure detection and recovery
- Performance monitoring

### 6. Worker Management (`src/worker/worker_server.py`)
- Stateless workers
- Layer caching
- Resource monitoring
- Graceful shutdown

## 🔌 API Design

### REST API (Coordinator)

```
POST /v1/chat/completions
{
    "model": "mlx-community/Qwen3-1.7B-8bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
}

GET /health
GET /cluster/status
GET /metrics
```

### gRPC Services

```protobuf
service InferenceService {
    // Process layers on this worker
    rpc ProcessLayers(LayerRequest) returns (LayerResponse);
    
    // Health check
    rpc HealthCheck(Empty) returns (HealthStatus);
    
    // Get device info
    rpc GetDeviceInfo(Empty) returns (DeviceInfo);
}

message LayerRequest {
    string request_id = 1;
    bytes input_tensor = 2;
    repeated int32 layer_indices = 3;
    TensorMetadata metadata = 4;
}

message LayerResponse {
    string request_id = 1;
    bytes output_tensor = 2;
    TensorMetadata metadata = 3;
    float processing_time_ms = 4;
}
```

## 🚀 Implementation Phases

### Phase 1: Foundation (Days 1-2)
1. Set up project structure
2. Create configuration system
3. Implement basic gRPC communication
4. Test tensor serialization

### Phase 2: Model Integration (Days 3-4)
1. Implement model loading and sharding
2. Create layer processor
3. Test single-device inference
4. Verify memory efficiency

### Phase 3: Distributed System (Days 5-6)
1. Implement orchestrator
2. Create worker nodes
3. Test multi-device communication
4. Add failure handling

### Phase 4: API & Monitoring (Days 7-8)
1. Implement REST API
2. Add GPU monitoring
3. Create dashboard
4. Performance optimization

### Phase 5: Testing & Deployment (Days 9-10)
1. Comprehensive testing
2. Documentation
3. Deployment scripts
4. Performance benchmarking

## 📊 Performance Targets

- **Latency**: < 100ms overhead for distributed vs single-device
- **Throughput**: > 50 tokens/second for Qwen3-1.7B
- **Memory**: < 6GB per device for model shards
- **Network**: < 50MB/s during inference
- **Availability**: 99.9% uptime with graceful degradation

## 🛡️ Security & Best Practices

1. **Authentication**: API key for REST endpoints
2. **Encryption**: TLS for gRPC (optional for LAN)
3. **Input Validation**: Sanitize all inputs
4. **Resource Limits**: Prevent DoS attacks
5. **Logging**: Structured logs with request IDs
6. **Monitoring**: Prometheus metrics export
7. **Error Handling**: No sensitive info in errors

## 🧪 Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Component interactions
3. **Load Tests**: Performance under stress
4. **Chaos Tests**: Failure scenarios
5. **GPU Tests**: Verify multi-device utilization

## 📊 Implementation Status

### ✅ Fully Working Components

1. **Project Structure & Build System**
   - Complete UV-based project setup with proper dependencies
   - Well-organized src/ directory structure
   - Generated gRPC protocol buffers from `protos/inference.proto`

2. **Configuration System** (`src/core/config.py`)
   - YAML-based cluster configuration with device capabilities
   - Dynamic device detection and role assignment
   - Layer distribution mapping across devices
   - Hardware-aware configuration loading

3. **Model Components**
   - **Model Loader** (`src/model/loader.py`): Loading MLX models with sharding support
   - **Sharding Strategy** (`src/model/sharding.py`): Layer distribution logic with validation
   - **Inference Engine** (`src/model/inference.py`): Local layer processing

4. **Communication Infrastructure**
   - **Tensor Serialization** (`src/communication/tensor_utils.py`): MLX array serialization/deserialization
   - **DNS Resolver** (`src/communication/dns_resolver.py`): .local hostname resolution with caching
   - **gRPC Client/Server** implementations with message size optimization

5. **Single Device Operation**
   - Model loading and inference works perfectly on single device
   - OpenAI-compatible API endpoints defined in FastAPI
   - Local testing confirms model functionality

### ⚠️ Implemented But Blocked

1. **Distributed Orchestrator** (`src/coordinator/orchestrator.py`)
   - Complete implementation for coordinating distributed inference
   - Blocked by worker connectivity issues
   - Cannot initialize due to failed health checks

2. **Worker Nodes** (`src/worker/worker_server.py`)
   - Full gRPC server implementation for layer processing
   - Not deployed on remote devices (mini2, master)
   - Health check endpoints implemented but unreachable

3. **gRPC Communication**
   - Protocol and client/server code complete
   - DNS resolution implemented but still failing for .local domains
   - Connection pooling and retry logic in place

### ❌ Critical Blockers

1. **DNS Resolution Failure**
   ```
   DNS resolution failed for mini2.local:50051: C-ares status is not ARES_SUCCESS
   qtype=A name=mini2.local is_balancer=0: Domain name not found
   ```
   - gRPC's C-ares resolver cannot resolve .local hostnames
   - Custom DNS resolver implemented but not fully effective
   - Network connectivity prerequisite not met

2. **Worker Deployment**
   - No worker processes running on mini2.local or master.local
   - SSH access to master.local (georgedekker@master.local) not configured
   - Remote deployment scripts exist but not functional

3. **Network Infrastructure**
   - mDNS/.local hostname resolution inconsistent across devices
   - No fallback to IP address resolution working
   - gRPC channel options not resolving C-ares limitations

### ❌ Missing Components

1. **Complete Monitoring System**
   - `src/monitoring/` directory exists but is empty (only `__init__.py`)
   - GPU utilization tracking not implemented
   - Performance metrics collection missing
   - Dashboard for cluster health not built

2. **Fault Tolerance**
   - No graceful degradation when workers fail
   - No automatic retry mechanisms
   - No worker health recovery logic

3. **Production Features**
   - No authentication/security for API endpoints
   - No rate limiting or resource management
   - No proper logging and error handling for production

## 🔍 Technical Debt

### Code Quality Issues

1. **Import Dependencies**
   - Several files import generated protobuf code that may not exist
   - Runtime imports in functions make debugging difficult
   - Missing proper error handling for import failures

2. **Error Handling**
   - DNS resolution failures not gracefully handled
   - gRPC connection errors cause complete system failure
   - No fallback mechanisms implemented

3. **Testing Coverage**
   - Multiple test files exist but no comprehensive test suite
   - End-to-end testing not functional due to network issues
   - Mock testing for distributed components missing

### Architecture Decisions

1. **DNS Strategy**
   - Reliance on .local hostnames problematic for gRPC
   - Should implement IP-based configuration option
   - mDNS resolution not reliable across all network configurations

2. **Deployment Model**
   - Manual deployment to remote devices not scalable
   - No container-based deployment option
   - SSH-based worker startup fragile

## 📈 Implementation Reality vs Original Plan

### Phase Progress Assessment

| Phase | Original Goal | Actual Status | Completion |
|-------|---------------|---------------|------------|
| Phase 1 | Foundation setup | DNS issues blocking | 85% |
| Phase 2 | Model integration | Single device works | 90% |
| Phase 3 | Distributed system | Cannot connect workers | 30% |
| Phase 4 | API & monitoring | API works, monitoring empty | 50% |
| Phase 5 | Testing & deployment | Blocked by connectivity | 20% |

### Successful Achievements ✅

1. **Solid Architecture Foundation**: Well-designed modular system
2. **MLX Integration**: Successfully working with MLX models and tokenizers
3. **Configuration Management**: Robust YAML-based configuration system
4. **Tensor Serialization**: Efficient MLX array serialization working
5. **Single Device Performance**: Model loading and inference functional
6. **Code Quality**: Clean, readable codebase following Python best practices

### Critical Gaps ❌

1. **Multi-Device Connectivity**: Cannot establish connections between devices
2. **Distributed Inference**: No successful end-to-end distributed inference
3. **Worker Management**: No workers running on remote devices
4. **Monitoring Infrastructure**: Monitoring system not implemented
5. **Production Readiness**: Missing authentication, logging, error handling

## 🚨 Current Status Summary

**Architecture Status: WELL-DESIGNED BUT NOT FUNCTIONAL FOR DISTRIBUTED USE**

The system has an excellent architectural foundation with clean abstractions and proper separation of concerns. All single-device functionality works perfectly. However, the distributed aspects are completely non-functional due to network connectivity issues.

**What Works:**
- ✅ Single device MLX inference with Qwen3-1.7B
- ✅ OpenAI-compatible API server (single device mode)
- ✅ Model loading, tokenization, and tensor operations
- ✅ Configuration system and device detection
- ✅ gRPC protocol definitions and serialization
- ✅ Clean, maintainable codebase

**What's Broken:**
- ❌ All multi-device communication
- ❌ Worker node deployment and connectivity
- ❌ Distributed inference orchestration
- ❌ GPU utilization across multiple devices
- ❌ System monitoring and health checks

**Root Cause:**
DNS resolution of .local hostnames with gRPC's C-ares resolver is the primary blocker preventing any distributed functionality from working.

## 🎯 Next Steps

### Immediate Priorities (Critical Path)

1. **Fix Network Connectivity**
   - Replace .local hostnames with IP addresses in configuration
   - Test direct IP-based gRPC connections
   - Implement network discovery alternative to mDNS

2. **Deploy Worker Nodes**
   - Set up SSH access to master.local (georgedekker user)
   - Deploy and start worker processes on mini2 and master
   - Verify worker health checks working

3. **Complete Monitoring System**
   - Implement GPU monitoring in `src/monitoring/`
   - Add performance metrics collection
   - Create cluster health dashboard

### Technical Improvements

4. **Enhanced Error Handling**
   - Add graceful degradation for worker failures
   - Implement connection retry and recovery logic
   - Better error messages and logging

5. **Production Features**
   - Add API authentication and rate limiting
   - Implement proper logging and metrics
   - Add comprehensive testing suite

6. **Performance Optimization**
   - Optimize tensor serialization for large models
   - Implement connection pooling improvements
   - Add caching for frequently used operations

## 🔧 Known Issues

### High Priority Issues

1. **DNS Resolution with gRPC C-ares**
   - **Issue**: gRPC cannot resolve .local hostnames using C-ares resolver
   - **Error**: `DNS resolution failed for mini2.local:50051: C-ares status is not ARES_SUCCESS`
   - **Impact**: Complete failure of distributed functionality
   - **Workaround**: None currently working
   - **Fix Required**: Switch to IP-based addressing or implement custom gRPC resolver

2. **Worker Process Deployment**
   - **Issue**: No automated deployment to remote devices
   - **Impact**: Cannot test distributed inference end-to-end
   - **Dependencies**: SSH access to master.local (georgedekker user)
   - **Current State**: Manual scripts exist but not functional

3. **Network Discovery**
   - **Issue**: mDNS/.local hostname resolution unreliable
   - **Impact**: Cannot automatically discover devices on network
   - **Current State**: Hard-coded device configurations only

### Medium Priority Issues

4. **Import Error Handling**
   - **Issue**: Runtime imports of generated protobuf code can fail
   - **Files Affected**: `grpc_server.py`, `grpc_client.py`, `orchestrator.py`
   - **Impact**: Debugging difficulties and potential runtime crashes
   - **Fix**: Add proper import error handling and generation verification

5. **Connection Pool Management**
   - **Issue**: gRPC connections not properly managed or reused
   - **Impact**: Resource leaks and connection overhead
   - **Current State**: Basic connection pooling implemented but untested

6. **Configuration Validation**
   - **Issue**: No validation of cluster configuration consistency
   - **Impact**: Silent failures with invalid configurations
   - **Examples**: Overlapping layer assignments, missing device capabilities

### Low Priority Issues

7. **Error Message Quality**
   - **Issue**: Technical error messages not user-friendly
   - **Impact**: Difficult troubleshooting for non-technical users
   - **Fix**: Add user-friendly error translations and suggestions

8. **Resource Cleanup**
   - **Issue**: No proper cleanup of resources on shutdown
   - **Impact**: Potential resource leaks in long-running processes
   - **Areas**: gRPC channels, model weights, temporary files

## 🏗️ Technical Debt

### Code Architecture Debt

1. **Hardcoded Dependencies**
   - **Location**: Multiple files with hardcoded model names and paths
   - **Risk**: Difficult to adapt to new models or deployment environments
   - **Effort**: Medium - requires configuration refactoring

2. **Mixed Synchronous/Asynchronous Code**
   - **Issue**: Inconsistent use of async/await patterns
   - **Files**: `orchestrator.py` mixes sync and async calls
   - **Risk**: Performance issues and potential deadlocks
   - **Effort**: High - requires significant refactoring

3. **Error Handling Inconsistency**
   - **Issue**: Different error handling patterns across modules
   - **Risk**: Unpredictable failure behavior
   - **Fix**: Standardize exception handling and logging patterns

### Infrastructure Debt

4. **No Containerization**
   - **Issue**: Manual deployment and dependency management
   - **Risk**: Environment inconsistencies and deployment complexity
   - **Solution**: Add Docker/container support

5. **Limited Testing Infrastructure**
   - **Issue**: No CI/CD, limited automated testing
   - **Current**: Only ad-hoc test scripts
   - **Need**: Comprehensive pytest suite with mocking

6. **Manual Configuration Management**
   - **Issue**: No automated configuration generation or validation
   - **Risk**: Configuration drift and human errors
   - **Solution**: Configuration management tools and validation

### Performance Debt

7. **Tensor Serialization Overhead**
   - **Issue**: No optimization for large tensor transfers
   - **Impact**: High latency for distributed inference
   - **Optimization**: Implement compression and streaming

8. **Model Loading Efficiency**
   - **Issue**: Full model loaded on each device (redundant data)
   - **Impact**: High memory usage and initialization time
   - **Solution**: Implement true model sharding with minimal redundancy

## 🚨 Known Bugs

### Critical Bugs

1. **gRPC Connection Hangs**
   - **Trigger**: Attempting to connect to unreachable .local hostname
   - **Behavior**: Application hangs indefinitely without timeout
   - **Workaround**: None
   - **Root Cause**: gRPC default timeout handling

2. **Model Loading Memory Leak**
   - **Trigger**: Multiple model loading attempts after failures
   - **Behavior**: Memory usage increases with each attempt
   - **Impact**: Eventually causes OOM on coordinator
   - **Workaround**: Restart process

### Non-Critical Bugs

3. **Log Message Formatting**
   - **Issue**: Inconsistent log formats across modules
   - **Impact**: Difficult log parsing and monitoring
   - **Fix**: Standardize logging configuration

4. **Configuration Path Resolution**
   - **Issue**: Relative paths in configuration not properly resolved
   - **Impact**: Configuration loading failures in different working directories
   - **Fix**: Use absolute path resolution

## 🔮 Next Steps

### Phase 1: Core Functionality (Week 1-2)

1. **Network Connectivity Resolution**
   - Replace .local hostnames with IP addresses in cluster configuration
   - Implement network discovery using ping/arp instead of mDNS
   - Add IP address validation and reachability testing
   - Test direct gRPC connections with IP addresses

2. **Worker Deployment**
   - Configure SSH access to master.local with georgedekker user
   - Create automated deployment scripts using SSH/scp
   - Implement worker process management (start/stop/health check)
   - Test worker nodes responding to health checks

3. **Basic Distributed Inference**
   - Fix import issues in distributed components
   - Implement end-to-end single token generation across devices
   - Add proper error handling for worker failures
   - Verify layer processing pipeline working

### Phase 2: Monitoring & Stability (Week 3-4)

4. **Complete Monitoring System**
   - Implement GPU utilization monitoring for Apple Silicon
   - Add performance metrics collection (latency, throughput)
   - Create simple terminal dashboard for cluster health
   - Add logging and metrics export

5. **Error Handling & Recovery**
   - Implement graceful degradation when workers fail
   - Add connection retry and circuit breaker patterns
   - Improve error messages and user feedback
   - Add configuration validation

6. **Testing Infrastructure**
   - Create comprehensive test suite with pytest
   - Add mock workers for testing distributed logic
   - Implement integration tests for end-to-end flows
   - Add performance benchmarking tests

### Phase 3: Production Features (Week 5-6)

7. **Production Readiness**
   - Add API authentication and rate limiting
   - Implement proper logging with structured formats
   - Add health checks and readiness probes
   - Create deployment documentation

8. **Performance Optimization**
   - Optimize tensor serialization with compression
   - Implement connection pooling improvements
   - Add caching for model weights and metadata
   - Profile and optimize hot paths

9. **Advanced Features**
   - Add support for multiple model types
   - Implement dynamic device addition/removal
   - Add load balancing for worker requests
   - Create web-based monitoring dashboard

### Long-term Improvements

10. **Containerization & Deployment**
    - Create Docker images for coordinator and workers
    - Add Kubernetes deployment manifests
    - Implement container-based auto-scaling
    - Add service mesh integration for advanced networking

---

**Bottom Line: Solid architectural foundation requires network connectivity resolution and deployment automation to become a functional distributed inference system. The codebase quality is high, but critical infrastructure gaps prevent distributed operation.**
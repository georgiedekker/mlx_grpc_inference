# MLX Distributed - Team Coordination Guide

## Overview
This document tracks the parallel development efforts of three independent teams working on the MLX Distributed project. Each team operates in a separate window without knowledge of the others' work.

## Active Teams

### Team A: Backend Infrastructure (Window 1)
**Focus**: Fix distributed infrastructure & API
**Key Files**:
- `distributed_comm.py` - Removing MPI, pure gRPC
- `distributed_api.py` - Fix initialization issues
- `grpc_server.py`, `grpc_client.py` - Core communication
- Launch scripts in root directory

**Critical Shared Resources**:
- gRPC protobuf definitions (`protos/`)
- Port 8000 for API server
- Port 50051+ for gRPC nodes

### Team B: ML Training (Window 2)
**Focus**: Implement distributed training system
**Key Files**:
- `src/mlx_distributed/training/distributed_trainer.py` (NEW)
- `src/mlx_distributed/training/sft/` (NEW)
- `mlx_lm/` imports for base training
- Training configuration files

**Critical Shared Resources**:
- Model abstraction layer (`model_abstraction.py`)
- Sharding strategies (`sharding_strategy.py`)
- Device capabilities (`device_capabilities.py`)

### Team C: Research/RLHF (Window 3)
**Focus**: RLHF implementation & advanced features
**UPDATE**: Team C has created **mlx_rlhf** as a completely standalone package!

**New Structure**:
- Separate repository/directory: `mlx_rlhf/`
- Own `setup.py` and `pyproject.toml`
- No dependencies on `mlx_distributed`
- Self-contained distributed utilities

**Benefits**:
- ✅ Zero collision risk with other teams
- ✅ Independent development and testing
- ✅ Can be installed separately: `pip install -e mlx_rlhf`
- ✅ Own documentation and examples

## Collision Prevention Rules

### 1. File Ownership
- **Exclusive to Team A**: 
  - `distributed_comm.py`, `distributed_api.py`
  - All launch scripts (`launch_*.sh`)
  - `mpi_wrapper.sh`, `hostfile.txt` (for deletion)

- **Exclusive to Team B**:
  - `src/mlx_distributed/training/` (except rlhf/)
  - Training-specific configs

- **Exclusive to Team C**:
  - `src/mlx_distributed/training/rlhf/`
  - Monitoring configurations

### 2. Shared File Protocol
For files that multiple teams need to modify:

**High-Risk Shared Files**:
1. `pyproject.toml` - Coordinate dependency additions
2. `model_abstraction.py` - Notify before modifying
3. `sharding_strategy.py` - Coordinate strategy additions
4. `distributed_config.py` - Synchronize config changes

**Protocol**:
- Check with coordinator before modifying shared files
- Make atomic, focused changes
- Add clear comments with team identifier
- Test changes don't break other teams' work

### 3. API Contracts
Teams must maintain these interfaces:

**Team A → Team B**:
```python
# Distributed inference client interface
class DistributedInferenceClient:
    async def forward_pipeline(input_ids, cache) -> (logits, cache)
```

**Team B → Team C**:
```python
# SFT trainer interface
class SFTTrainer:
    def train(model, dataset, config) -> trained_model
```

### 4. Configuration Namespace
Each team uses designated config sections:
```json
{
  "inference": {},  // Team A
  "training": {     // Team B
    "sft": {}
  },
  "rlhf": {}       // Team C
}
```

## Communication Protocol

### Daily Sync Points
1. **Morning**: Check for overnight changes
2. **Midday**: Coordinate any shared file modifications
3. **Evening**: Integration test results

### Escalation Path
1. Collision detected → Pause both teams
2. Coordinator resolves conflict
3. Clear communication to affected teams
4. Resume work with updated guidelines

## Integration Schedule

### Week 1-2: Independent Development
- Teams work in isolation
- Coordinator monitors for potential conflicts
- No integration required

### Week 3: Team A + B Integration
- Test distributed training on fixed infrastructure
- Coordinate API changes
- Performance benchmarking

### Week 4: Full Integration
- Team C integrates RLHF with SFT
- End-to-end testing
- Production readiness assessment

## Current Status (Real-time)

### Team A Status: 🟡 Awaiting Start
- Working on: Preparing to remove MPI dependencies
- Last activity: Not yet detected
- Blocked by: None
- Files monitoring: distributed_comm.py, distributed_api.py, grpc_*.py

### Team B Status: 🟢 Active  
- Working on: Distributed optimizer implementation (AdamW)
- Last activity: 2025-07-31 14:45
- Recent files:
  - src/mlx_distributed/training/optimizers/distributed_adamw.py ✅
  - src/mlx_distributed/training/advanced_features.py ✅
  - src/mlx_distributed/training/recovery.py ✅
- Blocked by: None

### Team C Status: 🟢 Active - Standalone Package
- Working on: mlx_rlhf standalone package
- Last activity: 2025-07-31 15:00
- Package structure:
  - mlx_rlhf/src/mlx_rlhf/algorithms/ (PPO, DPO, Reward Model)
  - mlx_rlhf/src/mlx_rlhf/utils/distributed.py (simplified interfaces)
  - mlx_rlhf/examples/ (training scripts)
  - mlx_rlhf/tests/ (comprehensive test suite)
- Blocked by: None
- Integration: Will connect via APIs only

## Conflict Log
- **2025-07-31 14:45**: No conflicts detected. Teams B and C working in separate directories.

## Coordination Notes
- Team B has established optimizer infrastructure in `training/optimizers/`
- Team C has pivoted to standalone `mlx_rlhf` package - NO COLLISION RISK! 🎉
- pyproject.toml modified (likely by Team A or linter)
- grpc_server.py had syntax fix (line 256)
- All teams following excellent separation practices

## Architecture Update
- mlx_distributed: Handles distributed inference and core training
- mlx_rlhf: Separate package for RLHF/DPO algorithms
- Integration: Via clean APIs and shared MLX primitives only

---
Last Updated: 2025-07-31 14:47
Coordinator: MLX Distributed Coordination Agent
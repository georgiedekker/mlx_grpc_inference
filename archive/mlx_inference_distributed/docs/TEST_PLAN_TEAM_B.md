# MLX Distributed Training Test Plan - Team B

## Overview
Comprehensive test plan for the distributed training infrastructure, covering completed components and upcoming features.

## 1. Core Infrastructure Tests (✅ Completed Components)

### 1.1 Distributed Trainer Tests (`test_distributed_trainer.py`)
```python
# Test the core distributed training loop
def test_trainer_initialization():
    """Verify trainer initializes with correct world size and rank"""
    
def test_forward_backward_pass():
    """Test complete training iteration with gradient computation"""
    
def test_gradient_accumulation():
    """Verify gradient accumulation across steps"""
    
def test_mixed_precision_training():
    """Test FP16/BF16 training stability"""
    
def test_memory_efficiency():
    """Monitor memory usage during training"""
```

### 1.2 gRPC Gradient Synchronization (`test_grpc_sync.py`)
```python
def test_gradient_allreduce():
    """Test gradient synchronization across devices via gRPC"""
    # Launch on 2+ devices
    # Compute gradients on same data
    # Verify gradients match after sync
    
def test_async_gradient_sync():
    """Test asynchronous gradient updates"""
    
def test_gradient_compression():
    """Test gradient compression for bandwidth efficiency"""
    
def test_sync_failure_handling():
    """Test behavior when a device fails during sync"""
```

### 1.3 Distributed Optimizer Tests (`test_distributed_optimizers.py`)
```python
def test_distributed_adamw():
    """Test DistributedAdamW optimizer"""
    # Verify momentum synchronization
    # Test weight decay application
    # Check convergence behavior
    
def test_optimizer_state_sharding():
    """Test optimizer state distribution across devices"""
    
def test_gradient_clipping():
    """Test distributed gradient clipping"""
    
def test_learning_rate_scheduling():
    """Test LR schedulers in distributed setting"""
```

### 1.4 Data Parallel Tests (`test_data_parallel.py`)
```python
def test_data_sharding():
    """Verify data is properly sharded across ranks"""
    # No duplicate samples
    # All samples covered
    # Balanced distribution
    
def test_data_loader_efficiency():
    """Test data loading performance"""
    # Prefetching works
    # No bottlenecks
    # Memory efficient
    
def test_dynamic_batching():
    """Test dynamic batch size adjustment"""
```

### 1.5 Pipeline Parallel Tests (`test_pipeline_parallel.py`)
```python
def test_model_partitioning():
    """Test model split across devices"""
    
def test_pipeline_bubble_optimization():
    """Verify pipeline bubble is minimized"""
    
def test_activation_checkpointing():
    """Test memory-efficient activation storage"""
    
def test_micro_batch_scheduling():
    """Test micro-batch pipeline scheduling"""
```

### 1.6 Checkpointing Tests (`test_checkpointing.py`)
```python
def test_save_checkpoint():
    """Test checkpoint saving"""
    # Model state saved correctly
    # Optimizer state preserved
    # Training metadata included
    
def test_resume_training():
    """Test resuming from checkpoint"""
    # Exact state restoration
    # Training continues smoothly
    # Metrics preserved
    
def test_distributed_checkpoint():
    """Test saving from multiple devices"""
    # Proper synchronization
    # No file conflicts
    # Efficient storage
    
def test_checkpoint_versioning():
    """Test checkpoint compatibility"""
```

### 1.7 CLI Tests (`test_cli.py`)
```python
def test_mlx_train_command():
    """Test mlx-train CLI functionality"""
    # Command parsing
    # Config loading
    # Error handling
    
def test_cli_resume():
    """Test --resume flag functionality"""
    
def test_cli_distributed_launch():
    """Test multi-device launch via CLI"""
```

## 2. Upcoming Feature Tests (🔄 To Be Implemented)

### 2.1 SFT Trainer Tests (`test_sft_trainer.py`)
```python
def test_sft_initialization():
    """Test SFT trainer setup"""
    
def test_instruction_following_loss():
    """Test loss computation for instruction tuning"""
    
def test_sft_data_preprocessing():
    """Test chat template application"""
    
def test_sft_evaluation():
    """Test instruction following metrics"""
```

### 2.2 Dataset Loader Tests (`test_datasets.py`)
```python
def test_alpaca_format_loading():
    """Test loading Alpaca-style datasets"""
    # Sample data:
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    }
    
def test_sharegpt_format_loading():
    """Test loading ShareGPT conversations"""
    # Sample data:
    {
        "conversations": [
            {"from": "human", "value": "Hello!"},
            {"from": "gpt", "value": "Hi there!"}
        ]
    }
    
def test_dataset_tokenization():
    """Test proper tokenization with padding"""
    
def test_dataset_streaming():
    """Test streaming large datasets"""
```

### 2.3 Evaluation Metrics Tests (`test_metrics.py`)
```python
def test_perplexity_calculation():
    """Test perplexity computation"""
    
def test_bleu_score():
    """Test BLEU metric implementation"""
    
def test_rouge_score():
    """Test ROUGE metric implementation"""
    
def test_distributed_metric_aggregation():
    """Test metric collection across devices"""
```

### 2.4 WandB Integration Tests (`test_wandb.py`)
```python
def test_wandb_initialization():
    """Test WandB project setup"""
    
def test_metric_logging():
    """Test logging metrics to WandB"""
    
def test_distributed_wandb():
    """Test WandB with multiple devices"""
    
def test_artifact_tracking():
    """Test model artifact versioning"""
```

## 3. Integration Tests

### 3.1 End-to-End Training (`test_e2e_training.py`)
```python
def test_full_training_pipeline():
    """Test complete training workflow"""
    # 1. Load model
    # 2. Prepare dataset
    # 3. Run training for N steps
    # 4. Save checkpoint
    # 5. Evaluate results
    
def test_distributed_training_convergence():
    """Verify distributed training matches single-device"""
    # Train same model both ways
    # Compare final loss
    # Verify similar convergence
```

### 3.2 Performance Benchmarks (`test_performance.py`)
```python
def test_scaling_efficiency():
    """Test multi-device scaling"""
    devices = [1, 2, 4, 8]
    for n in devices:
        time = measure_training_time(n_devices=n)
        # Expect 85-90% efficiency
        
def test_memory_scaling():
    """Test memory usage with scale"""
    
def test_communication_overhead():
    """Measure sync overhead"""
```

### 3.3 Stress Tests (`test_stress.py`)
```python
def test_long_training_run():
    """24-hour stability test"""
    
def test_large_model_training():
    """Test with 7B+ parameter models"""
    
def test_recovery_scenarios():
    """Test various failure modes"""
    # Device failure
    # Network partition
    # OOM recovery
```

## 4. Configuration Tests

### 4.1 YAML Config Tests (`test_config.py`)
```python
def test_config_loading():
    """Test YAML configuration parsing"""
    
def test_config_validation():
    """Test config parameter validation"""
    
def test_config_defaults():
    """Test default value handling"""
    
def test_distributed_config_sync():
    """Test config broadcast to all devices"""
```

## 5. Error Handling Tests

### 5.1 Failure Mode Tests (`test_failures.py`)
```python
def test_device_failure_during_training():
    """Test graceful handling of device failure"""
    
def test_checkpoint_corruption_recovery():
    """Test recovery from corrupted checkpoint"""
    
def test_data_loading_errors():
    """Test handling of data errors"""
    
def test_gradient_explosion_handling():
    """Test numerical stability"""
```

## 6. UV Environment Tests

### 6.1 Package Management (`test_uv_env.py`)
```python
def test_uv_commands():
    """Test UV CLI commands work correctly"""
    commands = [
        "uv run mlx-train --help",
        "uv run mlx-serve --help",
        "uv run mlx-distributed --help"
    ]
    
def test_dependency_resolution():
    """Verify all dependencies installed"""
    
def test_python_version():
    """Verify Python 3.13.5 is used"""
```

## Test Execution Plan

### Phase 1: Unit Tests (Completed Components)
- Run all tests for completed features
- Achieve 90%+ coverage on existing code
- Fix any bugs found

### Phase 2: Integration Tests
- Test multi-device scenarios
- Verify gRPC communication
- Benchmark performance

### Phase 3: New Feature Tests (As Implemented)
- Test each new feature as it's developed
- Maintain test-driven development
- Keep coverage above 90%

### Phase 4: System Tests
- End-to-end training runs
- Performance optimization
- Stress testing

## Test Infrastructure

### Directory Structure
```
mlx_distributed/tests/
├── unit/
│   ├── test_distributed_trainer.py
│   ├── test_optimizers.py
│   └── test_checkpointing.py
├── integration/
│   ├── test_grpc_sync.py
│   ├── test_data_parallel.py
│   └── test_pipeline_parallel.py
├── e2e/
│   ├── test_full_training.py
│   └── test_convergence.py
├── benchmarks/
│   ├── test_scaling.py
│   └── test_performance.py
└── fixtures/
    ├── sample_data/
    └── test_configs/
```

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run tests
        run: |
          uv sync
          uv run pytest tests/ -v --cov
```

## Success Metrics

1. **Code Coverage**: >90% for all modules
2. **Performance**: Linear scaling up to 4 devices
3. **Reliability**: Zero failures in 24-hour test
4. **Compatibility**: Works with all supported models
5. **Usability**: All examples run without modification

This comprehensive test plan ensures the distributed training system is production-ready and maintains high quality standards.
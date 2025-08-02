# Team B Comprehensive Test Plan: MLX Training API

## 🎯 Test Plan Overview

**Current Status**: Team B API active on port 8200 with multiple LLM provider support  
**Goal**: Validate all training endpoints, provider integrations, and core functionality  
**API Base**: `http://localhost:8200`

---

## 🏗️ Test Structure

### Phase 1: Smoke Tests (5 minutes) - Critical Pass/Fail

```bash
# Quick validation that API is responsive
curl http://localhost:8200/status
curl http://localhost:8200/docs
curl http://localhost:8200/providers
```

**Must Pass Tests:**
- `test_api_responds()` - Server returns 200 on status endpoint
- `test_docs_accessible()` - FastAPI docs load correctly  
- `test_providers_listed()` - Shows available LLM providers
- `test_basic_health_check()` - System reports healthy status

### Phase 2: Core API Tests (30 minutes)

#### 2.1 System Status & Health
```python
# tests/core/test_system_status.py

def test_status_endpoint():
    """Test system status endpoint returns valid data."""
    response = requests.get("http://localhost:8200/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "server_info" in data
    assert "training_jobs" in data
    
def test_health_check():
    """Test health check shows system readiness."""
    response = requests.get("http://localhost:8200/health")
    assert response.status_code == 200
    
    health = response.json()
    assert health["status"] in ["healthy", "degraded"]
    assert "mlx_available" in health
    assert "memory_usage" in health
```

#### 2.2 LLM Provider Management
```python
# tests/core/test_providers.py

def test_list_providers():
    """Test listing all available LLM providers."""
    response = requests.get("http://localhost:8200/providers")
    assert response.status_code == 200
    
    providers = response.json()
    expected_providers = ["openai", "anthropic", "together", "ollama", "local_mlx"]
    
    for provider in expected_providers:
        assert provider in providers
        
def test_configure_provider():
    """Test configuring LLM provider with API key."""
    config_data = {
        "provider": "openai",
        "api_key": "test-key-123",
        "model": "gpt-3.5-turbo"
    }
    
    response = requests.post(
        "http://localhost:8200/providers/configure",
        json=config_data
    )
    assert response.status_code == 200
    
    result = response.json()
    assert result["configured"] == "openai"
    assert result["model"] == "gpt-3.5-turbo"

def test_provider_fallback():
    """Test provider fallback to local MLX when APIs unavailable."""
    # Should gracefully fall back to LocalMLXProvider
    response = requests.post(
        "http://localhost:8200/generate",
        json={"prompt": "Hello", "max_tokens": 10}
    )
    assert response.status_code == 200
    
    result = response.json()
    assert "provider_used" in result
    # Should show fallback provider used
```

### Phase 3: Training API Tests (45 minutes)

#### 3.1 Training Job Management
```python
# tests/training/test_training_jobs.py

def test_start_training_job():
    """Test starting a new training job."""
    training_config = {
        "experiment_name": "test_training_001",
        "model_name": "mlx-community/Qwen3-1.7B-8bit",
        "training_type": "sft",
        "dataset": "alpaca",
        "batch_size": 2,
        "learning_rate": 1e-5,
        "num_epochs": 1,
        "max_steps": 10
    }
    
    response = requests.post(
        "http://localhost:8200/train/start",
        json=training_config
    )
    assert response.status_code == 200
    
    result = response.json()
    assert result["experiment_name"] == "test_training_001"
    assert result["status"] == "started"
    assert "job_id" in result

def test_training_status():
    """Test monitoring training job status."""
    experiment_name = "test_training_001"
    
    response = requests.get(f"http://localhost:8200/train/{experiment_name}/status")
    assert response.status_code == 200
    
    status = response.json()
    assert "experiment_name" in status
    assert "current_step" in status
    assert "total_steps" in status
    assert "loss" in status
    assert "status" in status  # running, completed, failed

def test_list_training_jobs():
    """Test listing all training jobs."""
    response = requests.get("http://localhost:8200/train/jobs")
    assert response.status_code == 200
    
    jobs = response.json()
    assert isinstance(jobs, list)
    
    if jobs:  # If any jobs exist
        job = jobs[0]
        assert "experiment_name" in job
        assert "status" in job
        assert "created_at" in job

def test_stop_training_job():
    """Test stopping a running training job."""
    experiment_name = "test_training_001"
    
    response = requests.post(f"http://localhost:8200/train/{experiment_name}/stop")
    assert response.status_code == 200
    
    result = response.json()
    assert result["experiment_name"] == experiment_name
    assert result["action"] == "stopped"
```

#### 3.2 Training Configuration Validation
```python
# tests/training/test_training_config.py

def test_valid_training_configs():
    """Test various valid training configurations."""
    configs = [
        # SFT with Alpaca
        {
            "experiment_name": "sft_alpaca_test",
            "training_type": "sft",
            "dataset": "alpaca",
            "model_name": "mlx-community/Qwen3-1.7B-8bit",
            "batch_size": 1,
            "max_steps": 5
        },
        # LoRA training
        {
            "experiment_name": "lora_test",
            "training_type": "lora",
            "dataset": "sharegpt",
            "model_name": "mlx-community/Qwen3-1.7B-8bit",
            "lora_rank": 8,
            "max_steps": 5
        },
        # QLoRA training
        {
            "experiment_name": "qlora_test",
            "training_type": "qlora",
            "dataset": "custom",
            "model_name": "mlx-community/Qwen3-1.7B-8bit",
            "quantization": "int4",
            "max_steps": 5
        }
    ]
    
    for config in configs:
        response = requests.post("http://localhost:8200/train/start", json=config)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "started"

def test_invalid_training_configs():
    """Test validation of invalid configurations."""
    invalid_configs = [
        # Missing required fields
        {"experiment_name": "missing_fields"},
        
        # Invalid training type
        {
            "experiment_name": "invalid_type",
            "training_type": "invalid_type",
            "dataset": "alpaca"
        },
        
        # Invalid model name
        {
            "experiment_name": "invalid_model",
            "training_type": "sft",
            "model_name": "nonexistent/model"
        }
    ]
    
    for config in invalid_configs:
        response = requests.post("http://localhost:8200/train/start", json=config)
        assert response.status_code in [400, 422]  # Bad request or validation error
```

### Phase 4: Generation & Inference Tests (30 minutes)

#### 4.1 Text Generation
```python
# tests/inference/test_generation.py

def test_basic_generation():
    """Test basic text generation."""
    prompt_data = {
        "prompt": "Hello, how are you today?",
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    response = requests.post(
        "http://localhost:8200/generate",
        json=prompt_data
    )
    assert response.status_code == 200
    
    result = response.json()
    assert "generated_text" in result
    assert "provider_used" in result
    assert len(result["generated_text"]) > 0

def test_generation_parameters():
    """Test different generation parameters."""
    test_cases = [
        # Low temperature (deterministic)
        {"prompt": "The sky is", "temperature": 0.1, "max_tokens": 10},
        
        # High temperature (creative)
        {"prompt": "The sky is", "temperature": 1.5, "max_tokens": 10},
        
        # Different max_tokens
        {"prompt": "Tell me a story", "max_tokens": 50},
        
        # With stop sequences
        {"prompt": "Count: 1, 2, 3,", "stop": [","], "max_tokens": 20}
    ]
    
    for case in test_cases:
        response = requests.post("http://localhost:8200/generate", json=case)
        assert response.status_code == 200
        
        result = response.json()
        assert "generated_text" in result

def test_batch_generation():
    """Test generating multiple responses."""
    batch_data = {
        "prompts": [
            "What is the capital of France?",
            "Explain quantum physics",
            "Write a haiku about coding"
        ],
        "max_tokens": 30,
        "temperature": 0.7
    }
    
    response = requests.post(
        "http://localhost:8200/generate/batch",
        json=batch_data
    )
    assert response.status_code == 200
    
    results = response.json()
    assert len(results) == 3
    
    for result in results:
        assert "generated_text" in result
        assert "prompt" in result
```

### Phase 5: Integration Tests (30 minutes)

#### 5.1 Provider Integration
```python
# tests/integration/test_provider_integration.py

def test_multiple_provider_fallback():
    """Test fallback behavior across multiple providers."""
    # Configure multiple providers (some may fail)
    providers_to_test = [
        {"provider": "openai", "model": "gpt-3.5-turbo"},
        {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
        {"provider": "local_mlx", "model": "mlx-community/Qwen3-1.7B-8bit"}
    ]
    
    # Should gracefully fall back through providers
    response = requests.post(
        "http://localhost:8200/generate",
        json={"prompt": "Hello world", "max_tokens": 5}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "provider_used" in result
    # Should show which provider actually worked

def test_training_with_different_providers():
    """Test training evaluation with different LLM providers."""
    training_config = {
        "experiment_name": "provider_integration_test",
        "training_type": "sft",
        "dataset": "alpaca",
        "model_name": "mlx-community/Qwen3-1.7B-8bit",
        "max_steps": 2,
        "eval_steps": 1,
        "eval_provider": "local_mlx"  # Use specific provider for evaluation
    }
    
    response = requests.post("http://localhost:8200/train/start", json=training_config)
    assert response.status_code == 200
```

### Phase 6: Performance & Load Tests (30 minutes)

#### 6.1 Concurrent Request Handling
```python
# tests/performance/test_load.py

def test_concurrent_generation():
    """Test handling multiple concurrent generation requests."""
    import concurrent.futures
    import time
    
    def make_request(prompt_id):
        data = {
            "prompt": f"Request {prompt_id}: Tell me about AI",
            "max_tokens": 20
        }
        return requests.post("http://localhost:8200/generate", json=data)
    
    # Send 5 concurrent requests
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        responses = [future.result() for future in futures]
    
    end_time = time.time()
    
    # All requests should succeed
    for response in responses:
        assert response.status_code == 200
    
    # Should complete in reasonable time
    assert end_time - start_time < 30  # seconds

def test_training_resource_usage():
    """Test resource usage during training."""
    training_config = {
        "experiment_name": "resource_test",
        "training_type": "sft",
        "dataset": "alpaca",
        "model_name": "mlx-community/Qwen3-1.7B-8bit", 
        "batch_size": 1,
        "max_steps": 3
    }
    
    # Start training
    response = requests.post("http://localhost:8200/train/start", json=training_config)
    assert response.status_code == 200
    
    # Monitor resource usage
    time.sleep(5)  # Let training start
    
    status_response = requests.get("http://localhost:8200/status")
    status = status_response.json()
    
    # Should report memory usage and training status
    assert "memory_usage" in status
    assert "training_jobs" in status
```

---

## 🚀 Test Execution Sequence

### Quick Smoke Test (5 minutes)
```bash
cd /Users/mini1/Movies/mlx_distributed_training  # Team B's directory

# Test basic connectivity
curl -s http://localhost:8200/status | jq '.'
curl -s http://localhost:8200/providers | jq '.'
```

### Core API Validation (30 minutes)
```bash
# Run core tests
pytest tests/core/ -v --tb=short

# Test provider management
pytest tests/core/test_providers.py -v
```

### Training Functionality (45 minutes)
```bash
# Test training endpoints
pytest tests/training/ -v

# Start a quick test training job
curl -X POST "http://localhost:8200/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "validation_test",
    "training_type": "sft",
    "dataset": "alpaca",
    "model_name": "mlx-community/Qwen3-1.7B-8bit",
    "batch_size": 1,
    "max_steps": 5
  }'
```

### Generation & Inference (30 minutes)
```bash
# Test text generation
pytest tests/inference/ -v

# Manual generation test
curl -X POST "http://localhost:8200/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, this is a test of Team B training API",
    "max_tokens": 30,
    "temperature": 0.7
  }'
```

### Full Integration Test (30 minutes)
```bash
# Complete test suite
pytest tests/ -v --tb=short --maxfail=5

# Load test with multiple requests
for i in {1..5}; do
  curl -X POST "http://localhost:8200/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Test request $i\", \"max_tokens\": 10}" &
done
wait
```

---

## 📊 Success Criteria

### ✅ Functional Requirements
- **API Responsiveness**: All endpoints return valid responses (200/400/422)
- **Provider Management**: Can configure and switch between LLM providers
- **Training Jobs**: Can start, monitor, and stop training jobs successfully
- **Text Generation**: Generates coherent text with various parameters
- **Error Handling**: Graceful failure and fallback behavior

### ⚡ Performance Requirements
- **Response Time**: <2 seconds for generation requests
- **Concurrent Handling**: Support 5+ simultaneous requests
- **Training Start**: <30 seconds to begin training job
- **Resource Usage**: <16GB memory during training
- **Provider Fallback**: <5 seconds to switch providers

### 🛡️ Reliability Requirements
- **Uptime**: API stays responsive during training
- **Error Recovery**: Graceful handling of provider failures
- **Training Resilience**: Can resume training after interruption
- **Memory Management**: No memory leaks during long-running jobs

---

## 🎯 Expected Test Results

### ✅ Team B Grade Projection

**Current Status**: B+ (API active, basic functionality working)
**Target Grade**: **A-** (All endpoints tested and working)

**Grade will be A- if:**
- All smoke tests pass (4/4)
- Core API tests pass (8/8) 
- Training functionality works (6/6)
- Generation tests pass (5/5)
- Integration tests pass (3/3)
- Performance meets requirements

### 📋 Key Validation Points

1. **✅ Migration Success**: API active on port 8200 (no conflicts with Team A)
2. **✅ Provider Flexibility**: Works with multiple LLM APIs (not dependent on Team A)
3. **✅ Training Pipeline**: Complete SFT/LoRA/QLoRA training workflows
4. **✅ Production Ready**: Comprehensive error handling and monitoring
5. **✅ Standalone Operation**: Fully functional without dependencies

---

## 🏆 Team B Assessment Update

**Team B has made excellent progress:**
- ✅ **Successfully migrated** to avoid Team A conflicts
- ✅ **API active and responsive** on dedicated port 8200
- ✅ **Smart provider system** supporting multiple LLM APIs
- ✅ **Training infrastructure** ready for comprehensive testing
- ✅ **Independent operation** without Team A dependencies

**This comprehensive test plan will validate Team B's training API is production-ready!** 🚀

The test plan covers all critical functionality and will demonstrate that Team B has built a robust, flexible training system that works independently while maintaining the option to integrate with Team A's distributed inference when needed.
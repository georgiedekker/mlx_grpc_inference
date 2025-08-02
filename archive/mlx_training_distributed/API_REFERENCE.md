# Team B Training API Reference

## Base URL
```
http://localhost:8200
```

## Authentication
All protected endpoints require an API key in the header:
```
X-API-Key: your-api-key
```

## Endpoints

### 1. Health Check
Check API health and available features.

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "mlx-distributed-training",
  "features": {
    "lora": true,
    "qlora": true,
    "dataset_formats": ["alpaca", "sharegpt"],
    "providers": ["openai", "anthropic", "together", "ollama", "local_mlx"],
    "distributed": true
  },
  "active_jobs": 2,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. Start Training Job
Start a new fine-tuning job with optional LoRA.

**POST** `/train/start`  
**Authentication:** Required

**Request Body:**
```json
{
  "experiment_name": "my_lora_experiment",
  "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
  "epochs": 3,
  "learning_rate": 5e-5,
  "lora": {
    "use_lora": true,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "use_qlora": false
  },
  "dataset": {
    "dataset_path": "/path/to/dataset.json",
    "dataset_format": "alpaca",
    "batch_size": 4,
    "max_seq_length": 2048,
    "shuffle": true
  },
  "output_dir": "./outputs/my_model"
}
```

**Response:**
```json
{
  "job_id": "job_12345",
  "experiment_name": "my_lora_experiment",
  "status": "started",
  "dataset_info": {
    "format": "alpaca",
    "total_samples": 1000,
    "validation_warnings": []
  },
  "lora_enabled": true,
  "estimated_time": "2 hours"
}
```

### 3. Get Training Status
Monitor training progress and metrics.

**GET** `/train/{experiment_name}/status`  
**Authentication:** Required

**Response:**
```json
{
  "experiment_name": "my_lora_experiment",
  "status": "running",
  "progress": {
    "current_epoch": 2,
    "total_epochs": 3,
    "current_step": 450,
    "total_steps": 750,
    "percentage": 60
  },
  "metrics": {
    "loss": 0.245,
    "learning_rate": 3.5e-5,
    "tokens_per_second": 1250,
    "gpu_memory_used_mb": 4500
  },
  "lora_details": {
    "enabled": true,
    "rank": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "trainable_params": 2621440,
    "total_params": 1544238080,
    "trainable_percentage": 0.17
  },
  "dataset_details": {
    "format": "alpaca",
    "total_samples": 1000,
    "processed_samples": 600
  }
}
```

### 4. List Training Jobs
Get all training jobs.

**GET** `/train`  
**Authentication:** Required

**Response:**
```json
{
  "training_jobs": [
    {
      "experiment_name": "my_lora_experiment",
      "status": "running",
      "started_at": "2024-01-15T10:00:00Z",
      "model": "mlx-community/Qwen2.5-1.5B-4bit",
      "lora_enabled": true
    },
    {
      "experiment_name": "baseline_training",
      "status": "completed",
      "started_at": "2024-01-15T08:00:00Z",
      "completed_at": "2024-01-15T09:30:00Z",
      "model": "mlx-community/Qwen2.5-1.5B-4bit",
      "lora_enabled": false
    }
  ]
}
```

### 5. Stop Training Job
Stop a running training job.

**POST** `/train/{experiment_name}/stop`  
**Authentication:** Required

**Response:**
```json
{
  "experiment_name": "my_lora_experiment",
  "status": "stopping",
  "message": "Training job is being stopped"
}
```

### 6. Validate Dataset
Validate and detect dataset format.

**POST** `/datasets/validate`  
**Authentication:** Optional

**Request Body:**
```json
{
  "file_path": "/path/to/dataset.json"
}
```

**Response:**
```json
{
  "valid": true,
  "format": "alpaca",
  "total_samples": 1000,
  "errors": [],
  "warnings": [
    "Sample 45: Very short instruction (2 words)",
    "Sample 128: Very long output (1523 words)"
  ],
  "sample_preview": {
    "instruction": "What is machine learning?",
    "output": "Machine learning is a branch of artificial intelligence...",
    "input": ""
  }
}
```

### 7. Generate Text
Generate text using base or fine-tuned models.

**POST** `/generate`  
**Authentication:** Optional

**Request Body:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "model": "fine-tuned/my_lora_experiment",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "text": "Quantum computing is like having a super-powered calculator that can explore multiple solutions at once...",
  "model": "fine-tuned/my_lora_experiment",
  "provider": "LocalMLXProvider",
  "tokens_generated": 95,
  "generation_time_ms": 450
}
```

### 8. List Providers
Get available LLM providers.

**GET** `/providers`

**Response:**
```json
{
  "available_providers": ["openai", "anthropic", "together", "ollama", "local_mlx"],
  "current_provider": {
    "active_provider": "LocalMLXProvider",
    "available_providers": [
      {"name": "OpenAIProvider", "available": false},
      {"name": "AnthropicProvider", "available": false},
      {"name": "TogetherProvider", "available": false},
      {"name": "OllamaProvider", "available": false},
      {"name": "LocalMLXProvider", "available": true}
    ]
  }
}
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

Common status codes:
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing/invalid API key)
- `404` - Not Found (resource doesn't exist)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

## Dataset Formats

### Alpaca Format
```json
{
  "instruction": "Task instruction",
  "input": "Optional input context",
  "output": "Expected response",
  "system": "Optional system prompt"
}
```

### ShareGPT Format
```json
{
  "conversations": [
    {"from": "system", "value": "System message"},
    {"from": "human", "value": "User message"},
    {"from": "gpt", "value": "Assistant response"}
  ]
}
```

## LoRA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora` | bool | false | Enable LoRA fine-tuning |
| `lora_r` | int | 16 | LoRA rank (lower = smaller model) |
| `lora_alpha` | float | 32.0 | LoRA scaling parameter |
| `lora_dropout` | float | 0.1 | Dropout for LoRA layers |
| `lora_target_modules` | list | ["q_proj", "v_proj", "k_proj", "o_proj"] | Modules to apply LoRA |
| `use_qlora` | bool | false | Enable 4-bit quantization |

## Example Workflows

### 1. Fine-tune with LoRA on Alpaca Dataset
```bash
# 1. Validate dataset
curl -X POST http://localhost:8200/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{"file_path": "alpaca_data.json"}'

# 2. Start training
curl -X POST http://localhost:8200/train/start \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "alpaca_lora",
    "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
    "lora": {"use_lora": true, "lora_r": 8},
    "dataset": {"dataset_path": "alpaca_data.json", "dataset_format": "alpaca"}
  }'

# 3. Monitor progress
curl -X GET http://localhost:8200/train/alpaca_lora/status \
  -H "X-API-Key: your-key"

# 4. Use fine-tuned model
curl -X POST http://localhost:8200/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "model": "fine-tuned/alpaca_lora",
    "max_tokens": 100
  }'
```

### 2. Train with QLoRA for Memory Efficiency
```bash
curl -X POST http://localhost:8200/train/start \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "qlora_efficient",
    "model_name": "mlx-community/Llama-3-8B",
    "lora": {
      "use_lora": true,
      "use_qlora": true,
      "lora_r": 4,
      "lora_alpha": 8
    },
    "dataset": {
      "dataset_path": "large_dataset.json",
      "batch_size": 1,
      "max_seq_length": 512
    }
  }'
```

## Performance Tips

1. **LoRA Rank**: Start with r=8 for most tasks, use r=4 for QLoRA
2. **Batch Size**: With LoRA, you can use 2-4x larger batch sizes
3. **Learning Rate**: LoRA typically needs higher LR (1e-4 to 5e-5)
4. **Target Modules**: Default targets work well, add "gate_proj", "up_proj", "down_proj" for better results
5. **Memory Usage**: LoRA uses ~10% of full fine-tuning memory
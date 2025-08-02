"""
Team B API Modifications - Complete Integration Guide
This file contains all the code modifications needed to add LoRA/QLoRA support
and dataset format handling to Team B's existing training API.

INSTRUCTIONS:
1. Copy the import statements to the top of your API file
2. Add the new request models to your schemas/models
3. Update your existing endpoints with the enhanced versions
4. Add the new endpoints for health and dataset validation
5. Update your training logic with LoRA support
"""

# ==============================================================================
# SECTION 1: IMPORT STATEMENTS TO ADD
# ==============================================================================
"""
Add these imports to the top of your main API file (e.g., app.py or main.py):
"""

# Standard library imports
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator

# MLX and training imports
import mlx.core as mx
import mlx.nn as nn

# Add your project to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import LoRA components
from src.lora.lora import (
    apply_lora_to_model,
    LoRAConfig as LoRAConfigImpl,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    print_lora_info,
    freeze_base_model,
    get_lora_parameters
)

# Import dataset components
from src.datasets.base_dataset import BaseDataset, DatasetConfig as BaseDatasetConfig
from src.datasets.alpaca_dataset import AlpacaDataset
from src.datasets.sharegpt_dataset import ShareGPTDataset
from src.datasets.dataset_utils import (
    detect_dataset_format,
    validate_dataset,
    DatasetValidationResult
)

# Import integration helpers
from src.integration.lora_integration import (
    create_lora_enabled_trainer,
    save_lora_checkpoint,
    load_lora_for_inference
)
from src.integration.dataset_integration import (
    validate_dataset,
    create_dataset_loader_config,
    DatasetValidationResult
)

# ==============================================================================
# SECTION 2: ENHANCED REQUEST MODELS
# ==============================================================================
"""
Add or update these Pydantic models in your schemas:
"""

class LoRAConfig(BaseModel):
    """LoRA configuration for fine-tuning."""
    use_lora: bool = Field(default=False, description="Enable LoRA fine-tuning")
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank (lower = more compression)")
    lora_alpha: float = Field(default=32.0, gt=0, description="LoRA scaling parameter")
    lora_dropout: float = Field(default=0.1, ge=0, le=1, description="LoRA dropout rate")
    lora_target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Modules to apply LoRA to"
    )
    use_qlora: bool = Field(default=False, description="Enable QLoRA (4-bit quantization)")
    
    @validator('lora_r')
    def validate_rank(cls, v):
        # Common LoRA ranks
        if v not in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            print(f"Warning: Unusual LoRA rank {v}. Common values: 4, 8, 16, 32")
        return v

class DatasetConfig(BaseModel):
    """Dataset configuration for training."""
    dataset_path: str = Field(..., description="Path to dataset file")
    dataset_format: Optional[str] = Field(
        default=None,
        description="Dataset format: 'alpaca' or 'sharegpt' (auto-detect if None)"
    )
    max_seq_length: int = Field(default=2048, ge=128, le=8192)
    batch_size: int = Field(default=8, ge=1, le=128)
    shuffle: bool = Field(default=True)
    validation_split: float = Field(default=0.1, ge=0, le=0.5)
    
    @validator('dataset_path')
    def validate_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v

class TrainingJobRequest(BaseModel):
    """Enhanced training job request with LoRA and dataset support."""
    experiment_name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., description="Model name or path")
    
    # Training hyperparameters
    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=5e-5, gt=0, le=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    save_steps: int = Field(default=500, ge=1)
    
    # New: LoRA configuration
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    # New: Dataset configuration
    dataset: DatasetConfig
    
    # Output configuration
    output_dir: str = Field(default="./outputs")
    save_total_limit: int = Field(default=3, ge=1)
    
    # Additional options
    resume_from_checkpoint: Optional[str] = None
    push_to_hub: bool = Field(default=False)
    hub_model_id: Optional[str] = None

class DatasetValidationRequest(BaseModel):
    """Request model for dataset validation."""
    file_path: str = Field(..., description="Path to dataset file")
    format_hint: Optional[str] = Field(None, description="Expected format (alpaca/sharegpt)")
    sample_size: int = Field(default=5, ge=1, le=100, description="Number of samples to preview")

# ==============================================================================
# SECTION 3: NEW ENDPOINTS TO ADD
# ==============================================================================
"""
Add these new endpoints to your FastAPI app:
"""

@app.get("/health")
async def health_check():
    """
    Health check endpoint with feature information.
    
    Returns current API status and available features.
    """
    # Get active training jobs count (implement based on your job tracking)
    active_jobs = len(get_active_training_jobs())  # You need to implement this
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "mlx-distributed-training",
        "features": {
            "lora": True,
            "qlora": True,
            "dataset_formats": ["alpaca", "sharegpt"],
            "providers": ["openai", "anthropic", "together", "ollama", "local_mlx"],
            "distributed": True,
            "model_quantization": ["4bit", "8bit"],
            "checkpoint_formats": ["mlx", "lora_only", "merged"]
        },
        "system": {
            "active_jobs": active_jobs,
            "gpu_available": mx.metal.is_available(),
            "memory_gb": mx.metal.get_active_memory() / 1e9 if mx.metal.is_available() else 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/datasets/validate")
async def validate_dataset_endpoint(request: DatasetValidationRequest):
    """
    Validate a dataset file and detect its format.
    
    Returns validation results including format detection, error checking,
    and sample preview.
    """
    try:
        # Validate dataset using integration helper
        validation_result = validate_dataset(request.file_path)
        
        # Load sample data for preview
        with open(request.file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data[:request.sample_size]
            else:
                samples = None
        
        return {
            "valid": validation_result.is_valid,
            "format": validation_result.format_type,
            "total_samples": validation_result.total_samples,
            "errors": validation_result.errors[:10],  # Limit to 10 errors
            "warnings": validation_result.warnings[:10],  # Limit to 10 warnings
            "statistics": {
                "avg_length": validation_result.statistics.get("avg_length", 0),
                "max_length": validation_result.statistics.get("max_length", 0),
                "min_length": validation_result.statistics.get("min_length", 0),
                "empty_samples": validation_result.statistics.get("empty_samples", 0)
            },
            "sample_preview": samples,
            "recommendations": validation_result.recommendations
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {request.file_path}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

# ==============================================================================
# SECTION 4: ENHANCED TRAINING ENDPOINT
# ==============================================================================
"""
Update your existing training endpoint with LoRA support:
"""

# Global job tracking (implement based on your needs)
training_jobs = {}
job_counter = 0

def get_active_training_jobs():
    """Get list of active training jobs."""
    return [job for job in training_jobs.values() if job["status"] in ["pending", "running"]]

async def prepare_model_for_training(model, config: TrainingJobRequest):
    """Prepare model with LoRA if enabled."""
    training_info = {
        "lora_enabled": False,
        "trainable_params": 0,
        "total_params": 0,
        "compression_ratio": 1.0
    }
    
    if config.lora.use_lora:
        # Create LoRA configuration
        lora_config = LoRAConfigImpl(
            r=config.lora.lora_r,
            alpha=config.lora.lora_alpha,
            dropout=config.lora.lora_dropout,
            target_modules=config.lora.lora_target_modules,
            use_qlora=config.lora.use_qlora
        )
        
        # Apply LoRA to model
        model = apply_lora_to_model(model, lora_config)
        
        # Freeze base model parameters
        freeze_base_model(model)
        
        # Get parameter counts
        lora_params = get_lora_parameters(model)
        total_params = sum(p.size for p in model.parameters())
        trainable_params = sum(p.size for p in lora_params.values())
        
        # Print LoRA information
        print_lora_info(model, lora_config)
        
        training_info = {
            "lora_enabled": True,
            "lora_config": lora_config.dict(),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "compression_ratio": total_params / trainable_params if trainable_params > 0 else 1.0,
            "memory_savings_pct": (1 - trainable_params / total_params) * 100
        }
    
    return model, training_info

async def load_training_dataset(config: DatasetConfig, tokenizer):
    """Load dataset with automatic format detection."""
    # Validate dataset first
    validation_result = validate_dataset(config.dataset_path)
    
    if not validation_result.is_valid:
        raise ValueError(f"Invalid dataset: {'; '.join(validation_result.errors[:5])}")
    
    # Detect format if not specified
    dataset_format = config.dataset_format
    if not dataset_format:
        dataset_format = validation_result.format_type
    
    # Create base dataset configuration
    base_config = BaseDatasetConfig(
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        validation_split=config.validation_split
    )
    
    # Load appropriate dataset
    if dataset_format == "alpaca":
        dataset = AlpacaDataset(
            data_source=config.dataset_path,
            config=base_config,
            tokenizer=tokenizer
        )
    elif dataset_format == "sharegpt":
        dataset = ShareGPTDataset(
            data_source=config.dataset_path,
            config=base_config,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    
    dataset_info = {
        "format": dataset_format,
        "total_samples": len(dataset),
        "train_samples": int(len(dataset) * (1 - config.validation_split)),
        "val_samples": int(len(dataset) * config.validation_split),
        "batch_size": config.batch_size,
        "max_seq_length": config.max_seq_length,
        "validation_warnings": validation_result.warnings[:5]
    }
    
    return dataset, dataset_info

@app.post("/v1/fine-tuning/jobs", response_model=Dict[str, Any])
async def create_fine_tuning_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(None)
):
    """
    Create a new fine-tuning job with LoRA and dataset support.
    
    This endpoint starts an asynchronous training job with optional LoRA
    parameter-efficient fine-tuning and support for multiple dataset formats.
    """
    # Validate API key (implement your auth logic)
    if not x_api_key or x_api_key != os.getenv("MLX_TRAINING_API_KEY", "test-api-key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    global job_counter
    job_counter += 1
    job_id = f"ftjob-{job_counter:06d}"
    
    try:
        # Load tokenizer (implement based on your model loading)
        tokenizer = load_tokenizer(request.model_name)  # You need to implement this
        
        # Load and validate dataset
        dataset, dataset_info = await load_training_dataset(request.dataset, tokenizer)
        
        # Load model (implement based on your model loading)
        model = load_model(request.model_name)  # You need to implement this
        
        # Apply LoRA if enabled
        model, lora_info = await prepare_model_for_training(model, request)
        
        # Calculate training estimates
        total_steps = (dataset_info["train_samples"] // request.dataset.batch_size) * request.epochs
        estimated_time = estimate_training_time(total_steps, lora_info["lora_enabled"])
        
        # Create job record
        job_info = {
            "id": job_id,
            "experiment_name": request.experiment_name,
            "model": request.model_name,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": request.dict(),
            "dataset_info": dataset_info,
            "lora_info": lora_info,
            "progress": {
                "current_epoch": 0,
                "total_epochs": request.epochs,
                "current_step": 0,
                "total_steps": total_steps,
                "percentage": 0
            },
            "metrics": {
                "loss": None,
                "learning_rate": request.learning_rate,
                "gpu_memory_used": 0
            },
            "estimated_completion_time": estimated_time
        }
        
        training_jobs[job_id] = job_info
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            model=model,
            dataset=dataset,
            config=request,
            job_info=job_info
        )
        
        # Return job creation response
        return {
            "id": job_id,
            "object": "fine-tuning.job",
            "model": request.model_name,
            "created_at": job_info["created_at"],
            "status": "pending",
            "experiment_name": request.experiment_name,
            "hyperparameters": {
                "n_epochs": request.epochs,
                "batch_size": request.dataset.batch_size,
                "learning_rate": request.learning_rate,
                "warmup_steps": request.warmup_steps
            },
            "lora_enabled": lora_info["lora_enabled"],
            "lora_details": lora_info if lora_info["lora_enabled"] else None,
            "dataset_info": dataset_info,
            "estimated_completion_time": estimated_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Job creation failed: {str(e)}")

@app.get("/v1/fine-tuning/jobs/{job_id}")
async def get_fine_tuning_job(job_id: str, x_api_key: str = Header(None)):
    """Get detailed status of a fine-tuning job."""
    # Validate API key
    if not x_api_key or x_api_key != os.getenv("MLX_TRAINING_API_KEY", "test-api-key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_info = training_jobs[job_id]
    
    return {
        "id": job_id,
        "object": "fine-tuning.job",
        "model": job_info["model"],
        "created_at": job_info["created_at"],
        "finished_at": job_info.get("finished_at"),
        "status": job_info["status"],
        "experiment_name": job_info["experiment_name"],
        "progress": job_info["progress"],
        "metrics": job_info["metrics"],
        "lora_details": job_info["lora_info"] if job_info["lora_info"]["lora_enabled"] else None,
        "dataset_details": job_info["dataset_info"],
        "error": job_info.get("error"),
        "result_files": job_info.get("result_files", [])
    }

# ==============================================================================
# SECTION 5: HELPER FUNCTIONS
# ==============================================================================
"""
Add these helper functions to support the enhanced endpoints:
"""

def estimate_training_time(total_steps: int, lora_enabled: bool) -> str:
    """Estimate training completion time."""
    # Simple estimation (adjust based on your hardware)
    if lora_enabled:
        seconds_per_step = 0.5  # LoRA is faster
    else:
        seconds_per_step = 2.0  # Full fine-tuning
    
    total_seconds = total_steps * seconds_per_step
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

async def run_training_job(job_id: str, model, dataset, config: TrainingJobRequest, job_info: dict):
    """Run the actual training job (implement based on your training logic)."""
    try:
        # Update status
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Create trainer with LoRA support
        trainer = create_lora_enabled_trainer(
            model=model,
            dataset=dataset,
            config=config,
            job_id=job_id
        )
        
        # Training loop (pseudo-code - implement based on your needs)
        for epoch in range(config.epochs):
            for step, batch in enumerate(dataset):
                # Update progress
                current_step = epoch * len(dataset) + step
                progress = {
                    "current_epoch": epoch + 1,
                    "total_epochs": config.epochs,
                    "current_step": current_step,
                    "total_steps": job_info["progress"]["total_steps"],
                    "percentage": (current_step / job_info["progress"]["total_steps"]) * 100
                }
                training_jobs[job_id]["progress"] = progress
                
                # Training step (implement your training logic)
                loss = trainer.training_step(batch)
                
                # Update metrics
                training_jobs[job_id]["metrics"]["loss"] = float(loss)
                
                # Save checkpoint if needed
                if step % config.save_steps == 0:
                    if config.lora.use_lora:
                        checkpoint_path = save_lora_checkpoint(
                            model=model,
                            job_id=job_id,
                            step=current_step,
                            output_dir=config.output_dir
                        )
                    else:
                        checkpoint_path = save_full_checkpoint(
                            model=model,
                            job_id=job_id,
                            step=current_step,
                            output_dir=config.output_dir
                        )
        
        # Training completed
        training_jobs[job_id]["status"] = "succeeded"
        training_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        training_jobs[job_id]["result_files"] = [
            f"{config.output_dir}/{job_id}/final_model",
            f"{config.output_dir}/{job_id}/training_logs.json"
        ]
        
    except Exception as e:
        # Handle training failure
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

# ==============================================================================
# SECTION 6: EXAMPLE USAGE WITH CURL COMMANDS
# ==============================================================================
"""
Example curl commands to test the integration:

1. Health Check:
curl -X GET http://localhost:8200/health

2. Validate Dataset:
curl -X POST http://localhost:8200/v1/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/alpaca_dataset.json",
    "sample_size": 3
  }'

3. Create LoRA Fine-tuning Job:
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key" \
  -d '{
    "experiment_name": "my_lora_experiment",
    "model_name": "mlx-community/Qwen2.5-1.5B-4bit",
    "epochs": 3,
    "learning_rate": 5e-5,
    "lora": {
      "use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16.0,
      "lora_dropout": 0.05,
      "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "dataset": {
      "dataset_path": "/path/to/alpaca_dataset.json",
      "dataset_format": "alpaca",
      "batch_size": 8,
      "max_seq_length": 2048
    },
    "output_dir": "./outputs/lora_experiment"
  }'

4. Check Job Status:
curl -X GET http://localhost:8200/v1/fine-tuning/jobs/ftjob-000001 \
  -H "X-API-Key: test-api-key"

5. Create QLoRA Job (4-bit):
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key" \
  -d '{
    "experiment_name": "qlora_memory_efficient",
    "model_name": "mlx-community/Qwen2.5-7B-4bit",
    "epochs": 1,
    "lora": {
      "use_lora": true,
      "use_qlora": true,
      "lora_r": 4,
      "lora_alpha": 8.0
    },
    "dataset": {
      "dataset_path": "/path/to/dataset.json",
      "batch_size": 4,
      "max_seq_length": 1024
    }
  }'
"""

# ==============================================================================
# SECTION 7: NOTES FOR TEAM B
# ==============================================================================
"""
IMPLEMENTATION NOTES:

1. Replace placeholder functions with your implementations:
   - get_active_training_jobs()
   - load_tokenizer()
   - load_model()
   - save_full_checkpoint()

2. Adjust the training loop in run_training_job() to match your training logic

3. Configure authentication based on your security requirements

4. Update error handling to match your application's patterns

5. Consider adding:
   - Rate limiting for API endpoints
   - Request validation middleware
   - Logging and monitoring
   - Webhook notifications for job completion

6. Performance tips:
   - Use async/await for I/O operations
   - Implement caching for dataset validation
   - Use background tasks for long-running operations
   - Consider using a job queue (e.g., Celery) for production

7. Testing:
   - Run the integration tests provided
   - Test with various dataset sizes
   - Verify memory usage with LoRA vs full fine-tuning
   - Check checkpoint sizes

Remember: LoRA typically reduces memory usage by 70-90% and speeds up training by 2-4x!
"""
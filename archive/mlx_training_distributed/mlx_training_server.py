#!/usr/bin/env python3
"""
MLX Training Server - Independent training framework
Runs on port 8500
"""

from fastapi import FastAPI, HTTPException, Body, Header
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
from datetime import datetime, timezone
import os
import json

app = FastAPI(
    title="MLX Training Framework",
    description="Advanced training framework with distributed optimizers and SFT",
    version="1.0.0"
)

# Storage for training jobs
training_jobs = {}
job_counter = 0

# ==============================================================================
# Request/Response Models
# ==============================================================================

class OptimizerConfig(BaseModel):
    type: str = Field(default="adamw", description="Optimizer type: adamw, sgd, lion")
    learning_rate: float = Field(default=5e-5)
    weight_decay: float = Field(default=0.01)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)

class SFTConfig(BaseModel):
    instruction_format: str = Field(default="alpaca", description="Format: alpaca, sharegpt")
    response_only_loss: bool = Field(default=True)
    max_seq_length: int = Field(default=2048)
    use_lora: bool = Field(default=False)
    lora_r: int = Field(default=16)
    lora_alpha: float = Field(default=32.0)

class TrainingRequest(BaseModel):
    experiment_name: str
    model_name: str = Field(default="mlx-community/Qwen2.5-1.5B-4bit")
    dataset_path: str
    training_type: str = Field(default="sft", description="Training type: sft, pretrain, custom")
    epochs: int = Field(default=3)
    batch_size: int = Field(default=4)
    gradient_accumulation_steps: int = Field(default=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    sft_config: Optional[SFTConfig] = Field(default_factory=SFTConfig)
    distributed: bool = Field(default=False)
    num_devices: int = Field(default=1)

# ==============================================================================
# Health & Info Endpoints
# ==============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MLX Training Framework",
        "version": "1.0.0",
        "port": 8500,
        "features": {
            "optimizers": ["adamw", "sgd", "lion"],
            "training_types": ["sft", "pretrain", "custom"],
            "distributed": True,
            "lora": True,
            "gradient_accumulation": True,
            "mixed_precision": True,
            "recovery": True
        },
        "capabilities": {
            "max_batch_size": 128,
            "max_seq_length": 8192,
            "distributed_backends": ["allreduce", "ring_allreduce", "parameter_server"],
            "dataset_formats": ["alpaca", "sharegpt", "jsonl", "parquet"]
        },
        "system": {
            "active_jobs": len([j for j in training_jobs.values() if j["status"] == "running"]),
            "total_jobs": len(training_jobs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MLX Training Framework API",
        "description": "Advanced training framework with distributed optimizers and SFT",
        "endpoints": {
            "health": "/health",
            "training": {
                "start": "POST /v1/training/start",
                "list": "GET /v1/training/jobs",
                "status": "GET /v1/training/jobs/{job_id}",
                "stop": "POST /v1/training/jobs/{job_id}/stop"
            },
            "optimizers": {
                "list": "GET /v1/optimizers",
                "config": "GET /v1/optimizers/{optimizer_type}"
            },
            "datasets": {
                "validate": "POST /v1/datasets/validate",
                "formats": "GET /v1/datasets/formats"
            }
        }
    }

# ==============================================================================
# Training Endpoints
# ==============================================================================

@app.post("/v1/training/start")
async def start_training(
    request: TrainingRequest,
    x_api_key: str = Header(None)
):
    """Start a new training job."""
    # API key validation
    api_key = os.getenv("MLX_TRAINING_API_KEY", "mlx-training-key")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    global job_counter
    job_counter += 1
    job_id = f"train-{job_counter:06d}"
    
    # Calculate training parameters
    effective_batch_size = request.batch_size * request.gradient_accumulation_steps
    if request.distributed:
        effective_batch_size *= request.num_devices
    
    # Create job info
    job_info = {
        "id": job_id,
        "experiment_name": request.experiment_name,
        "model_name": request.model_name,
        "dataset_path": request.dataset_path,
        "training_type": request.training_type,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "learning_rate": request.optimizer.learning_rate,
            "optimizer": request.optimizer.type
        },
        "distributed_config": {
            "enabled": request.distributed,
            "num_devices": request.num_devices,
            "backend": "allreduce" if request.distributed else None
        },
        "progress": {
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "current_step": 0,
            "total_steps": 0
        }
    }
    
    # Add SFT-specific info if applicable
    if request.training_type == "sft" and request.sft_config:
        job_info["sft_config"] = {
            "instruction_format": request.sft_config.instruction_format,
            "response_only_loss": request.sft_config.response_only_loss,
            "lora_enabled": request.sft_config.use_lora,
            "lora_rank": request.sft_config.lora_r if request.sft_config.use_lora else None
        }
    
    training_jobs[job_id] = job_info
    return job_info

@app.get("/v1/training/jobs")
async def list_training_jobs(
    x_api_key: str = Header(None),
    status: Optional[str] = None
):
    """List all training jobs."""
    # API key validation
    api_key = os.getenv("MLX_TRAINING_API_KEY", "mlx-training-key")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    jobs_list = list(training_jobs.values())
    
    # Filter by status if provided
    if status:
        jobs_list = [j for j in jobs_list if j["status"] == status]
    
    # Sort by creation time (newest first)
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "jobs": jobs_list,
        "total": len(jobs_list),
        "filtered_by": {"status": status} if status else None
    }

@app.get("/v1/training/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    x_api_key: str = Header(None)
):
    """Get detailed status of a training job."""
    # API key validation
    api_key = os.getenv("MLX_TRAINING_API_KEY", "mlx-training-key")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_info = training_jobs[job_id].copy()
    
    # Simulate progress for demo
    if job_info["status"] == "pending":
        job_info["status"] = "running"
        job_info["started_at"] = datetime.now(timezone.utc).isoformat()
        job_info["progress"]["current_epoch"] = 1
        job_info["progress"]["current_step"] = 100
        job_info["progress"]["total_steps"] = 300
        
        # Add metrics
        job_info["metrics"] = {
            "loss": 2.34,
            "learning_rate": job_info["hyperparameters"]["learning_rate"],
            "grad_norm": 1.25,
            "throughput": {
                "samples_per_second": 45.6,
                "tokens_per_second": 2340
            }
        }
        
        # Update stored job
        training_jobs[job_id] = job_info
    
    return job_info

@app.post("/v1/training/jobs/{job_id}/stop")
async def stop_training_job(
    job_id: str,
    x_api_key: str = Header(None)
):
    """Stop a running training job."""
    # API key validation
    api_key = os.getenv("MLX_TRAINING_API_KEY", "mlx-training-key")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_info = training_jobs[job_id]
    
    if job_info["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Job {job_id} is not running (status: {job_info['status']})"
        )
    
    # Update job status
    job_info["status"] = "stopped"
    job_info["stopped_at"] = datetime.now(timezone.utc).isoformat()
    
    return {"message": f"Job {job_id} stopped successfully", "job": job_info}

# ==============================================================================
# Optimizer Endpoints
# ==============================================================================

@app.get("/v1/optimizers")
async def list_optimizers():
    """List available optimizers and their capabilities."""
    return {
        "optimizers": {
            "adamw": {
                "name": "AdamW",
                "description": "Adam with weight decay",
                "distributed": True,
                "parameters": ["learning_rate", "beta1", "beta2", "epsilon", "weight_decay"]
            },
            "sgd": {
                "name": "SGD",
                "description": "Stochastic Gradient Descent",
                "distributed": True,
                "parameters": ["learning_rate", "momentum", "weight_decay", "nesterov"]
            },
            "lion": {
                "name": "Lion",
                "description": "Evolved Sign Momentum optimizer",
                "distributed": True,
                "parameters": ["learning_rate", "beta1", "beta2", "weight_decay"]
            }
        }
    }

@app.get("/v1/optimizers/{optimizer_type}")
async def get_optimizer_config(optimizer_type: str):
    """Get detailed configuration for a specific optimizer."""
    configs = {
        "adamw": {
            "type": "adamw",
            "default_config": {
                "learning_rate": 5e-5,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01
            },
            "recommended_for": ["language_models", "vision_models"],
            "gradient_sync_modes": ["allreduce", "ring_allreduce"]
        },
        "sgd": {
            "type": "sgd",
            "default_config": {
                "learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "nesterov": True
            },
            "recommended_for": ["computer_vision", "simple_models"],
            "gradient_sync_modes": ["allreduce"]
        },
        "lion": {
            "type": "lion",
            "default_config": {
                "learning_rate": 1e-4,
                "beta1": 0.9,
                "beta2": 0.99,
                "weight_decay": 0.0
            },
            "recommended_for": ["large_language_models", "efficient_training"],
            "gradient_sync_modes": ["allreduce", "parameter_server"]
        }
    }
    
    if optimizer_type not in configs:
        raise HTTPException(status_code=404, detail=f"Optimizer {optimizer_type} not found")
    
    return configs[optimizer_type]

# ==============================================================================
# Dataset Endpoints
# ==============================================================================

@app.post("/v1/datasets/validate")
async def validate_dataset(request: Dict[str, Any] = Body(...)):
    """Validate a dataset for training."""
    file_path = request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    
    # Simulated validation
    return {
        "valid": True,
        "format": "alpaca",
        "total_samples": 1000,
        "sample_stats": {
            "avg_instruction_length": 45,
            "avg_response_length": 128,
            "max_sequence_length": 512
        },
        "warnings": [],
        "recommendations": [
            "Consider using gradient accumulation for better batch efficiency",
            "Dataset is suitable for LoRA fine-tuning"
        ]
    }

@app.get("/v1/datasets/formats")
async def list_dataset_formats():
    """List supported dataset formats."""
    return {
        "formats": {
            "alpaca": {
                "name": "Alpaca",
                "description": "Instruction-following format with instruction/input/output",
                "required_fields": ["instruction", "output"],
                "optional_fields": ["input", "system"]
            },
            "sharegpt": {
                "name": "ShareGPT",
                "description": "Multi-turn conversation format",
                "required_fields": ["conversations"],
                "conversation_fields": ["from", "value"]
            },
            "jsonl": {
                "name": "JSONL",
                "description": "Generic JSON Lines format",
                "required_fields": ["text"],
                "optional_fields": ["metadata"]
            },
            "parquet": {
                "name": "Parquet",
                "description": "Columnar format for large datasets",
                "required_fields": ["text"],
                "optional_fields": ["metadata", "source"]
            }
        }
    }

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Set API key if not already set
    if not os.getenv("MLX_TRAINING_API_KEY"):
        os.environ["MLX_TRAINING_API_KEY"] = "mlx-training-key"
        print("🔑 Set API key to 'mlx-training-key'")
    
    print("🚀 Starting MLX Training Server...")
    print("📡 Available endpoints:")
    print("  GET  /health")
    print("  POST /v1/training/start")
    print("  GET  /v1/training/jobs")
    print("  GET  /v1/training/jobs/{job_id}")
    print("  POST /v1/training/jobs/{job_id}/stop")
    print("  GET  /v1/optimizers")
    print("  GET  /v1/optimizers/{optimizer_type}")
    print("  POST /v1/datasets/validate")
    print("  GET  /v1/datasets/formats")
    print("\n🎯 MLX Training Framework ready on port 8500!")
    
    uvicorn.run(app, host="0.0.0.0", port=8500)
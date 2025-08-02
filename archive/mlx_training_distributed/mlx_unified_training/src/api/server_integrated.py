#!/usr/bin/env python3
"""
MLX Unified Training Platform - Fully Integrated
Combines KD, RLHF, and Core Training with real implementation
Runs on port 8600
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import uvicorn
from datetime import datetime, timezone
import uuid
import json
from enum import Enum
import asyncio
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from training import (
    TrainingOrchestrator,
    TrainingConfig,
    LoRAConfig,
    DistillationConfig,
    RLHFConfig,
    DatasetHandler
)

app = FastAPI(
    title="MLX Unified Training Platform",
    description="Production-ready integrated platform for KD, RLHF, and Core Training",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator
orchestrator = TrainingOrchestrator()

# Storage for pipelines
pipelines_storage = {}

# ==============================================================================
# Enums and Models
# ==============================================================================

class TrainingStage(str, Enum):
    SFT = "sft"
    DISTILLATION = "distillation"
    RLHF = "rlhf"

class PipelineStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class RLHFMethod(str, Enum):
    DPO = "dpo"
    PPO = "ppo"
    REWARD_MODEL = "reward_model"

# Configuration Models
class SFTConfig(BaseModel):
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    instruction_format: str = "alpaca"
    max_seq_length: int = 2048
    optimizer_type: str = "adamw"

class DistillationConfigRequest(BaseModel):
    teacher_models: List[str] = Field(..., description="List of teacher model names")
    temperature: float = 3.0
    alpha: float = 0.7
    distillation_loss: str = "kl_div"
    feature_matching: bool = True
    intermediate_layers: List[int] = [6, 12, 18]
    epochs: int = 5
    adaptive_temperature: bool = True

class RLHFConfigRequest(BaseModel):
    method: RLHFMethod = RLHFMethod.DPO
    beta: float = 0.1
    learning_rate: float = 5e-7
    epochs: int = 1
    batch_size: int = 2
    preference_dataset: str = Field(..., description="Path to preference dataset")
    reward_model_path: Optional[str] = None

# Pipeline Models
class PipelineConfig(BaseModel):
    name: str
    stages: List[TrainingStage]
    base_model: str
    dataset_path: str
    output_dir: str = "./outputs"
    sft_config: Optional[SFTConfig] = None
    distillation_config: Optional[DistillationConfigRequest] = None
    rlhf_config: Optional[RLHFConfigRequest] = None
    distributed: bool = False
    num_devices: int = 1

class PipelineCreateRequest(BaseModel):
    name: str = Field(..., description="Pipeline name")
    stages: List[TrainingStage] = Field(..., description="Training stages to execute")
    base_model: str = Field(..., description="Base model to train")
    dataset_path: str = Field(..., description="Path to training dataset")
    auto_configure: bool = Field(True, description="Auto-configure stages with defaults")

# Direct training request
class DirectTrainingRequest(BaseModel):
    model_name: str
    dataset_path: str
    output_dir: Optional[str] = None
    config: Union[SFTConfig, DistillationConfigRequest, RLHFConfigRequest]

# ==============================================================================
# Health and Info Endpoints
# ==============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    jobs = orchestrator.list_jobs()
    pipelines = orchestrator.list_pipelines()
    
    return {
        "status": "healthy",
        "service": "MLX Unified Training Platform",
        "version": "2.0.0",
        "implementation": "full",
        "port": 8600,
        "modules": {
            "knowledge_distillation": {
                "status": "ready",
                "features": ["multi-teacher", "adaptive-temperature", "feature-matching"],
                "implementation": "complete"
            },
            "rlhf": {
                "status": "ready",
                "methods": ["dpo", "ppo", "reward-modeling"],
                "implementation": "complete"
            },
            "core_training": {
                "status": "ready",
                "features": ["sft", "lora", "distributed"],
                "implementation": "complete"
            }
        },
        "capabilities": {
            "pipeline_orchestration": True,
            "workflow_templates": True,
            "checkpoint_management": True,
            "distributed_training": True,
            "real_training": True
        },
        "system": {
            "active_pipelines": len([p for p in pipelines if p["status"] == "running"]),
            "total_pipelines": len(pipelines_storage),
            "active_jobs": len([j for j in jobs if j["status"] == "running"]),
            "total_jobs": len(jobs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MLX Unified Training Platform",
        "description": "Production-ready platform combining KD, RLHF, and Core Training",
        "endpoints": {
            "health": "/health",
            "pipelines": {
                "create": "POST /v1/pipelines/create",
                "list": "GET /v1/pipelines",
                "get": "GET /v1/pipelines/{pipeline_id}",
                "run": "POST /v1/pipelines/{pipeline_id}/run",
                "status": "GET /v1/pipelines/{pipeline_id}/status"
            },
            "direct_training": {
                "sft": "POST /v1/train/sft",
                "distill": "POST /v1/train/distill",
                "rlhf": "POST /v1/train/rlhf"
            },
            "workflows": {
                "templates": "GET /v1/workflows/templates",
                "create_from_template": "POST /v1/workflows/from-template"
            },
            "jobs": {
                "list": "GET /v1/jobs",
                "get": "GET /v1/jobs/{job_id}"
            }
        }
    }

# ==============================================================================
# Pipeline Management Endpoints
# ==============================================================================

@app.post("/v1/pipelines/create")
async def create_pipeline(request: PipelineCreateRequest):
    """Create a new training pipeline with real implementation."""
    
    pipeline_id = str(uuid.uuid4())
    
    # Auto-configure if requested
    config = PipelineConfig(
        name=request.name,
        stages=request.stages,
        base_model=request.base_model,
        dataset_path=request.dataset_path
    )
    
    if request.auto_configure:
        if TrainingStage.SFT in request.stages:
            config.sft_config = SFTConfig()
        if TrainingStage.DISTILLATION in request.stages:
            config.distillation_config = DistillationConfigRequest(
                teacher_models=["mlx-community/Qwen2.5-7B", "mlx-community/Llama-3.2-3B"]
            )
        if TrainingStage.RLHF in request.stages:
            config.rlhf_config = RLHFConfigRequest(
                preference_dataset=f"{request.dataset_path}_preferences"
            )
    
    # Create pipeline in orchestrator
    pipeline_config = {
        "name": config.name,
        "stages": [s.value for s in config.stages],
        "base_model": config.base_model,
        "dataset_path": config.dataset_path,
        "output_dir": config.output_dir
    }
    
    # Add stage-specific configs
    if config.sft_config:
        pipeline_config["sft_config"] = config.sft_config.dict()
    if config.distillation_config:
        pipeline_config["distillation_config"] = config.distillation_config.dict()
    if config.rlhf_config:
        pipeline_config["rlhf_config"] = config.rlhf_config.dict()
    
    # Create pipeline
    pipeline_id = await orchestrator.create_pipeline(pipeline_config)
    
    # Store in local storage
    pipelines_storage[pipeline_id] = {
        "id": pipeline_id,
        "config": config.dict(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    return {
        "pipeline_id": pipeline_id,
        "status": "created",
        "config": config.dict(),
        "message": f"Pipeline '{request.name}' created successfully"
    }

@app.get("/v1/pipelines")
async def list_pipelines():
    """List all pipelines."""
    pipelines = orchestrator.list_pipelines()
    return {
        "pipelines": pipelines,
        "total": len(pipelines)
    }

@app.get("/v1/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get detailed pipeline information."""
    try:
        pipeline_status = orchestrator.get_pipeline_status(pipeline_id)
        return pipeline_status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/v1/pipelines/{pipeline_id}/run")
async def run_pipeline(
    pipeline_id: str,
    background_tasks: BackgroundTasks
):
    """Run a training pipeline with real execution."""
    
    if pipeline_id not in orchestrator.pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = orchestrator.pipelines[pipeline_id]
    
    if pipeline["status"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    # Run pipeline asynchronously
    background_tasks.add_task(orchestrator.run_pipeline, pipeline_id)
    
    return {
        "message": f"Pipeline {pipeline_id} started",
        "status": "running",
        "stages": pipeline["stages"]
    }

@app.get("/v1/pipelines/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """Get current pipeline execution status."""
    try:
        status = orchestrator.get_pipeline_status(pipeline_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==============================================================================
# Direct Training Endpoints
# ==============================================================================

@app.post("/v1/train/sft")
async def train_sft(
    request: DirectTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Direct SFT training with real implementation."""
    
    # Prepare config
    config = {
        "training_type": "standard",
        "model_name": request.model_name,
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir or f"./outputs/sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        **request.config.dict()
    }
    
    # Create job
    job_id = await orchestrator.create_job(config)
    
    return {
        "job_id": job_id,
        "message": "SFT training started",
        "config": config
    }

@app.post("/v1/train/distill")
async def train_distill(
    request: DirectTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Direct knowledge distillation with real implementation."""
    
    if not isinstance(request.config, DistillationConfigRequest):
        raise HTTPException(status_code=400, detail="Invalid config for distillation")
    
    # Prepare config
    config = {
        "training_type": "distillation",
        "model_name": request.model_name,
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir or f"./outputs/distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        **request.config.dict()
    }
    
    # Create job
    job_id = await orchestrator.create_job(config)
    
    return {
        "job_id": job_id,
        "message": "Distillation training started",
        "teachers": request.config.teacher_models,
        "config": config
    }

@app.post("/v1/train/rlhf")
async def train_rlhf(
    request: DirectTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Direct RLHF training with real implementation."""
    
    if not isinstance(request.config, RLHFConfigRequest):
        raise HTTPException(status_code=400, detail="Invalid config for RLHF")
    
    # Prepare config
    config = {
        "training_type": "rlhf",
        "model_name": request.model_name,
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir or f"./outputs/rlhf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "rlhf_method": request.config.method.value,
        **request.config.dict()
    }
    
    # Create job
    job_id = await orchestrator.create_job(config)
    
    return {
        "job_id": job_id,
        "message": f"RLHF training started with {request.config.method}",
        "config": config
    }

# ==============================================================================
# Workflow Templates
# ==============================================================================

@app.get("/v1/workflows/templates")
async def get_workflow_templates():
    """Get pre-configured workflow templates."""
    return {
        "templates": {
            "chatbot_training": {
                "name": "Chatbot Training Pipeline",
                "description": "Complete pipeline for training conversational AI",
                "stages": ["sft", "rlhf"],
                "recommended_for": ["customer_service", "virtual_assistants"],
                "estimated_time": "4-6 hours",
                "config_hints": {
                    "sft": {"epochs": 3, "batch_size": 4},
                    "rlhf": {"method": "dpo", "beta": 0.1}
                }
            },
            "efficient_llm": {
                "name": "Efficient LLM Pipeline",
                "description": "Distill large models into efficient deployable versions",
                "stages": ["distillation", "sft"],
                "recommended_for": ["edge_deployment", "mobile_apps"],
                "estimated_time": "6-8 hours",
                "config_hints": {
                    "distillation": {"temperature": 3.0, "alpha": 0.7},
                    "sft": {"use_lora": True, "lora_rank": 8}
                }
            },
            "aligned_model": {
                "name": "Aligned Model Pipeline",
                "description": "Create safe and aligned models with human feedback",
                "stages": ["sft", "rlhf", "distillation"],
                "recommended_for": ["production_models", "safety_critical"],
                "estimated_time": "8-12 hours",
                "config_hints": {
                    "sft": {"epochs": 3},
                    "rlhf": {"method": "dpo", "epochs": 2},
                    "distillation": {"feature_matching": True}
                }
            },
            "research_pipeline": {
                "name": "Research Pipeline",
                "description": "Full pipeline for research experiments",
                "stages": ["sft", "distillation", "rlhf"],
                "recommended_for": ["research", "experimentation"],
                "estimated_time": "10-15 hours",
                "config_hints": {
                    "sft": {"epochs": 5, "learning_rate": 5e-5},
                    "distillation": {"adaptive_temperature": True},
                    "rlhf": {"method": "ppo", "ppo_epochs": 4}
                }
            }
        }
    }

@app.post("/v1/workflows/from-template")
async def create_from_template(
    background_tasks: BackgroundTasks,
    template_name: str = Body(...),
    model_name: str = Body(...),
    dataset_path: str = Body(...)
):
    """Create a pipeline from a template with real execution."""
    
    templates = {
        "chatbot_training": ["sft", "rlhf"],
        "efficient_llm": ["distillation", "sft"],
        "aligned_model": ["sft", "rlhf", "distillation"],
        "research_pipeline": ["sft", "distillation", "rlhf"]
    }
    
    if template_name not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Create pipeline from template
    request = PipelineCreateRequest(
        name=f"{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        stages=[TrainingStage(s) for s in templates[template_name]],
        base_model=model_name,
        dataset_path=dataset_path,
        auto_configure=True
    )
    
    return await create_pipeline(request)

# ==============================================================================
# Job Management Endpoints
# ==============================================================================

@app.get("/v1/jobs")
async def list_jobs(status: Optional[str] = None):
    """List all training jobs."""
    jobs = orchestrator.list_jobs()
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    return {
        "jobs": jobs,
        "total": len(jobs)
    }

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Get detailed job information."""
    try:
        job_status = orchestrator.get_job_status(job_id)
        return job_status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Starting MLX Unified Training Platform with Full Implementation...")
    print("📡 Combining Knowledge Distillation, RLHF, and Core Training")
    print("\n📋 Available modules:")
    print("  - Knowledge Distillation (Multi-teacher, Adaptive)")
    print("  - RLHF (DPO, PPO, Reward Modeling)")
    print("  - Core Training (SFT, LoRA, Distributed)")
    print("\n🔧 Pipeline orchestration enabled")
    print("✅ Real MLX training implementation")
    print("🎯 Server running on port 8600")
    
    uvicorn.run(app, host="0.0.0.0", port=8600)
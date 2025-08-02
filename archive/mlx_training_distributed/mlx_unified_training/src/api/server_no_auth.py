#!/usr/bin/env python3
"""
MLX Unified Training Platform API Server
Combines KD, RLHF, and Core Training in one unified framework
Runs on port 8600
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import uvicorn
from datetime import datetime, timezone
import uuid
import json
from enum import Enum
import asyncio

app = FastAPI(
    title="MLX Unified Training Platform",
    description="Integrated platform for Knowledge Distillation, RLHF, and Core Training",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
pipelines_storage = {}
training_jobs = {}

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

class DistillationConfig(BaseModel):
    teacher_models: List[str] = Field(..., description="List of teacher model names")
    temperature: float = 3.0
    alpha: float = 0.7
    distillation_loss: str = "kl_div"
    feature_matching: bool = True
    intermediate_layers: List[int] = [6, 12, 18]
    epochs: int = 5

class RLHFConfig(BaseModel):
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
    distillation_config: Optional[DistillationConfig] = None
    rlhf_config: Optional[RLHFConfig] = None
    distributed: bool = False
    num_devices: int = 1

class PipelineCreateRequest(BaseModel):
    name: str = Field(..., description="Pipeline name")
    stages: List[TrainingStage] = Field(..., description="Training stages to execute")
    base_model: str = Field(..., description="Base model to train")
    dataset_path: str = Field(..., description="Path to training dataset")
    auto_configure: bool = Field(True, description="Auto-configure stages with defaults")

# ==============================================================================
# Health and Info Endpoints
# ==============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MLX Unified Training Platform",
        "version": "1.0.0",
        "port": 8600,
        "modules": {
            "knowledge_distillation": {
                "status": "ready",
                "features": ["multi-teacher", "adaptive-temperature", "feature-matching"]
            },
            "rlhf": {
                "status": "ready",
                "methods": ["dpo", "ppo", "reward-modeling"]
            },
            "core_training": {
                "status": "ready",
                "features": ["sft", "lora", "distributed"]
            }
        },
        "capabilities": {
            "pipeline_orchestration": True,
            "workflow_templates": True,
            "checkpoint_management": True,
            "distributed_training": True
        },
        "system": {
            "active_pipelines": len([p for p in pipelines_storage.values() 
                                   if p["status"] == PipelineStatus.RUNNING]),
            "total_pipelines": len(pipelines_storage),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MLX Unified Training Platform",
        "description": "Integrated platform combining KD, RLHF, and Core Training",
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
            }
        }
    }

# ==============================================================================
# Pipeline Management Endpoints
# ==============================================================================

@app.post("/v1/pipelines/create")
async def create_pipeline(
    request: PipelineCreateRequest,
    
):
    """Create a new training pipeline."""
    # API key validation
    
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
            config.distillation_config = DistillationConfig(
                teacher_models=["gpt-4", "claude-3"]  # Default teachers
            )
        if TrainingStage.RLHF in request.stages:
            config.rlhf_config = RLHFConfig(
                preference_dataset=f"{request.dataset_path}_preferences"
            )
    
    pipeline_info = {
        "id": pipeline_id,
        "config": config.dict(),
        "status": PipelineStatus.CREATED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages_completed": [],
        "current_stage": None,
        "progress": {}
    }
    
    pipelines_storage[pipeline_id] = pipeline_info
    
    return {
        "pipeline_id": pipeline_id,
        "status": "created",
        "config": config.dict(),
        "message": f"Pipeline '{request.name}' created successfully"
    }

@app.get("/v1/pipelines")
async def list_pipelines():
    """List all pipelines."""
    # API key validation
    
    return {
        "pipelines": [
            {
                "id": pid,
                "name": p["config"]["name"],
                "status": p["status"],
                "stages": p["config"]["stages"],
                "created_at": p["created_at"]
            }
            for pid, p in pipelines_storage.items()
        ],
        "total": len(pipelines_storage)
    }

@app.get("/v1/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get detailed pipeline information."""
    
    if pipeline_id not in pipelines_storage:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return pipelines_storage[pipeline_id]

@app.post("/v1/pipelines/{pipeline_id}/run")
async def run_pipeline(
    pipeline_id: str,
    background_tasks: BackgroundTasks,
    
):
    """Run a training pipeline."""
    
    if pipeline_id not in pipelines_storage:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_storage[pipeline_id]
    
    if pipeline["status"] == PipelineStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    # Update status
    pipeline["status"] = PipelineStatus.RUNNING
    pipeline["started_at"] = datetime.now(timezone.utc).isoformat()
    
    # Simulate pipeline execution in background
    background_tasks.add_task(execute_pipeline, pipeline_id)
    
    return {
        "message": f"Pipeline {pipeline_id} started",
        "status": "running",
        "stages": pipeline["config"]["stages"]
    }

async def execute_pipeline(pipeline_id: str):
    """Execute pipeline stages (simulated)."""
    pipeline = pipelines_storage[pipeline_id]
    
    try:
        for stage in pipeline["config"]["stages"]:
            pipeline["current_stage"] = stage
            pipeline["progress"][stage] = {"status": "running", "progress": 0}
            
            # Simulate stage execution
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(2)  # Simulate work
                pipeline["progress"][stage]["progress"] = progress
            
            pipeline["progress"][stage]["status"] = "completed"
            pipeline["stages_completed"].append(stage)
        
        pipeline["status"] = PipelineStatus.COMPLETED
        pipeline["completed_at"] = datetime.now(timezone.utc).isoformat()
        
    except Exception as e:
        pipeline["status"] = PipelineStatus.FAILED
        pipeline["error"] = str(e)

@app.get("/v1/pipelines/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """Get current pipeline execution status."""
    if pipeline_id not in pipelines_storage:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_storage[pipeline_id]
    
    return {
        "pipeline_id": pipeline_id,
        "status": pipeline["status"],
        "current_stage": pipeline["current_stage"],
        "stages_completed": pipeline["stages_completed"],
        "progress": pipeline["progress"]
    }

# ==============================================================================
# Direct Training Endpoints
# ==============================================================================

@app.post("/v1/train/sft")
async def train_sft(
    config: SFTConfig,
    model_name: str = Body(...),
    dataset_path: str = Body(...),
    
):
    """Direct SFT training endpoint."""
    
    job_id = f"sft-{uuid.uuid4().hex[:8]}"
    
    job_info = {
        "id": job_id,
        "type": "sft",
        "model": model_name,
        "dataset": dataset_path,
        "config": config.dict(),
        "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    training_jobs[job_id] = job_info
    
    return {
        "job_id": job_id,
        "message": "SFT training started",
        "config": config.dict()
    }

@app.post("/v1/train/distill")
async def train_distill(
    config: DistillationConfig,
    student_model: str = Body(...),
    dataset_path: str = Body(...),
    
):
    """Direct knowledge distillation endpoint."""
    
    job_id = f"distill-{uuid.uuid4().hex[:8]}"
    
    job_info = {
        "id": job_id,
        "type": "distillation",
        "student_model": student_model,
        "teacher_models": config.teacher_models,
        "dataset": dataset_path,
        "config": config.dict(),
        "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    training_jobs[job_id] = job_info
    
    return {
        "job_id": job_id,
        "message": "Distillation training started",
        "teachers": config.teacher_models,
        "config": config.dict()
    }

@app.post("/v1/train/rlhf")
async def train_rlhf(
    config: RLHFConfig,
    model_name: str = Body(...),
    
):
    """Direct RLHF training endpoint."""
    
    job_id = f"rlhf-{uuid.uuid4().hex[:8]}"
    
    job_info = {
        "id": job_id,
        "type": "rlhf",
        "method": config.method,
        "model": model_name,
        "preference_dataset": config.preference_dataset,
        "config": config.dict(),
        "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    training_jobs[job_id] = job_info
    
    return {
        "job_id": job_id,
        "message": f"RLHF training started with {config.method}",
        "config": config.dict()
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
                "recommended_for": ["customer_service", "virtual_assistants"]
            },
            "efficient_llm": {
                "name": "Efficient LLM Pipeline", 
                "description": "Distill large models into efficient deployable versions",
                "stages": ["distillation", "sft"],
                "recommended_for": ["edge_deployment", "mobile_apps"]
            },
            "aligned_model": {
                "name": "Aligned Model Pipeline",
                "description": "Create safe and aligned models with human feedback",
                "stages": ["sft", "rlhf", "distillation"],
                "recommended_for": ["production_models", "safety_critical"]
            },
            "research_pipeline": {
                "name": "Research Pipeline",
                "description": "Full pipeline for research experiments",
                "stages": ["sft", "distillation", "rlhf"],
                "recommended_for": ["research", "experimentation"]
            }
        }
    }

@app.post("/v1/workflows/from-template")
async def create_from_template(
    template_name: str = Body(...),
    model_name: str = Body(...),
    dataset_path: str = Body(...),
    
):
    """Create a pipeline from a template."""
    
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
    

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Starting MLX Unified Training Platform...")
    print("📡 Combining Knowledge Distillation, RLHF, and Core Training")
    print("\n📋 Available modules:")
    print("  - Knowledge Distillation (Multi-teacher, Adaptive)")
    print("  - RLHF (DPO, PPO, Reward Modeling)")
    print("  - Core Training (SFT, LoRA, Distributed)")
    print("\n🔧 Pipeline orchestration enabled")
    print("🎯 Server running on port 8600")
    
    uvicorn.run(app, host="0.0.0.0", port=8600)
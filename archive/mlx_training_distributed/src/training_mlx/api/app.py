"""
Training MLX API - FastAPI application for distributed training

Provides RESTful endpoints for training management on port 8200.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..utils.logging import setup_logging
from ..adapters.distributed_integration import get_integration_status

# Setup logging
logger = setup_logging()


# Pydantic models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime
    version: str = "0.1.0"
    service: str = "training-mlx"
    port: int = 8200
    features: Dict[str, Any]


class TrainingRequest(BaseModel):
    """Training job request"""
    experiment_name: str = Field(..., description="Unique experiment name")
    config_path: str = Field(..., description="Path to training configuration")
    model: str = Field(default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    dataset_path: str = Field(..., description="Path to training dataset")
    dataset_format: str = Field(default="alpaca", pattern="^(alpaca|sharegpt)$")
    use_lora: bool = Field(default=True, description="Use LoRA fine-tuning")
    distributed: bool = Field(default=False, description="Use distributed training")


class TrainingJob(BaseModel):
    """Training job status"""
    job_id: str
    experiment_name: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    progress: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class DatasetValidationRequest(BaseModel):
    """Dataset validation request"""
    dataset_path: str
    dataset_format: Optional[str] = None  # auto-detect if not provided
    sample_size: int = Field(default=100, ge=1, le=1000)


# Global state (in production, use a database)
training_jobs: Dict[str, TrainingJob] = {}
job_counter = 0


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Training MLX API",
        description="Distributed training API for MLX models with LoRA support",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirects to docs"""
        return {"message": "Training MLX API - Visit /docs for documentation"}
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        integration_status = get_integration_status()
        
        return HealthResponse(
            timestamp=datetime.utcnow(),
            features={
                "lora": True,
                "qlora": True,
                "dataset_formats": ["alpaca", "sharegpt"],
                "distributed": integration_status["distributed_available"],
                "providers": ["openai", "anthropic", "together", "ollama", "local_mlx"],
                "monitoring": ["wandb", "tensorboard"],
            }
        )
    
    @app.post("/v1/fine-tuning/jobs", response_model=TrainingJob)
    async def create_training_job(
        request: TrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """Create a new training job"""
        global job_counter
        
        # Generate job ID
        job_counter += 1
        job_id = f"ftjob-{job_counter:06d}"
        
        # Create job entry
        job = TrainingJob(
            job_id=job_id,
            experiment_name=request.experiment_name,
            status="pending",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            progress={"stage": "initializing", "epoch": 0, "step": 0}
        )
        
        training_jobs[job_id] = job
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            request=request
        )
        
        logger.info(f"Created training job: {job_id}")
        return job
    
    @app.get("/v1/fine-tuning/jobs", response_model=List[TrainingJob])
    async def list_training_jobs(
        status: Optional[str] = None,
        limit: int = 100
    ):
        """List all training jobs"""
        jobs = list(training_jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
    
    @app.get("/v1/fine-tuning/jobs/{job_id}", response_model=TrainingJob)
    async def get_training_job(job_id: str):
        """Get specific training job status"""
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return training_jobs[job_id]
    
    @app.post("/v1/fine-tuning/jobs/{job_id}/cancel")
    async def cancel_training_job(job_id: str):
        """Cancel a training job"""
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job = training_jobs[job_id]
        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job in {job.status} state"
            )
        
        job.status = "cancelled"
        job.updated_at = datetime.utcnow()
        
        logger.info(f"Cancelled job: {job_id}")
        return {"message": f"Job {job_id} cancelled"}
    
    @app.post("/v1/datasets/validate")
    async def validate_dataset(request: DatasetValidationRequest):
        """Validate a dataset file"""
        try:
            # Import here to avoid circular imports
            from ..training.datasets import detect_dataset_format, validate_dataset
            
            # Detect format if not provided
            if not request.dataset_format:
                format_info = detect_dataset_format(request.dataset_path)
                dataset_format = format_info["format"]
            else:
                dataset_format = request.dataset_format
            
            # Validate dataset
            validation_result = validate_dataset(
                request.dataset_path,
                format=dataset_format,
                sample_size=request.sample_size
            )
            
            return {
                "valid": validation_result["valid"],
                "format": dataset_format,
                "num_samples": validation_result["num_samples"],
                "issues": validation_result.get("issues", []),
                "warnings": validation_result.get("warnings", []),
                "sample": validation_result.get("sample")
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/v1/models")
    async def list_available_models():
        """List available models for fine-tuning"""
        return {
            "models": [
                {
                    "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                    "name": "Qwen 2.5 0.5B (4-bit)",
                    "size": "0.5B",
                    "quantization": "4bit",
                    "recommended": True
                },
                {
                    "id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", 
                    "name": "Qwen 2.5 1.5B (4-bit)",
                    "size": "1.5B",
                    "quantization": "4bit"
                },
                {
                    "id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "name": "Qwen 2.5 3B (4-bit)",
                    "size": "3B",
                    "quantization": "4bit"
                }
            ]
        }
    
    @app.get("/status")
    async def api_status():
        """Detailed API status"""
        integration_status = get_integration_status()
        
        return {
            "service": "training-mlx",
            "version": "0.1.0",
            "port": 8200,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "active_jobs": len([j for j in training_jobs.values() if j.status == "running"]),
            "total_jobs": len(training_jobs),
            "team_a_integration": integration_status,
            "features_enabled": {
                "lora": True,
                "distributed": integration_status["distributed_available"],
                "wandb": os.getenv("WANDB_API_KEY") is not None,
                "multi_gpu": False  # MLX doesn't support multi-GPU yet
            }
        }
    
    return app


async def run_training_job(job_id: str, request: TrainingRequest):
    """Background task to run training job"""
    job = training_jobs[job_id]
    
    try:
        # Update status
        job.status = "running"
        job.updated_at = datetime.utcnow()
        
        # Simulate training progress
        import asyncio
        for epoch in range(3):
            for step in range(100):
                await asyncio.sleep(0.1)  # Simulate work
                
                # Update progress
                job.progress = {
                    "stage": "training",
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "total_steps": 300,
                    "percent": ((epoch * 100 + step + 1) / 300) * 100
                }
                job.metrics = {
                    "loss": 2.5 - (epoch * 0.5) - (step * 0.001),
                    "learning_rate": 1e-4 * (1 - (epoch * 100 + step) / 300),
                }
                job.updated_at = datetime.utcnow()
        
        # Mark as completed
        job.status = "completed"
        job.progress["stage"] = "completed"
        job.updated_at = datetime.utcnow()
        
        logger.info(f"Completed training job: {job_id}")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.updated_at = datetime.utcnow()
        logger.error(f"Training job {job_id} failed: {e}")


# For running directly
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8200)
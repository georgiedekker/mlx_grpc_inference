#!/usr/bin/env python3
"""
Team B Complete API Modifications
Exact code changes needed to add LoRA and dataset support to the existing API.
"""

# ============================================================================
# STEP 1: UPDATE YOUR MAIN API FILE (e.g., main.py or app.py)
# ============================================================================

# Add these imports at the top of your existing API file:
API_IMPORTS_TO_ADD = '''
# Add these imports to your existing API file
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json
import os
from pathlib import Path

# LoRA integration imports
from mlx_distributed_training.integration.lora_integration import (
    LoRATrainingConfig, 
    create_lora_enabled_trainer,
    update_training_metrics_with_lora,
    save_lora_checkpoint
)

# Dataset integration imports  
from mlx_distributed_training.integration.dataset_integration import (
    validate_dataset,
    detect_dataset_format,
    create_dataset_loader_config,
    DatasetValidationResult
)

# Dataset loaders
from mlx_distributed_training.datasets.alpaca_dataset import AlpacaDataset
from mlx_distributed_training.datasets.sharegpt_dataset import ShareGPTDataset
from mlx_distributed_training.datasets.base_dataset import DatasetConfig as BaseDatasetConfig
'''

# ============================================================================
# STEP 2: UPDATE YOUR TRAINING REQUEST MODELS
# ============================================================================

UPDATED_REQUEST_MODELS = '''
# Replace or update your existing training request models with these:

class LoRAConfigRequest(BaseModel):
    """LoRA configuration for training requests."""
    use_lora: bool = Field(default=False, description="Enable LoRA fine-tuning")
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: float = Field(default=32.0, ge=0.1, description="LoRA scaling parameter")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="LoRA dropout rate")
    lora_target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Modules to apply LoRA to"
    )
    lora_modules_to_save: List[str] = Field(default=[], description="Additional modules to save")
    lora_bias: str = Field(default="none", regex="^(none|all|lora_only)$", description="Bias handling")
    use_qlora: bool = Field(default=False, description="Enable QLoRA (4-bit quantization)")
    qlora_compute_dtype: str = Field(default="float16", description="QLoRA compute dtype")


class DatasetConfigRequest(BaseModel):
    """Dataset configuration for training requests."""
    dataset_path: str = Field(..., description="Path to dataset file")
    dataset_format: Optional[str] = Field(
        default=None, 
        regex="^(alpaca|sharegpt)$",
        description="Dataset format: 'alpaca' or 'sharegpt' (auto-detect if not specified)"
    )
    max_seq_length: int = Field(default=2048, ge=128, le=8192, description="Maximum sequence length")
    batch_size: int = Field(default=8, ge=1, le=128, description="Training batch size")
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    validation_split: float = Field(default=0.0, ge=0.0, le=0.3, description="Validation split ratio")

    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Dataset file not found: {v}")
        return v


class TrainingJobRequest(BaseModel):
    """Enhanced training job request with LoRA and dataset support."""
    # Basic parameters
    model: str = Field(..., description="Model name or path")
    experiment_name: str = Field(..., description="Unique experiment name")
    
    # Dataset configuration
    training_file: str = Field(..., description="Path to training dataset")
    dataset_config: DatasetConfigRequest = Field(default_factory=DatasetConfigRequest)
    
    # Training hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # LoRA configuration
    lora_config: LoRAConfigRequest = Field(default_factory=LoRAConfigRequest)
    
    # Training parameters
    n_epochs: int = Field(default=3, ge=1, le=100, description="Number of epochs")
    learning_rate: float = Field(default=5e-5, ge=1e-7, le=1e-2, description="Learning rate")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    save_steps: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    output_dir: str = Field(default="./outputs", description="Output directory")
    
    # Advanced options
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Max gradient norm for clipping")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    seed: int = Field(default=42, description="Random seed")

    class Config:
        schema_extra = {
            "example": {
                "model": "mlx-community/Qwen2.5-1.5B-4bit",
                "experiment_name": "qwen_lora_alpaca_v1",
                "training_file": "alpaca_data.json",
                "dataset_config": {
                    "dataset_format": "alpaca",
                    "batch_size": 4,
                    "max_seq_length": 2048
                },
                "lora_config": {
                    "use_lora": True,
                    "lora_r": 8,
                    "lora_alpha": 16.0,
                    "use_qlora": True
                },
                "n_epochs": 3,
                "learning_rate": 5e-5
            }
        }


class TrainingJobResponse(BaseModel):
    """Training job response with enhanced information."""
    job_id: str
    experiment_name: str
    status: str
    created_at: datetime
    model: str
    
    # Dataset information
    dataset_info: Dict[str, Any] = Field(default_factory=dict)
    
    # LoRA information
    lora_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Training configuration
    training_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress information
    progress: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DatasetValidationRequest(BaseModel):
    """Request for dataset validation."""
    file_path: str = Field(..., description="Path to dataset file")
    max_samples: Optional[int] = Field(default=100, ge=1, le=1000, description="Maximum samples to validate")


class DatasetValidationResponse(BaseModel):
    """Response for dataset validation."""
    valid: bool
    format: str
    total_samples: int
    errors: List[str]
    warnings: List[str]
    sample_preview: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)
'''

# ============================================================================
# STEP 3: UPDATE YOUR TRAINING ENDPOINT
# ============================================================================

UPDATED_TRAINING_ENDPOINT = '''
# Replace your existing training endpoint with this enhanced version:

@app.post("/v1/fine-tuning/jobs", response_model=TrainingJobResponse)
async def create_fine_tuning_job(request: TrainingJobRequest):
    """Create a fine-tuning job with LoRA and dataset support."""
    
    try:
        # Generate job ID
        job_id = f"ftjob-{int(datetime.utcnow().timestamp())}-{request.experiment_name}"
        
        # Step 1: Validate and load dataset
        print(f"🔍 Validating dataset: {request.training_file}")
        
        # Auto-detect format if not specified
        dataset_format = request.dataset_config.dataset_format
        if not dataset_format:
            dataset_format = detect_dataset_format(request.training_file)
            if dataset_format == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail="Unable to detect dataset format. Please specify 'alpaca' or 'sharegpt'."
                )
        
        # Validate dataset
        validation_result = validate_dataset(request.training_file)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid dataset",
                    "errors": validation_result.errors[:5],
                    "format": validation_result.format_type
                }
            )
        
        # Create dataset loader config
        dataset_config = create_dataset_loader_config(
            dataset_path=request.training_file,
            format_type=dataset_format,
            batch_size=request.dataset_config.batch_size,
            max_seq_length=request.dataset_config.max_seq_length,
            shuffle=request.dataset_config.shuffle
        )
        
        # Load dataset
        print(f"📊 Loading {dataset_format} dataset...")
        base_config = BaseDatasetConfig(
            max_seq_length=request.dataset_config.max_seq_length,
            batch_size=request.dataset_config.batch_size,
            shuffle=request.dataset_config.shuffle
        )
        
        if dataset_format == "alpaca":
            dataset = AlpacaDataset(
                data_source=request.training_file,
                config=base_config
            )
        elif dataset_format == "sharegpt":
            dataset = ShareGPTDataset(
                data_source=request.training_file,
                config=base_config
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset format: {dataset_format}")
        
        dataset_info = {
            "format": dataset_format,
            "total_samples": len(dataset),
            "batch_size": request.dataset_config.batch_size,
            "max_seq_length": request.dataset_config.max_seq_length,
            "validation_warnings": validation_result.warnings[:3]
        }
        
        # Step 2: Load model
        print(f"🤖 Loading model: {request.model}")
        # Your existing model loading logic here
        # model = load_model(request.model)
        
        # Step 3: Apply LoRA if enabled
        lora_info = {"lora_enabled": False}
        if request.lora_config.use_lora:
            print(f"🔧 Applying LoRA with rank {request.lora_config.lora_r}")
            
            # Convert to internal LoRA config
            lora_training_config = {
                "use_lora": True,
                "lora_r": request.lora_config.lora_r,
                "lora_alpha": request.lora_config.lora_alpha,
                "lora_dropout": request.lora_config.lora_dropout,
                "lora_target_modules": request.lora_config.lora_target_modules,
                "use_qlora": request.lora_config.use_qlora
            }
            
            # Apply LoRA to model
            # model, lora_info = create_lora_enabled_trainer(model, lora_training_config)
            
            # Mock LoRA info for now
            lora_info = {
                "lora_enabled": True,
                "rank": request.lora_config.lora_r,
                "alpha": request.lora_config.lora_alpha,
                "target_modules": request.lora_config.lora_target_modules,
                "trainable_params": "TBD - integrate after copying lora.py"
            }
        
        # Step 4: Start training job
        print(f"🚀 Starting training job: {job_id}")
        
        # Create training configuration
        training_config = {
            "model": request.model,
            "experiment_name": request.experiment_name,
            "n_epochs": request.n_epochs,
            "learning_rate": request.learning_rate,
            "warmup_steps": request.warmup_steps,
            "save_steps": request.save_steps,
            "output_dir": request.output_dir,
            "seed": request.seed
        }
        
        # Your existing training job creation logic here
        # job_status = start_training_job(model, dataset, training_config, lora_info, dataset_info)
        
        # Mock response for now
        response = TrainingJobResponse(
            job_id=job_id,
            experiment_name=request.experiment_name,
            status="running",
            created_at=datetime.utcnow(),
            model=request.model,
            dataset_info=dataset_info,
            lora_info=lora_info,
            training_config=training_config,
            progress={
                "current_epoch": 0,
                "total_epochs": request.n_epochs,
                "current_step": 0,
                "total_steps": len(dataset) * request.n_epochs // request.dataset_config.batch_size
            },
            metrics={}
        )
        
        print(f"✅ Training job created successfully: {job_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Training job creation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Enhanced training status endpoint
@app.get("/v1/fine-tuning/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_fine_tuning_job(job_id: str):
    """Get training job status with LoRA and dataset information."""
    
    # Your existing job status logic
    # job_status = get_job_status(job_id)
    
    # Mock enhanced status for now
    if not job_id.startswith("ftjob-"):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Enhanced status with LoRA and dataset info
    enhanced_status = TrainingJobResponse(
        job_id=job_id,
        experiment_name="example_experiment",
        status="running",  # running, completed, failed, stopped
        created_at=datetime.utcnow(),
        model="mlx-community/Qwen2.5-1.5B-4bit",
        dataset_info={
            "format": "alpaca",
            "total_samples": 1000,
            "processed_samples": 500
        },
        lora_info={
            "lora_enabled": True,
            "rank": 8,
            "alpha": 16.0,
            "trainable_params": 1024000,
            "total_params": 1500000000
        },
        progress={
            "current_epoch": 1,
            "total_epochs": 3,
            "current_step": 100,
            "total_steps": 300,
            "completion_percentage": 33.33
        },
        metrics={
            "train_loss": 0.85,
            "learning_rate": 5e-5,
            "tokens_per_second": 1250,
            "gpu_memory_usage": "8.2GB"
        }
    )
    
    return enhanced_status
'''

# ============================================================================
# STEP 4: ADD NEW ENDPOINTS
# ============================================================================

NEW_ENDPOINTS = '''
# Add these new endpoints to your API:

@app.post("/v1/datasets/validate", response_model=DatasetValidationResponse)  
async def validate_training_dataset(request: DatasetValidationRequest):
    """Validate a training dataset and detect its format."""
    
    try:
        # Validate dataset
        validation_result = validate_dataset(request.file_path)
        
        # Generate recommendations
        recommendations = []
        if validation_result.format_type == "unknown":
            recommendations.append("Ensure your dataset follows Alpaca or ShareGPT format")
        if len(validation_result.warnings) > 0:
            recommendations.append("Review warnings to improve dataset quality")
        if validation_result.total_samples < 100:
            recommendations.append("Consider using more training samples for better results")
        
        return DatasetValidationResponse(
            valid=validation_result.is_valid,
            format=validation_result.format_type,
            total_samples=validation_result.total_samples,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
            sample_preview=validation_result.sample_preview,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/v1/datasets/formats")
async def get_supported_dataset_formats():
    """Get information about supported dataset formats."""
    
    return {
        "supported_formats": ["alpaca", "sharegpt"],
        "format_descriptions": {
            "alpaca": {
                "description": "Stanford Alpaca format with instruction/output pairs",
                "required_fields": ["instruction", "output"],
                "optional_fields": ["input", "system"],
                "example": {
                    "instruction": "What is the capital of France?",
                    "output": "The capital of France is Paris.",
                    "input": "",
                    "system": "You are a helpful assistant."
                }
            },
            "sharegpt": {
                "description": "ShareGPT format with conversation turns",
                "required_fields": ["conversations"],
                "conversation_structure": {
                    "from": "Role (human/gpt/system)",
                    "value": "Message content"
                },
                "example": {
                    "conversations": [
                        {"from": "human", "value": "What is the capital of France?"},
                        {"from": "gpt", "value": "The capital of France is Paris."}
                    ]
                }
            }
        }
    }


@app.get("/health")
async def health_check():
    """Enhanced health check with feature information."""
    
    # Test imports
    lora_available = True
    dataset_available = True
    
    try:
        from mlx_distributed_training.training.lora.lora import LoRAConfig
    except ImportError:
        lora_available = False
    
    try:
        from mlx_distributed_training.datasets.alpaca_dataset import AlpacaDataset
    except ImportError:
        dataset_available = False
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "service": "mlx-distributed-training",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "lora": lora_available,
            "qlora": lora_available,  # QLoRA depends on LoRA
            "dataset_formats": ["alpaca", "sharegpt"] if dataset_available else [],
            "providers": ["openai", "anthropic", "together", "ollama", "local_mlx"],
            "distributed": True,
            "model_quantization": True
        },
        "active_jobs": 0,  # Replace with actual count
        "system_info": {
            "gpu_available": True,  # Check actual GPU availability
            "memory_gb": 16.0  # Check actual memory  
        }
    }


@app.get("/v1/models")
async def list_available_models():
    """List models optimized for LoRA training."""
    
    return {
        "data": [
            {
                "id": "mlx-community/Qwen2.5-1.5B-4bit",
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "mlx-community",
                "description": "Quantized Qwen2.5 1.5B model optimized for LoRA",
                "recommended_lora_config": {
                    "lora_r": 8,
                    "lora_alpha": 16.0,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                }
            },
            {
                "id": "mlx-community/Llama-3.2-1B-4bit",
                "object": "model", 
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "mlx-community",
                "description": "Quantized Llama 3.2 1B model optimized for LoRA",
                "recommended_lora_config": {
                    "lora_r": 16,
                    "lora_alpha": 32.0,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                }
            }
        ]
    }
'''

# ============================================================================
# STEP 5: INTEGRATION HELPER FUNCTIONS
# ============================================================================

HELPER_FUNCTIONS = '''
# Add these helper functions to your API file:

def load_and_prepare_dataset(dataset_config: DatasetConfigRequest, training_file: str):
    """Load and prepare dataset for training."""
    
    # Detect format if not specified
    dataset_format = dataset_config.dataset_format
    if not dataset_format:
        dataset_format = detect_dataset_format(training_file)
    
    # Validate dataset
    validation_result = validate_dataset(training_file)
    if not validation_result.is_valid:
        raise ValueError(f"Invalid dataset: {validation_result.errors}")
    
    # Create dataset loader config
    loader_config = create_dataset_loader_config(
        dataset_path=training_file,
        format_type=dataset_format,
        batch_size=dataset_config.batch_size,
        max_seq_length=dataset_config.max_seq_length,
        shuffle=dataset_config.shuffle
    )
    
    # Load dataset
    base_config = BaseDatasetConfig(
        max_seq_length=dataset_config.max_seq_length,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle
    )
    
    if dataset_format == "alpaca":
        dataset = AlpacaDataset(data_source=training_file, config=base_config)
    elif dataset_format == "sharegpt":
        dataset = ShareGPTDataset(data_source=training_file, config=base_config)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    
    return dataset, {
        "format": dataset_format,
        "total_samples": len(dataset),
        "validation_warnings": validation_result.warnings[:3]
    }


def prepare_model_with_lora(model, lora_config: LoRAConfigRequest):
    """Prepare model with LoRA if enabled."""
    
    if not lora_config.use_lora:
        return model, {"lora_enabled": False}
    
    # Convert to internal config format
    training_config = {
        "use_lora": True,
        "lora_r": lora_config.lora_r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_target_modules": lora_config.lora_target_modules,
        "use_qlora": lora_config.use_qlora
    }
    
    # Apply LoRA
    model, lora_info = create_lora_enabled_trainer(model, training_config)
    
    return model, lora_info


def create_training_job_metadata(request: TrainingJobRequest, dataset_info: dict, lora_info: dict):
    """Create comprehensive training job metadata."""
    
    return {
        "job_config": {
            "model": request.model,
            "experiment_name": request.experiment_name,
            "training_parameters": {
                "n_epochs": request.n_epochs,
                "learning_rate": request.learning_rate,
                "batch_size": request.dataset_config.batch_size,
                "max_seq_length": request.dataset_config.max_seq_length,
                "warmup_steps": request.warmup_steps,
                "save_steps": request.save_steps
            }
        },
        "dataset_info": dataset_info,
        "lora_info": lora_info,
        "created_at": datetime.utcnow().isoformat(),
        "status": "initializing"
    }
'''

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

USAGE_EXAMPLES = '''
# ============================================================================
# USAGE EXAMPLES FOR TESTING
# ============================================================================

# 1. Test dataset validation
curl -X POST http://localhost:8200/v1/datasets/validate \\
  -H "Content-Type: application/json" \\
  -d '{
    "file_path": "examples/alpaca_example.json",
    "max_samples": 50
  }'

# 2. Start LoRA training job
curl -X POST http://localhost:8200/v1/fine-tuning/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mlx-community/Qwen2.5-1.5B-4bit",
    "experiment_name": "qwen_lora_test_v1",
    "training_file": "examples/alpaca_example.json",
    "dataset_config": {
      "dataset_format": "alpaca",
      "batch_size": 4,
      "max_seq_length": 2048
    },
    "lora_config": {
      "use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16.0,
      "lora_dropout": 0.1,
      "use_qlora": true
    },
    "n_epochs": 3,
    "learning_rate": 5e-5
  }'

# 3. Check training status
curl -X GET http://localhost:8200/v1/fine-tuning/jobs/ftjob-1234567890-experiment_name

# 4. Get supported formats
curl -X GET http://localhost:8200/v1/datasets/formats

# 5. Health check
curl -X GET http://localhost:8200/health

# 6. List available models
curl -X GET http://localhost:8200/v1/models
'''

def main():
    """Generate complete API modification guide."""
    
    print("=" * 80)
    print("TEAM B COMPLETE API MODIFICATIONS")
    print("=" * 80)
    
    sections = [
        ("1. REQUIRED IMPORTS", API_IMPORTS_TO_ADD),
        ("2. UPDATED REQUEST MODELS", UPDATED_REQUEST_MODELS),
        ("3. UPDATED TRAINING ENDPOINT", UPDATED_TRAINING_ENDPOINT),
        ("4. NEW ENDPOINTS", NEW_ENDPOINTS),
        ("5. HELPER FUNCTIONS", HELPER_FUNCTIONS),
        ("6. USAGE EXAMPLES", USAGE_EXAMPLES)
    ]
    
    for title, content in sections:
        print(f"\n{title}")
        print("=" * len(title))
        print(content)
    
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE!")
    print("=" * 80)
    print("Next steps:")
    print("1. Copy the code sections above into your API file")
    print("2. Run the auto-setup script: ./team_b_auto_setup.sh")
    print("3. Test with: python test_integration.py")
    print("4. Test the API endpoints with the curl examples")

if __name__ == "__main__":
    main()
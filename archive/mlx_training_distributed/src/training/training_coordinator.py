#!/usr/bin/env python3
"""
Training coordinator that integrates all training components
"""

import mlx
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import asyncio
from datetime import datetime
import os

from .mlx_trainer import MLXTrainer, TrainingConfig
from .lora_trainer import LoRATrainer, LoRAConfig
from .distributed_trainer import DistributedTrainer, DistributedConfig, launch_distributed_training
from .distillation_trainer import DistillationTrainer, DistillationConfig
from .rlhf_trainer import create_rlhf_trainer, RLHFConfig
from .optimizers import create_optimizer


class TrainingCoordinator:
    """Coordinates different training strategies and configurations."""
    
    def __init__(self, job_id: str, base_config: Dict[str, Any]):
        self.job_id = job_id
        self.base_config = base_config
        self.trainer = None
        self.status = "initialized"
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        
    def setup_training(self) -> str:
        """Setup appropriate trainer based on configuration."""
        training_type = self.base_config.get("training_type", "standard")
        
        # Create base training config
        base_training_config = TrainingConfig(
            model_name=self.base_config["model_name"],
            dataset_path=self.base_config["dataset_path"],
            output_dir=self.base_config.get("output_dir", f"./outputs/{self.job_id}"),
            epochs=self.base_config.get("epochs", 3),
            batch_size=self.base_config.get("batch_size", 4),
            learning_rate=self.base_config.get("learning_rate", 5e-5),
            gradient_accumulation_steps=self.base_config.get("gradient_accumulation_steps", 1),
            max_seq_length=self.base_config.get("max_seq_length", 2048),
            optimizer_type=self.base_config.get("optimizer_type", "adamw"),
            use_lora=self.base_config.get("use_lora", False),
            distributed=self.base_config.get("distributed", False)
        )
        
        # Create base trainer
        base_trainer = MLXTrainer(base_training_config)
        
        # Setup specific training type
        if training_type == "lora" or base_training_config.use_lora:
            # LoRA training
            lora_config = LoRAConfig(
                r=self.base_config.get("lora_rank", 16),
                alpha=self.base_config.get("lora_alpha", 32.0),
                dropout=self.base_config.get("lora_dropout", 0.05),
                target_modules=self.base_config.get("lora_target_modules"),
                use_qora=self.base_config.get("use_qora", False)
            )
            self.trainer = LoRATrainer(base_trainer, lora_config)
            return "LoRA trainer initialized"
            
        elif training_type == "distributed" or base_training_config.distributed:
            # Distributed training
            dist_config = DistributedConfig(
                world_size=self.base_config.get("num_devices", 1),
                backend=self.base_config.get("distributed_backend", "allreduce")
            )
            self.trainer = DistributedTrainer(base_trainer, dist_config)
            return "Distributed trainer initialized"
            
        elif training_type == "distillation":
            # Knowledge distillation
            distill_config = DistillationConfig(
                teacher_models=self.base_config["teacher_models"],
                student_model=self.base_config["model_name"],
                temperature=self.base_config.get("temperature", 3.0),
                alpha=self.base_config.get("distillation_alpha", 0.7),
                feature_matching=self.base_config.get("feature_matching", True),
                adaptive_temperature=self.base_config.get("adaptive_temperature", True)
            )
            self.trainer = DistillationTrainer(base_trainer, distill_config)
            return "Distillation trainer initialized"
            
        elif training_type == "rlhf":
            # RLHF training
            rlhf_config = RLHFConfig(
                method=self.base_config.get("rlhf_method", "dpo"),
                beta=self.base_config.get("rlhf_beta", 0.1),
                learning_rate=self.base_config.get("learning_rate", 5e-7),
                epochs=self.base_config.get("epochs", 1),
                batch_size=self.base_config.get("batch_size", 2),
                preference_dataset=self.base_config["preference_dataset"],
                reward_model_path=self.base_config.get("reward_model_path")
            )
            self.trainer = create_rlhf_trainer(base_trainer, rlhf_config)
            return f"RLHF ({rlhf_config.method}) trainer initialized"
            
        else:
            # Standard training
            self.trainer = base_trainer
            return "Standard trainer initialized"
            
    async def start_training(self) -> Dict[str, Any]:
        """Start the training process."""
        self.status = "running"
        self.start_time = datetime.now()
        
        try:
            # Setup trainer
            setup_msg = self.setup_training()
            print(f"[{self.job_id}] {setup_msg}")
            
            # Load model and dataset
            if hasattr(self.trainer, 'prepare_model'):
                self.trainer.prepare_model()
            else:
                self.trainer.load_model()
                self.trainer.load_dataset()
                self.trainer.create_optimizer()
            
            # Run training
            if isinstance(self.trainer, DistributedTrainer) and self.trainer.config.world_size > 1:
                # Launch distributed training
                results = await self._run_distributed_training()
            else:
                # Run single process training
                results = await self._run_single_training()
                
            self.status = "completed"
            self.metrics = results
            
        except Exception as e:
            self.status = "failed"
            self.metrics["error"] = str(e)
            raise e
            
        finally:
            self.end_time = datetime.now()
            
        return self.get_status()
        
    async def _run_single_training(self) -> Dict[str, Any]:
        """Run training in single process."""
        # Run training synchronously in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.trainer.train)
        return results
        
    async def _run_distributed_training(self) -> Dict[str, Any]:
        """Run distributed training across multiple processes."""
        # This would launch multiple processes
        # For now, return placeholder
        return {
            "status": "distributed training would launch here",
            "world_size": self.trainer.config.world_size
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        status_info = {
            "job_id": self.job_id,
            "status": self.status,
            "training_type": self.base_config.get("training_type", "standard"),
            "model": self.base_config["model_name"],
            "dataset": self.base_config["dataset_path"],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics
        }
        
        # Add trainer-specific info
        if hasattr(self.trainer, 'config'):
            status_info["config"] = self.trainer.config.__dict__ if hasattr(self.trainer.config, '__dict__') else self.trainer.config
            
        return status_info
        
    def stop_training(self):
        """Stop the training process."""
        # This would implement graceful shutdown
        self.status = "stopped"
        return {"status": "Training stopped"}
        

class TrainingOrchestrator:
    """Orchestrates multiple training jobs and pipelines."""
    
    def __init__(self):
        self.jobs = {}
        self.pipelines = {}
        
    async def create_job(self, job_config: Dict[str, Any]) -> str:
        """Create a new training job."""
        job_id = f"job-{len(self.jobs) + 1:06d}"
        coordinator = TrainingCoordinator(job_id, job_config)
        self.jobs[job_id] = coordinator
        
        # Start training asynchronously
        asyncio.create_task(coordinator.start_training())
        
        return job_id
        
    async def create_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Create a multi-stage training pipeline."""
        pipeline_id = f"pipeline-{len(self.pipelines) + 1:06d}"
        
        pipeline = {
            "id": pipeline_id,
            "config": pipeline_config,
            "stages": pipeline_config["stages"],
            "current_stage": 0,
            "status": "created",
            "jobs": []
        }
        
        self.pipelines[pipeline_id] = pipeline
        return pipeline_id
        
    async def run_pipeline(self, pipeline_id: str):
        """Execute a training pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        pipeline["status"] = "running"
        base_config = pipeline["config"]
        
        for i, stage in enumerate(pipeline["stages"]):
            pipeline["current_stage"] = i
            
            # Create stage-specific config
            stage_config = base_config.copy()
            stage_config["training_type"] = stage
            
            # Adjust config based on stage
            if stage == "sft":
                # Standard fine-tuning
                pass
            elif stage == "distillation":
                # Use previous stage output as student
                if i > 0 and pipeline["jobs"]:
                    prev_job = self.jobs[pipeline["jobs"][-1]]
                    stage_config["model_name"] = f"{prev_job.base_config['output_dir']}/final_model"
            elif stage == "rlhf":
                # Use previous stage output for RLHF
                if i > 0 and pipeline["jobs"]:
                    prev_job = self.jobs[pipeline["jobs"][-1]]
                    stage_config["model_name"] = f"{prev_job.base_config['output_dir']}/final_model"
                    
            # Create and run job
            job_id = await self.create_job(stage_config)
            pipeline["jobs"].append(job_id)
            
            # Wait for job completion
            await self._wait_for_job(job_id)
            
        pipeline["status"] = "completed"
        
    async def _wait_for_job(self, job_id: str):
        """Wait for a job to complete."""
        while True:
            job = self.jobs.get(job_id)
            if job and job.status in ["completed", "failed", "stopped"]:
                break
            await asyncio.sleep(5)
            
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        return job.get_status()
        
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get status of a pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        status = {
            "id": pipeline_id,
            "status": pipeline["status"],
            "current_stage": pipeline["current_stage"],
            "total_stages": len(pipeline["stages"]),
            "stages": pipeline["stages"],
            "jobs": []
        }
        
        # Add job statuses
        for job_id in pipeline.get("jobs", []):
            if job_id in self.jobs:
                status["jobs"].append(self.jobs[job_id].get_status())
                
        return status
        
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return [job.get_status() for job in self.jobs.values()]
        
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines."""
        return [self.get_pipeline_status(pid) for pid in self.pipelines.keys()]


# Global orchestrator instance
orchestrator = TrainingOrchestrator()
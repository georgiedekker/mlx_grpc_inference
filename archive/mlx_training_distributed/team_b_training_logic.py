"""
Team B Enhanced Training Logic with LoRA Support
This file provides a complete training pipeline with LoRA integration
that can be dropped into Team B's existing training infrastructure.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Import LoRA components
from mlx_distributed_training.training.lora.lora import (
    apply_lora_to_model,
    LoRAConfig,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    freeze_base_model,
    get_lora_parameters,
    print_lora_info
)

# Import dataset components
from mlx_distributed_training.datasets.base_dataset import BaseDataset
from mlx_distributed_training.datasets.alpaca_dataset import AlpacaDataset
from mlx_distributed_training.datasets.sharegpt_dataset import ShareGPTDataset


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    learning_rate: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_memory_gb: float = 0.0
    gradient_norm: float = 0.0
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gradient_norm": self.gradient_norm,
            "epoch": self.epoch,
            "step": self.step,
            "total_steps": self.total_steps,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class EnhancedTrainer:
    """Enhanced trainer with LoRA support and advanced features."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: BaseDataset,
        config: Dict[str, Any],
        job_id: str,
        tokenizer: Any = None
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.job_id = job_id
        self.tokenizer = tokenizer
        
        # Training configuration
        self.epochs = config.get("epochs", 3)
        self.learning_rate = config.get("learning_rate", 5e-5)
        self.batch_size = config.get("dataset", {}).get("batch_size", 8)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.save_steps = config.get("save_steps", 500)
        self.output_dir = Path(config.get("output_dir", "./outputs")) / job_id
        
        # LoRA configuration
        self.lora_config = config.get("lora", {})
        self.use_lora = self.lora_config.get("use_lora", False)
        self.lora_info = None
        
        # Setup model with LoRA if enabled
        if self.use_lora:
            self._setup_lora()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial configuration
        self._save_config()
    
    def _setup_lora(self):
        """Setup LoRA on the model."""
        print(f"🔧 Setting up LoRA for training...")
        
        # Create LoRA configuration
        lora_config = LoRAConfig(
            r=self.lora_config.get("lora_r", 16),
            alpha=self.lora_config.get("lora_alpha", 32.0),
            dropout=self.lora_config.get("lora_dropout", 0.1),
            target_modules=self.lora_config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            use_qlora=self.lora_config.get("use_qlora", False)
        )
        
        # Apply LoRA to model
        self.model = apply_lora_to_model(self.model, lora_config)
        
        # Freeze base model parameters
        freeze_base_model(self.model)
        
        # Get LoRA parameter information
        lora_params = get_lora_parameters(self.model)
        total_params = sum(p.size for p in self.model.parameters())
        trainable_params = sum(p.size for p in lora_params.values())
        
        self.lora_info = {
            "enabled": True,
            "config": lora_config.__dict__,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "compression_ratio": total_params / trainable_params if trainable_params > 0 else 1.0,
            "memory_savings_pct": (1 - trainable_params / total_params) * 100
        }
        
        # Print LoRA information
        print_lora_info(self.model, lora_config)
        print(f"💾 Memory savings: {self.lora_info['memory_savings_pct']:.1f}%")
        print(f"🚀 Parameter reduction: {self.lora_info['compression_ratio']:.1f}x")
    
    def _setup_optimizer(self):
        """Setup optimizer with appropriate parameters."""
        if self.use_lora:
            # Only optimize LoRA parameters
            lora_params = get_lora_parameters(self.model)
            self.optimizer = optim.AdamW(
                learning_rate=self.learning_rate,
                betas=[0.9, 0.999],
                weight_decay=0.01
            )
            # Initialize optimizer state only for LoRA parameters
            self.optimizer_state = self.optimizer.init(lora_params)
        else:
            # Optimize all parameters
            self.optimizer = optim.AdamW(
                learning_rate=self.learning_rate,
                betas=[0.9, 0.999],
                weight_decay=0.01
            )
            self.optimizer_state = self.optimizer.init(self.model.parameters())
    
    def _save_config(self):
        """Save training configuration."""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "job_id": self.job_id,
                "config": self.config,
                "lora_info": self.lora_info,
                "dataset_info": {
                    "total_samples": len(self.dataset),
                    "batch_size": self.batch_size,
                    "max_seq_length": getattr(self.dataset.config, "max_seq_length", 2048)
                }
            }, f, indent=2)
    
    def compute_loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute loss for a batch."""
        # Forward pass
        logits = self.model(batch["input_ids"])
        
        # Compute cross-entropy loss
        # Shift labels for next-token prediction
        labels = batch["labels"]
        logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        labels = labels[..., 1:].reshape(-1)
        
        # Compute loss with label smoothing
        loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction='none'))
        
        return loss
    
    def training_step(self, batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, float]]:
        """Perform a single training step."""
        # Define loss function for autograd
        def loss_fn(model, batch):
            return self.compute_loss(batch)
        
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.model, batch)
        
        # Clip gradients
        grads = tree_flatten(grads)
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in grads[0]))
        
        # Update parameters
        if self.use_lora:
            # Update only LoRA parameters
            lora_params = get_lora_parameters(self.model)
            self.optimizer_state, lora_params = self.optimizer.update(
                self.optimizer_state, grads, lora_params
            )
            # Update model with new LoRA parameters
            for name, param in lora_params.items():
                setattr(self.model, name, param)
        else:
            # Update all parameters
            self.optimizer_state, self.model = self.optimizer.update(
                self.optimizer_state, grads, self.model
            )
        
        # Compute metrics
        metrics = {
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "learning_rate": self._get_learning_rate(),
            "gpu_memory_gb": mx.metal.get_active_memory() / 1e9 if mx.metal.is_available() else 0
        }
        
        return loss, metrics
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate with warmup and decay."""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.global_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.learning_rate * (0.5 * (1 + mx.cos(mx.pi * progress)))
    
    def save_checkpoint(self, step: Optional[int] = None, is_final: bool = False):
        """Save model checkpoint."""
        if step is None:
            step = self.global_step
        
        checkpoint_dir = self.output_dir / f"checkpoint-{step}" if not is_final else self.output_dir / "final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            # Save only LoRA weights
            save_lora_weights(
                self.model,
                checkpoint_dir / "lora_weights.safetensors",
                metadata={
                    "step": step,
                    "epoch": self.current_epoch,
                    "loss": self.best_loss,
                    "lora_config": self.lora_info["config"]
                }
            )
            print(f"💾 Saved LoRA checkpoint to {checkpoint_dir}")
        else:
            # Save full model weights
            mx.save(checkpoint_dir / "model_weights.safetensors", dict(self.model.parameters()))
            print(f"💾 Saved full model checkpoint to {checkpoint_dir}")
        
        # Save training state
        state = {
            "step": step,
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer_state,
            "training_history": self.training_history[-100:]  # Keep last 100 entries
        }
        mx.save(checkpoint_dir / "training_state.safetensors", state)
        
        # Save tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(checkpoint_dir)
    
    def train(self) -> Dict[str, Any]:
        """Run the complete training loop."""
        print(f"🚀 Starting training for {self.epochs} epochs")
        print(f"📊 Total samples: {len(self.dataset)}")
        print(f"🔢 Batch size: {self.batch_size}")
        
        # Calculate total steps
        steps_per_epoch = len(self.dataset) // self.batch_size
        self.total_steps = steps_per_epoch * self.epochs
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            # Training epoch
            for step, batch in enumerate(self.dataset):
                # Training step
                loss, metrics = self.training_step(batch)
                
                # Update tracking
                self.global_step += 1
                epoch_loss += metrics["loss"]
                
                # Record metrics
                training_metrics = TrainingMetrics(
                    loss=metrics["loss"],
                    learning_rate=metrics["learning_rate"],
                    gradient_norm=metrics["grad_norm"],
                    gpu_memory_gb=metrics["gpu_memory_gb"],
                    epoch=epoch,
                    step=self.global_step,
                    total_steps=self.total_steps
                )
                self.training_history.append(training_metrics.to_dict())
                
                # Log progress
                if step % 10 == 0:
                    progress = (self.global_step / self.total_steps) * 100
                    print(f"Epoch {epoch+1}/{self.epochs} | Step {step}/{steps_per_epoch} | "
                          f"Loss: {metrics['loss']:.4f} | Progress: {progress:.1f}%")
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
                
                # Update best loss
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / steps_per_epoch
            epoch_time = time.time() - epoch_start_time
            print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Training completed
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time/60:.1f} minutes")
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Prepare final results
        results = {
            "job_id": self.job_id,
            "status": "completed",
            "total_time_seconds": total_time,
            "final_loss": self.best_loss,
            "total_steps": self.global_step,
            "checkpoints": list(self.output_dir.glob("checkpoint-*")),
            "final_model_path": str(self.output_dir / "final"),
            "lora_info": self.lora_info
        }
        
        return results


class AsyncTrainingManager:
    """Async manager for running training jobs in the background."""
    
    def __init__(self):
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.job_results: Dict[str, Dict[str, Any]] = {}
    
    async def start_training(
        self,
        job_id: str,
        model: nn.Module,
        dataset: BaseDataset,
        config: Dict[str, Any],
        tokenizer: Any = None,
        callback: Optional[callable] = None
    ) -> str:
        """Start an async training job."""
        # Create trainer
        trainer = EnhancedTrainer(
            model=model,
            dataset=dataset,
            config=config,
            job_id=job_id,
            tokenizer=tokenizer
        )
        
        # Create async task
        task = asyncio.create_task(self._run_training(trainer, callback))
        self.active_jobs[job_id] = task
        
        return job_id
    
    async def _run_training(self, trainer: EnhancedTrainer, callback: Optional[callable] = None):
        """Run training in async context."""
        try:
            # Run training (this is CPU-bound, so we run it in executor)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, trainer.train)
            
            # Store results
            self.job_results[trainer.job_id] = results
            
            # Call callback if provided
            if callback:
                await callback(trainer.job_id, results)
            
        except Exception as e:
            # Store error
            self.job_results[trainer.job_id] = {
                "job_id": trainer.job_id,
                "status": "failed",
                "error": str(e)
            }
            
            if callback:
                await callback(trainer.job_id, self.job_results[trainer.job_id])
        
        finally:
            # Remove from active jobs
            if trainer.job_id in self.active_jobs:
                del self.active_jobs[trainer.job_id]
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current job status."""
        if job_id in self.active_jobs:
            return {"status": "running", "job_id": job_id}
        elif job_id in self.job_results:
            return self.job_results[job_id]
        else:
            return {"status": "not_found", "job_id": job_id}
    
    async def stop_training(self, job_id: str) -> bool:
        """Stop a running training job."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            return True
        return False


# ==============================================================================
# INTEGRATION EXAMPLE
# ==============================================================================

async def create_training_job_with_lora(
    model_name: str,
    dataset_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Example function showing how to create a training job with LoRA.
    
    This demonstrates the complete integration flow.
    """
    # 1. Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Validate and load dataset
    from mlx_distributed_training.integration.dataset_integration import validate_dataset
    
    validation_result = validate_dataset(dataset_path)
    if not validation_result.is_valid:
        raise ValueError(f"Invalid dataset: {validation_result.errors}")
    
    # 3. Create dataset based on format
    if validation_result.format_type == "alpaca":
        dataset = AlpacaDataset(
            data_source=dataset_path,
            config=BaseDatasetConfig(
                max_seq_length=config.get("max_seq_length", 2048),
                batch_size=config.get("batch_size", 8),
                shuffle=True
            ),
            tokenizer=tokenizer
        )
    else:
        dataset = ShareGPTDataset(
            data_source=dataset_path,
            config=BaseDatasetConfig(
                max_seq_length=config.get("max_seq_length", 2048),
                batch_size=config.get("batch_size", 8),
                shuffle=True
            ),
            tokenizer=tokenizer
        )
    
    # 4. Create training manager
    manager = AsyncTrainingManager()
    
    # 5. Start training with LoRA
    job_id = f"lora_job_{int(time.time())}"
    await manager.start_training(
        job_id=job_id,
        model=model,
        dataset=dataset,
        config=config,
        tokenizer=tokenizer,
        callback=lambda jid, results: print(f"Job {jid} completed: {results['status']}")
    )
    
    return {
        "job_id": job_id,
        "status": "started",
        "dataset_format": validation_result.format_type,
        "lora_enabled": config.get("lora", {}).get("use_lora", False)
    }


# Example configuration for LoRA training
EXAMPLE_LORA_CONFIG = {
    "epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "save_steps": 500,
    "output_dir": "./outputs",
    "lora": {
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "use_qlora": False
    }
}

# Usage example:
# asyncio.run(create_training_job_with_lora(
#     model_name="mlx-community/Qwen2.5-1.5B-4bit",
#     dataset_path="path/to/alpaca_dataset.json",
#     config=EXAMPLE_LORA_CONFIG
# ))
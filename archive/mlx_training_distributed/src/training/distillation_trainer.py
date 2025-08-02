#!/usr/bin/env python3
"""
Knowledge Distillation implementation for MLX
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_models: List[str]
    student_model: str
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for distillation loss
    distillation_loss: str = "kl_div"  # kl_div, mse, cosine
    feature_matching: bool = True
    intermediate_layers: List[int] = None
    adaptive_temperature: bool = True
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    temperature_schedule: str = "linear"  # linear, exponential, cosine
    multi_teacher_strategy: str = "average"  # average, weighted, best
    teacher_weights: Optional[List[float]] = None
    

class MultiTeacherDistillation:
    """Multi-teacher knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teachers = []
        self.student = None
        self.temperature = config.temperature
        self.teacher_weights = config.teacher_weights
        
    def load_teachers(self):
        """Load all teacher models."""
        from mlx_lm import load
        
        print(f"Loading {len(self.config.teacher_models)} teacher models...")
        for teacher_name in self.config.teacher_models:
            print(f"  Loading teacher: {teacher_name}")
            model, tokenizer = load(teacher_name)
            model.eval()  # Teachers always in eval mode
            self.teachers.append({
                "name": teacher_name,
                "model": model,
                "tokenizer": tokenizer
            })
            
        # Set teacher weights if not provided
        if self.teacher_weights is None:
            self.teacher_weights = [1.0 / len(self.teachers)] * len(self.teachers)
            
    def load_student(self):
        """Load student model."""
        from mlx_lm import load
        
        print(f"Loading student model: {self.config.student_model}")
        self.student, self.student_tokenizer = load(self.config.student_model)
        self.student.train()
        
    def get_teacher_outputs(self, input_ids: mx.array, attention_mask: mx.array) -> Dict[str, Any]:
        """Get outputs from all teachers."""
        teacher_outputs = []
        
        for teacher in self.teachers:
            with mx.no_grad():
                outputs = teacher["model"](input_ids, attention_mask=attention_mask)
                teacher_outputs.append({
                    "logits": outputs.logits,
                    "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
                    "attentions": outputs.attentions if hasattr(outputs, "attentions") else None
                })
                
        return teacher_outputs
        
    def compute_distillation_loss(
        self,
        student_logits: mx.array,
        teacher_outputs: List[Dict[str, Any]],
        labels: Optional[mx.array] = None
    ) -> mx.array:
        """Compute knowledge distillation loss."""
        
        # Get teacher logits
        teacher_logits = [output["logits"] for output in teacher_outputs]
        
        # Apply temperature scaling
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = [logits / self.temperature for logits in teacher_logits]
        
        # Compute soft targets from teachers
        if self.config.multi_teacher_strategy == "average":
            # Average teacher predictions
            teacher_probs = []
            for logits in teacher_logits_scaled:
                probs = nn.softmax(logits, axis=-1)
                teacher_probs.append(probs)
            avg_teacher_probs = sum(p * w for p, w in zip(teacher_probs, self.teacher_weights))
            
        elif self.config.multi_teacher_strategy == "weighted":
            # Weighted combination based on teacher confidence
            teacher_probs = []
            confidences = []
            for logits in teacher_logits_scaled:
                probs = nn.softmax(logits, axis=-1)
                # Confidence as negative entropy
                entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
                confidence = 1.0 / (1.0 + entropy)
                teacher_probs.append(probs)
                confidences.append(confidence)
                
            # Normalize confidences
            total_conf = sum(confidences)
            weights = [c / total_conf for c in confidences]
            avg_teacher_probs = sum(p * w for p, w in zip(teacher_probs, weights))
            
        elif self.config.multi_teacher_strategy == "best":
            # Select best teacher per example
            best_teacher_idx = self._select_best_teacher(teacher_logits, labels)
            avg_teacher_probs = nn.softmax(teacher_logits_scaled[best_teacher_idx], axis=-1)
            
        # Compute distillation loss
        if self.config.distillation_loss == "kl_div":
            # KL divergence loss
            student_log_probs = nn.log_softmax(student_logits_scaled, axis=-1)
            distill_loss = -mx.sum(avg_teacher_probs * student_log_probs, axis=-1)
            distill_loss = mx.mean(distill_loss)
            
        elif self.config.distillation_loss == "mse":
            # MSE loss on logits
            avg_teacher_logits = sum(l * w for l, w in zip(teacher_logits, self.teacher_weights))
            distill_loss = mx.mean((student_logits - avg_teacher_logits) ** 2)
            
        elif self.config.distillation_loss == "cosine":
            # Cosine similarity loss
            avg_teacher_logits = sum(l * w for l, w in zip(teacher_logits, self.teacher_weights))
            student_norm = mx.sqrt(mx.sum(student_logits ** 2, axis=-1, keepdims=True))
            teacher_norm = mx.sqrt(mx.sum(avg_teacher_logits ** 2, axis=-1, keepdims=True))
            cosine_sim = mx.sum(student_logits * avg_teacher_logits, axis=-1) / (student_norm * teacher_norm + 1e-8)
            distill_loss = 1.0 - mx.mean(cosine_sim)
            
        return distill_loss * self.temperature ** 2  # Scale by T^2 as per Hinton et al.
        
    def compute_feature_matching_loss(
        self,
        student_hidden: List[mx.array],
        teacher_hidden: List[List[mx.array]]
    ) -> mx.array:
        """Compute feature matching loss for intermediate layers."""
        if not self.config.feature_matching or not student_hidden:
            return mx.array(0.0)
            
        feature_loss = mx.array(0.0)
        num_layers = 0
        
        # Match intermediate layers
        for layer_idx in (self.config.intermediate_layers or []):
            if layer_idx < len(student_hidden):
                student_features = student_hidden[layer_idx]
                
                # Average teacher features
                teacher_features = []
                for t_hidden in teacher_hidden:
                    if layer_idx < len(t_hidden):
                        teacher_features.append(t_hidden[layer_idx])
                        
                if teacher_features:
                    avg_teacher_features = sum(f * w for f, w in zip(teacher_features, self.teacher_weights))
                    
                    # L2 loss on features
                    layer_loss = mx.mean((student_features - avg_teacher_features) ** 2)
                    feature_loss = feature_loss + layer_loss
                    num_layers += 1
                    
        return feature_loss / max(num_layers, 1)
        
    def update_temperature(self, step: int, total_steps: int):
        """Update temperature based on schedule."""
        if not self.config.adaptive_temperature:
            return
            
        progress = step / total_steps
        
        if self.config.temperature_schedule == "linear":
            # Linear decay
            self.temperature = self.config.max_temperature - (self.config.max_temperature - self.config.min_temperature) * progress
            
        elif self.config.temperature_schedule == "exponential":
            # Exponential decay
            decay_rate = np.log(self.config.min_temperature / self.config.max_temperature)
            self.temperature = self.config.max_temperature * np.exp(decay_rate * progress)
            
        elif self.config.temperature_schedule == "cosine":
            # Cosine annealing
            self.temperature = self.config.min_temperature + 0.5 * (self.config.max_temperature - self.config.min_temperature) * (1 + np.cos(np.pi * progress))
            
    def _select_best_teacher(self, teacher_logits: List[mx.array], labels: mx.array) -> int:
        """Select best teacher based on cross-entropy with labels."""
        if labels is None:
            return 0
            
        min_loss = float('inf')
        best_idx = 0
        
        for i, logits in enumerate(teacher_logits):
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if loss < min_loss:
                min_loss = loss
                best_idx = i
                
        return best_idx


class DistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(self, base_trainer, distillation_config: DistillationConfig):
        self.base_trainer = base_trainer
        self.config = distillation_config
        self.distiller = MultiTeacherDistillation(distillation_config)
        
    def setup(self):
        """Setup distillation training."""
        # Load teachers and student
        self.distiller.load_teachers()
        self.distiller.load_student()
        
        # Update base trainer's model
        self.base_trainer.model = self.distiller.student
        self.base_trainer.tokenizer = self.distiller.student_tokenizer
        
        # Create optimizer for student
        self.base_trainer.create_optimizer()
        
        # Load dataset
        self.base_trainer.load_dataset()
        
    def training_step(self, batch: Dict[str, Any], step: int, total_steps: int) -> Dict[str, float]:
        """Single distillation training step."""
        # Update temperature
        self.distiller.update_temperature(step, total_steps)
        
        # Get teacher outputs
        teacher_outputs = self.distiller.get_teacher_outputs(
            batch["input_ids"],
            batch["attention_mask"]
        )
        
        def loss_fn(model):
            # Student forward pass
            student_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=self.config.feature_matching,
                output_attentions=False
            )
            
            # Standard cross-entropy loss
            labels = batch["labels"]
            shift_logits = student_outputs.logits[..., :-1, :].reshape(-1, student_outputs.logits.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            ce_loss = nn.losses.cross_entropy(shift_logits, shift_labels)
            
            # Distillation loss
            distill_loss = self.distiller.compute_distillation_loss(
                student_outputs.logits,
                teacher_outputs,
                labels
            )
            
            # Feature matching loss
            feature_loss = mx.array(0.0)
            if self.config.feature_matching and hasattr(student_outputs, "hidden_states"):
                teacher_hidden = [t.get("hidden_states", []) for t in teacher_outputs]
                feature_loss = self.distiller.compute_feature_matching_loss(
                    student_outputs.hidden_states,
                    teacher_hidden
                )
                
            # Combined loss
            total_loss = (1 - self.config.alpha) * ce_loss + self.config.alpha * distill_loss + 0.1 * feature_loss
            
            return total_loss, {"ce_loss": ce_loss, "distill_loss": distill_loss, "feature_loss": feature_loss}
            
        # Compute loss and gradients
        (loss, loss_components), grads = mx.value_and_grad(loss_fn, has_aux=True)(self.base_trainer.model)
        
        # Update weights
        self.base_trainer.optimizer.update(self.base_trainer.model, grads)
        
        # Evaluate losses
        mx.eval(loss)
        mx.eval(loss_components["ce_loss"])
        mx.eval(loss_components["distill_loss"])
        mx.eval(loss_components["feature_loss"])
        
        return {
            "total_loss": loss.item(),
            "ce_loss": loss_components["ce_loss"].item(),
            "distill_loss": loss_components["distill_loss"].item(),
            "feature_loss": loss_components["feature_loss"].item(),
            "temperature": self.distiller.temperature
        }
        
    def train(self):
        """Distillation training loop."""
        print("Starting knowledge distillation training...")
        self.setup()
        
        self.distiller.student.train()
        total_steps = len(self.base_trainer.train_dataset) // self.base_trainer.config.batch_size * self.base_trainer.config.epochs
        step = 0
        
        history = []
        
        for epoch in range(self.base_trainer.config.epochs):
            epoch_metrics = {"total_loss": 0, "ce_loss": 0, "distill_loss": 0, "feature_loss": 0}
            num_batches = 0
            
            # Shuffle dataset
            indices = np.random.permutation(len(self.base_trainer.train_dataset))
            
            for i in range(0, len(self.base_trainer.train_dataset), self.base_trainer.config.batch_size):
                batch_indices = indices[i:i + self.base_trainer.config.batch_size]
                batch_data = [self.base_trainer.train_dataset[idx] for idx in batch_indices]
                batch = self.base_trainer.prepare_batch(batch_data)
                
                # Training step
                metrics = self.training_step(batch, step, total_steps)
                
                # Update metrics
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                        
                num_batches += 1
                step += 1
                
                # Logging
                if step % self.base_trainer.config.logging_steps == 0:
                    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
                    print(f"Step {step}/{total_steps} | Loss: {avg_metrics['total_loss']:.4f} | "
                          f"CE: {avg_metrics['ce_loss']:.4f} | Distill: {avg_metrics['distill_loss']:.4f} | "
                          f"T: {metrics['temperature']:.2f}")
                    
                    history.append({
                        "step": step,
                        "epoch": epoch,
                        **avg_metrics,
                        "temperature": metrics["temperature"]
                    })
                    
                # Save checkpoint
                if step % self.base_trainer.config.save_steps == 0:
                    self.save_checkpoint(f"{self.base_trainer.config.output_dir}/distill_step_{step}")
                    
        # Final save
        self.save_checkpoint(f"{self.base_trainer.config.output_dir}/distilled_final")
        
        return {
            "final_metrics": {k: v / num_batches for k, v in epoch_metrics.items()},
            "history": history
        }
        
    def save_checkpoint(self, path: str):
        """Save distilled model checkpoint."""
        self.base_trainer.save_checkpoint(path)
        
        # Save distillation config
        config_data = {
            "teacher_models": self.config.teacher_models,
            "student_model": self.config.student_model,
            "temperature": self.distiller.temperature,
            "alpha": self.config.alpha,
            "distillation_config": self.config.__dict__
        }
        
        with open(f"{path}/distillation_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
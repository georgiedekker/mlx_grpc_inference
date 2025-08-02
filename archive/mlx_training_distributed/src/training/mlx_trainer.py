#!/usr/bin/env python3
"""
Actual MLX training implementation
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time
from datetime import datetime

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str
    dataset_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 2048
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    optimizer_type: str = "adamw"
    distributed: bool = False
    mixed_precision: bool = True

class MLXTrainer:
    """Core MLX trainer implementation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.tokenizer = None
        self.step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
    def load_model(self):
        """Load MLX model and tokenizer."""
        from mlx_lm import load
        
        print(f"Loading model: {self.config.model_name}")
        self.model, self.tokenizer = load(self.config.model_name)
        
        if self.config.use_lora:
            self._apply_lora()
            
        # Move model to MLX device
        self.model.eval()
        return self.model
        
    def _apply_lora(self):
        """Apply LoRA to the model."""
        from mlx_lm.tuner.lora import LoRALinear
        
        # Replace linear layers with LoRA layers
        def replace_with_lora(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with LoRA linear
                    lora_linear = LoRALinear(
                        in_features=child.weight.shape[1],
                        out_features=child.weight.shape[0],
                        r=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    # Initialize with pretrained weights
                    lora_linear.linear.weight = child.weight
                    if child.bias is not None:
                        lora_linear.linear.bias = child.bias
                    setattr(module, name, lora_linear)
                else:
                    replace_with_lora(child)
                    
        replace_with_lora(self.model)
        print(f"Applied LoRA with rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        
    def load_dataset(self):
        """Load and prepare dataset."""
        dataset_path = Path(self.config.dataset_path)
        
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
            
        # Split into train/eval
        split_idx = int(len(data) * 0.9)
        self.train_dataset = data[:split_idx]
        self.eval_dataset = data[split_idx:]
        
        print(f"Loaded dataset: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval")
        return self.train_dataset, self.eval_dataset
        
    def create_optimizer(self):
        """Create optimizer based on config."""
        trainable_params = self.model.trainable_parameters()
        
        if self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                learning_rate=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "lion":
            self.optimizer = optim.Lion(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
            
        print(f"Created {self.config.optimizer_type} optimizer with lr={self.config.learning_rate}")
        return self.optimizer
        
    def prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch for training."""
        # Tokenize inputs
        texts = []
        for item in batch:
            if "instruction" in item and "output" in item:
                # Alpaca format
                text = f"### Instruction: {item['instruction']}\n"
                if item.get("input"):
                    text += f"### Input: {item['input']}\n"
                text += f"### Response: {item['output']}"
            elif "messages" in item:
                # ShareGPT format
                text = ""
                for msg in item["messages"]:
                    text += f"{msg['role']}: {msg['content']}\n"
            else:
                # Plain text
                text = item.get("text", "")
            texts.append(text)
            
        # Tokenize
        tokens = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        # Convert to MLX arrays
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # For language modeling
        }
        
    def compute_loss(self, batch: Dict[str, Any]) -> mx.array:
        """Compute training loss."""
        # Forward pass
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Compute cross-entropy loss
        logits = outputs.logits
        labels = batch["labels"]
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)
        
        # Cross entropy loss
        loss = nn.losses.cross_entropy(shift_logits, shift_labels)
        
        return loss
        
    def training_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        def loss_fn(model):
            outputs = model(batch["input_ids"])
            logits = outputs.logits
            labels = batch["labels"]
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            
            return nn.losses.cross_entropy(shift_logits, shift_labels)
        
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        
        # Clip gradients
        grads = tree_flatten(grads)
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in grads[0]))
        if grad_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / grad_norm
            grads = [(g * scale for g in grads[0]), grads[1]]
        grads = tree_unflatten(grads)
        
        # Update weights
        self.optimizer.update(self.model, grads)
        
        # Evaluate to get the loss value
        mx.eval(loss)
        
        return loss.item()
        
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(self.eval_dataset), self.config.batch_size):
            batch_data = self.eval_dataset[i:i + self.config.batch_size]
            batch = self.prepare_batch(batch_data)
            
            with mx.no_grad():
                loss = self.compute_loss(batch)
                mx.eval(loss)
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
        
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = f"{self.config.output_dir}/checkpoint-{self.step}"
            
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights = dict(tree_flatten(self.model.parameters()))
        mx.savez(f"{path}/model.npz", **weights)
        
        # Save optimizer state
        opt_state = dict(tree_flatten(self.optimizer.state))
        mx.savez(f"{path}/optimizer.npz", **opt_state)
        
        # Save config and training state
        state = {
            "step": self.step,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "training_history": self.training_history
        }
        with open(f"{path}/training_state.json", "w") as f:
            json.dump(state, f, indent=2)
            
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # Load model weights
        weights = mx.load(f"{path}/model.npz")
        self.model.update(tree_unflatten(list(weights.items())))
        
        # Load optimizer state
        opt_state = mx.load(f"{path}/optimizer.npz")
        self.optimizer.state.update(tree_unflatten(list(opt_state.items())))
        
        # Load training state
        with open(f"{path}/training_state.json", "r") as f:
            state = json.load(f)
            self.step = state["step"]
            self.best_loss = state["best_loss"]
            self.training_history = state["training_history"]
            
        print(f"Loaded checkpoint from {path}")
        
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print("Starting training...")
        self.model.train()
        
        total_steps = len(self.train_dataset) // self.config.batch_size * self.config.epochs
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle dataset
            indices = np.random.permutation(len(self.train_dataset))
            
            for i in range(0, len(self.train_dataset), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_data = [self.train_dataset[idx] for idx in batch_indices]
                batch = self.prepare_batch(batch_data)
                
                # Training step
                loss = self.training_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.step += 1
                
                # Logging
                if self.step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Step {self.step}/{total_steps} | Loss: {avg_loss:.4f}")
                    self.training_history.append({
                        "step": self.step,
                        "loss": avg_loss,
                        "epoch": epoch,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                # Evaluation
                if self.step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"Eval loss: {eval_loss:.4f}")
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(f"{self.config.output_dir}/best_model")
                        
                # Save checkpoint
                if self.step % self.config.save_steps == 0:
                    self.save_checkpoint()
                    
        # Final save
        self.save_checkpoint(f"{self.config.output_dir}/final_model")
        
        return {
            "final_loss": epoch_loss / num_batches,
            "best_loss": self.best_loss,
            "total_steps": self.step,
            "training_history": self.training_history
        }
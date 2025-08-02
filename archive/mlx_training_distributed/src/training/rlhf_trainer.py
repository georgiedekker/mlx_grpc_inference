#!/usr/bin/env python3
"""
RLHF (Reinforcement Learning from Human Feedback) implementation for MLX
Includes DPO, PPO, and Reward Model training
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
from collections import deque

@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    method: str = "dpo"  # dpo, ppo, reward_model
    beta: float = 0.1  # DPO temperature
    learning_rate: float = 5e-7
    epochs: int = 1
    batch_size: int = 2
    max_seq_length: int = 512
    preference_dataset: str = ""
    reward_model_path: Optional[str] = None
    
    # PPO specific
    ppo_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # DPO specific
    label_smoothing: float = 0.0
    reference_free: bool = False
    

class RewardModel(nn.Module):
    """Reward model for RLHF."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass to compute rewards."""
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Pool over sequence (use last token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            sum_hidden = mx.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = mx.sum(mask_expanded, axis=1)
            pooled = sum_hidden / (sum_mask + 1e-8)
        else:
            # Use last token
            pooled = hidden_states[:, -1, :]
            
        # Compute reward
        reward = self.reward_head(pooled).squeeze(-1)
        return reward
        

class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, base_trainer, config: RLHFConfig):
        self.base_trainer = base_trainer
        self.config = config
        self.reference_model = None
        
    def setup(self):
        """Setup DPO training."""
        # Load model
        self.base_trainer.load_model()
        
        # Create reference model (frozen copy)
        if not self.config.reference_free:
            print("Creating reference model...")
            from copy import deepcopy
            self.reference_model = deepcopy(self.base_trainer.model)
            self.reference_model.eval()
            
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
                
        # Load preference dataset
        self.load_preference_dataset()
        
        # Create optimizer
        self.base_trainer.create_optimizer()
        
    def load_preference_dataset(self):
        """Load preference dataset."""
        dataset_path = Path(self.config.preference_dataset)
        
        with open(dataset_path, 'r') as f:
            if dataset_path.suffix == '.json':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f]
                
        # Expected format: {"prompt": str, "chosen": str, "rejected": str}
        self.preference_data = data
        print(f"Loaded {len(self.preference_data)} preference pairs")
        
    def compute_dpo_loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute DPO loss."""
        # Get model outputs for chosen and rejected
        chosen_logits = self.base_trainer.model(batch["chosen_ids"], attention_mask=batch["chosen_mask"]).logits
        rejected_logits = self.base_trainer.model(batch["rejected_ids"], attention_mask=batch["rejected_mask"]).logits
        
        # Get log probabilities
        chosen_logprobs = self._get_logprobs(chosen_logits, batch["chosen_ids"], batch["chosen_mask"])
        rejected_logprobs = self._get_logprobs(rejected_logits, batch["rejected_ids"], batch["rejected_mask"])
        
        # Get reference model log probabilities if not reference-free
        if not self.config.reference_free and self.reference_model is not None:
            with mx.no_grad():
                ref_chosen_logits = self.reference_model(batch["chosen_ids"], attention_mask=batch["chosen_mask"]).logits
                ref_rejected_logits = self.reference_model(batch["rejected_ids"], attention_mask=batch["rejected_mask"]).logits
                
                ref_chosen_logprobs = self._get_logprobs(ref_chosen_logits, batch["chosen_ids"], batch["chosen_mask"])
                ref_rejected_logprobs = self._get_logprobs(ref_rejected_logits, batch["rejected_ids"], batch["rejected_mask"])
        else:
            ref_chosen_logprobs = mlx.zeros_like(chosen_logprobs)
            ref_rejected_logprobs = mlx.zeros_like(rejected_logprobs)
            
        # Compute DPO loss
        chosen_rewards = self.config.beta * (chosen_logprobs - ref_chosen_logprobs)
        rejected_rewards = self.config.beta * (rejected_logprobs - ref_rejected_logprobs)
        
        # Label smoothing
        if self.config.label_smoothing > 0:
            chosen_rewards = chosen_rewards * (1 - self.config.label_smoothing) + rejected_rewards * self.config.label_smoothing
            rejected_rewards = rejected_rewards * (1 - self.config.label_smoothing) + chosen_rewards * self.config.label_smoothing
            
        # Binary cross-entropy style loss
        loss = -mlx.log_sigmoid(chosen_rewards - rejected_rewards)
        
        return mx.mean(loss)
        
    def _get_logprobs(self, logits: mx.array, labels: mx.array, mask: mx.array) -> mx.array:
        """Get log probabilities of labels under the model."""
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_mask = mask[..., 1:]
        
        # Get log probabilities
        log_probs = nn.log_softmax(shift_logits, axis=-1)
        
        # Gather log probs of actual tokens
        batch_size, seq_len = shift_labels.shape
        gathered_log_probs = mlx.take_along_axis(
            log_probs,
            shift_labels.reshape(batch_size, seq_len, 1),
            axis=-1
        ).squeeze(-1)
        
        # Apply mask and sum
        masked_log_probs = gathered_log_probs * shift_mask
        return mx.sum(masked_log_probs, axis=1) / (mx.sum(shift_mask, axis=1) + 1e-8)
        
    def prepare_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Prepare preference batch."""
        prompts = []
        chosen = []
        rejected = []
        
        for item in batch_data:
            prompts.append(item["prompt"])
            chosen.append(item["prompt"] + " " + item["chosen"])
            rejected.append(item["prompt"] + " " + item["rejected"])
            
        # Tokenize
        chosen_tokens = self.base_trainer.tokenizer(
            chosen,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        rejected_tokens = self.base_trainer.tokenizer(
            rejected,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        return {
            "chosen_ids": mx.array(chosen_tokens["input_ids"]),
            "chosen_mask": mx.array(chosen_tokens["attention_mask"]),
            "rejected_ids": mx.array(rejected_tokens["input_ids"]),
            "rejected_mask": mx.array(rejected_tokens["attention_mask"])
        }
        
    def train(self):
        """DPO training loop."""
        print("Starting DPO training...")
        self.setup()
        
        self.base_trainer.model.train()
        history = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(self.preference_data))
            
            for i in range(0, len(self.preference_data), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_data = [self.preference_data[idx] for idx in batch_indices]
                batch = self.prepare_batch(batch_data)
                
                # Compute loss and gradients
                def loss_fn(model):
                    self.base_trainer.model = model
                    return self.compute_dpo_loss(batch)
                    
                loss, grads = mx.value_and_grad(loss_fn)(self.base_trainer.model)
                
                # Update weights
                self.base_trainer.optimizer.update(self.base_trainer.model, grads)
                
                mx.eval(loss)
                epoch_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Epoch {epoch+1} | Batch {num_batches} | Loss: {avg_loss:.4f}")
                    
            # Save checkpoint
            if (epoch + 1) % self.base_trainer.config.save_epochs == 0:
                self.base_trainer.save_checkpoint(f"{self.base_trainer.config.output_dir}/dpo_epoch_{epoch+1}")
                
        return {"final_loss": epoch_loss / num_batches, "history": history}
        

class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(self, base_trainer, config: RLHFConfig):
        self.base_trainer = base_trainer
        self.config = config
        self.reward_model = None
        self.value_model = None
        self.reference_model = None
        
    def setup(self):
        """Setup PPO training."""
        # Load model (policy)
        self.base_trainer.load_model()
        
        # Load reward model
        if self.config.reward_model_path:
            print(f"Loading reward model from {self.config.reward_model_path}")
            self.load_reward_model(self.config.reward_model_path)
        else:
            print("Creating new reward model")
            self.reward_model = RewardModel(self.base_trainer.model)
            
        # Create value network (shares base with policy)
        self.value_head = nn.Linear(self.base_trainer.model.config.hidden_size, 1)
        
        # Create reference model
        from copy import deepcopy
        self.reference_model = deepcopy(self.base_trainer.model)
        self.reference_model.eval()
        
        # Freeze reference
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Create optimizer
        self.base_trainer.create_optimizer()
        
    def compute_rewards(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Compute rewards using reward model."""
        with mx.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
        
    def compute_advantages(self, rewards: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute advantages using GAE."""
        advantages = []
        returns = []
        
        # Convert to numpy for easier manipulation
        rewards_np = np.array(rewards)
        values_np = np.array(values)
        
        for r, v in zip(rewards_np, values_np):
            # Compute returns and advantages for each sequence
            seq_advantages = []
            seq_returns = []
            
            gae = 0
            for t in reversed(range(len(r))):
                if t == len(r) - 1:
                    next_value = 0
                else:
                    next_value = v[t + 1]
                    
                delta = r[t] + self.config.gamma * next_value - v[t]
                gae = delta + self.config.gamma * self.config.lam * gae
                
                seq_advantages.insert(0, gae)
                seq_returns.insert(0, gae + v[t])
                
            advantages.append(seq_advantages)
            returns.append(seq_returns)
            
        return mx.array(advantages), mx.array(returns)
        
    def ppo_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single PPO update step."""
        # Get old log probabilities
        with mx.no_grad():
            old_logits = self.base_trainer.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            old_logprobs = self._get_action_logprobs(old_logits, batch["actions"], batch["action_mask"])
            
            # Compute values
            hidden_states = self.base_trainer.model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            ).hidden_states[-1]
            old_values = self.value_head(hidden_states).squeeze(-1)
            
        # Compute rewards
        rewards = self.compute_rewards(batch["input_ids"], batch["attention_mask"])
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, old_values)
        
        # Normalize advantages
        advantages = (advantages - mx.mean(advantages)) / (mlx.std(advantages) + 1e-8)
        
        # PPO epochs
        total_loss = 0
        for _ in range(self.config.ppo_epochs):
            # Get current log probabilities
            logits = self.base_trainer.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            logprobs = self._get_action_logprobs(logits, batch["actions"], batch["action_mask"])
            
            # Compute ratio
            ratio = mx.exp(logprobs - old_logprobs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = mx.clip(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
            policy_loss = -mx.mean(mx.minimum(surr1, surr2))
            
            # Value loss
            hidden_states = self.base_trainer.model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            ).hidden_states[-1]
            values = self.value_head(hidden_states).squeeze(-1)
            value_loss = mx.mean((values - returns) ** 2)
            
            # Entropy bonus
            entropy = self._compute_entropy(logits, batch["action_mask"])
            
            # Total loss
            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            # Compute gradients and update
            grads = mx.grad(lambda: loss)(self.base_trainer.model)
            self.base_trainer.optimizer.update(self.base_trainer.model, grads)
            
            total_loss += loss.item()
            
        return {
            "total_loss": total_loss / self.config.ppo_epochs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }
        
    def _get_action_logprobs(self, logits: mx.array, actions: mx.array, mask: mx.array) -> mx.array:
        """Get log probabilities of actions."""
        log_probs = nn.log_softmax(logits, axis=-1)
        
        # Gather log probs of actions
        batch_size, seq_len = actions.shape
        action_logprobs = mlx.take_along_axis(
            log_probs,
            actions.reshape(batch_size, seq_len, 1),
            axis=-1
        ).squeeze(-1)
        
        # Apply mask
        masked_logprobs = action_logprobs * mask
        return mx.sum(masked_logprobs, axis=1) / (mx.sum(mask, axis=1) + 1e-8)
        
    def _compute_entropy(self, logits: mx.array, mask: mx.array) -> mx.array:
        """Compute entropy of policy."""
        probs = nn.softmax(logits, axis=-1)
        log_probs = nn.log_softmax(logits, axis=-1)
        
        entropy = -mx.sum(probs * log_probs, axis=-1)
        masked_entropy = entropy * mask
        
        return mx.mean(mx.sum(masked_entropy, axis=1) / (mx.sum(mask, axis=1) + 1e-8))
        
    def train(self):
        """PPO training loop."""
        print("Starting PPO training...")
        self.setup()
        
        # Note: PPO typically requires online data collection
        # This is a simplified version
        print("Warning: Full PPO requires online data collection. This is a simplified implementation.")
        
        return {"status": "PPO training requires environment interaction"}
        

class RewardModelTrainer:
    """Trainer for reward models."""
    
    def __init__(self, base_trainer, config: RLHFConfig):
        self.base_trainer = base_trainer
        self.config = config
        
    def setup(self):
        """Setup reward model training."""
        # Load base model
        self.base_trainer.load_model()
        
        # Create reward model
        self.reward_model = RewardModel(self.base_trainer.model)
        self.base_trainer.model = self.reward_model
        
        # Load preference dataset
        self.load_preference_dataset()
        
        # Create optimizer
        self.base_trainer.create_optimizer()
        
    def load_preference_dataset(self):
        """Load preference dataset for reward model training."""
        dataset_path = Path(self.config.preference_dataset)
        
        with open(dataset_path, 'r') as f:
            if dataset_path.suffix == '.json':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f]
                
        self.preference_data = data
        print(f"Loaded {len(self.preference_data)} preference pairs for reward model training")
        
    def compute_reward_loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute loss for reward model training."""
        # Get rewards for chosen and rejected
        chosen_rewards = self.reward_model(batch["chosen_ids"], batch["chosen_mask"])
        rejected_rewards = self.reward_model(batch["rejected_ids"], batch["rejected_mask"])
        
        # Pairwise ranking loss
        loss = -mlx.log_sigmoid(chosen_rewards - rejected_rewards)
        
        return mx.mean(loss)
        
    def train(self):
        """Train reward model."""
        print("Starting reward model training...")
        self.setup()
        
        history = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(self.preference_data))
            
            for i in range(0, len(self.preference_data), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_data = [self.preference_data[idx] for idx in batch_indices]
                
                # Prepare batch (same as DPO)
                batch = self.prepare_preference_batch(batch_data)
                
                # Compute loss and gradients
                def loss_fn(model):
                    self.reward_model = model
                    return self.compute_reward_loss(batch)
                    
                loss, grads = mx.value_and_grad(loss_fn)(self.reward_model)
                
                # Update weights
                self.base_trainer.optimizer.update(self.reward_model, grads)
                
                mx.eval(loss)
                epoch_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"Epoch {epoch+1} | Batch {num_batches} | Loss: {avg_loss:.4f}")
                    
            # Save checkpoint
            self.save_reward_model(f"{self.base_trainer.config.output_dir}/reward_model_epoch_{epoch+1}")
            
        return {"final_loss": epoch_loss / num_batches, "history": history}
        
    def save_reward_model(self, path: str):
        """Save reward model."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights = dict(tree_flatten(self.reward_model.parameters()))
        mx.savez(f"{path}/reward_model.npz", **weights)
        
        # Save config
        config_data = {
            "base_model": self.base_trainer.config.model_name,
            "reward_model_config": self.config.__dict__
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config_data, f, indent=2)
            
        print(f"Saved reward model to {path}")
        
    def prepare_preference_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Prepare batch for reward model training."""
        chosen = []
        rejected = []
        
        for item in batch_data:
            chosen.append(item["prompt"] + " " + item["chosen"])
            rejected.append(item["prompt"] + " " + item["rejected"])
            
        # Tokenize
        chosen_tokens = self.base_trainer.tokenizer(
            chosen,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        rejected_tokens = self.base_trainer.tokenizer(
            rejected,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        return {
            "chosen_ids": mx.array(chosen_tokens["input_ids"]),
            "chosen_mask": mx.array(chosen_tokens["attention_mask"]),
            "rejected_ids": mx.array(rejected_tokens["input_ids"]),
            "rejected_mask": mx.array(rejected_tokens["attention_mask"])
        }


def create_rlhf_trainer(base_trainer, config: RLHFConfig):
    """Factory function to create appropriate RLHF trainer."""
    if config.method == "dpo":
        return DPOTrainer(base_trainer, config)
    elif config.method == "ppo":
        return PPOTrainer(base_trainer, config)
    elif config.method == "reward_model":
        return RewardModelTrainer(base_trainer, config)
    else:
        raise ValueError(f"Unknown RLHF method: {config.method}")
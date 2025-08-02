#!/usr/bin/env python3
"""
Advanced optimizer implementations for MLX
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

class Lion(optim.Optimizer):
    """
    Lion optimizer - Symbolic Discovery of Optimization Algorithms
    More memory efficient than Adam, often achieves better performance
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
    def init_single(self, parameter: mx.array) -> Dict[str, Any]:
        """Initialize state for a single parameter."""
        return {"m": mlx.zeros_like(parameter)}
        
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Apply Lion update to a single parameter."""
        m = state["m"]
        
        # Update biased first moment estimate
        update = mlx.sign(m * self.beta1 + gradient * (1 - self.beta1))
        
        # Apply weight decay
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)
            
        # Update parameter
        parameter = parameter - self.learning_rate * update
        
        # Update momentum
        m = m * self.beta2 + gradient * (1 - self.beta2)
        
        return parameter, {"m": m}


class Adafactor(optim.Optimizer):
    """
    Adafactor optimizer - Memory efficient adaptive learning rate optimizer
    Reduces memory usage compared to Adam by not storing full second moments
    """
    
    def __init__(
        self,
        learning_rate: float = None,
        eps: tuple = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self._step = 0
        
    def init_single(self, parameter: mx.array) -> Dict[str, Any]:
        """Initialize state for a single parameter."""
        state = {"step": 0}
        
        # Factored second moments for memory efficiency
        if len(parameter.shape) >= 2:
            # Use factored representation for matrices
            state["v_row"] = mx.zeros(parameter.shape[0])
            state["v_col"] = mx.zeros(parameter.shape[1])
        else:
            # Use standard representation for vectors
            state["v"] = mlx.zeros_like(parameter)
            
        if self.beta1 is not None:
            state["m"] = mlx.zeros_like(parameter)
            
        return state
        
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Apply Adafactor update to a single parameter."""
        state["step"] += 1
        step = state["step"]
        
        # Compute learning rate
        if self.relative_step:
            min_step = 1e-6 * step if self.warmup_init else 1.0
            rel_step_sz = min(min_step, 1.0 / np.sqrt(step))
            param_scale = 1.0
            if self.scale_parameter:
                param_scale = np.sqrt(mx.mean(parameter * parameter).item())
            lr = rel_step_sz * param_scale
        else:
            lr = self.learning_rate
            
        # Gradient clipping
        grad_norm = mx.sqrt(mx.mean(gradient * gradient))
        clipped_gradient = gradient * mx.minimum(1.0, self.clip_threshold / grad_norm)
        
        # Update second moment estimate
        if len(parameter.shape) >= 2:
            # Factored second moment
            v_row = state["v_row"]
            v_col = state["v_col"]
            
            # Update row and column statistics
            row_mean = mx.mean(clipped_gradient * clipped_gradient, axis=1)
            col_mean = mx.mean(clipped_gradient * clipped_gradient, axis=0)
            
            # Exponential moving average
            beta2 = 1.0 - (step + 1) ** self.decay_rate
            v_row = beta2 * v_row + (1 - beta2) * row_mean
            v_col = beta2 * v_col + (1 - beta2) * col_mean
            
            state["v_row"] = v_row
            state["v_col"] = v_col
            
            # Reconstruct second moment
            v = mlx.outer(v_row, v_col) / mx.mean(v_col)
            v = v + self.eps[0]
            
        else:
            # Standard second moment for vectors
            v = state["v"]
            beta2 = 1.0 - (step + 1) ** self.decay_rate
            v = beta2 * v + (1 - beta2) * clipped_gradient * clipped_gradient
            state["v"] = v
            
        # Update first moment estimate if using momentum
        if self.beta1 is not None:
            m = state["m"]
            m = self.beta1 * m + (1 - self.beta1) * clipped_gradient
            state["m"] = m
            update = m / (mx.sqrt(v) + self.eps[1])
        else:
            update = clipped_gradient / (mx.sqrt(v) + self.eps[1])
            
        # Apply weight decay
        if self.weight_decay > 0:
            parameter = parameter * (1 - lr * self.weight_decay)
            
        # Update parameter
        parameter = parameter - lr * update
        
        return parameter, state


class SAM(optim.Optimizer):
    """
    Sharpness Aware Minimization (SAM) optimizer
    Seeks parameters that lie in neighborhoods with uniformly low loss
    """
    
    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False
    ):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.first_step = True
        
    def init_single(self, parameter: mx.array) -> Dict[str, Any]:
        """Initialize state for a single parameter."""
        base_state = self.base_optimizer.init_single(parameter)
        return {
            "base_state": base_state,
            "old_p": mlx.zeros_like(parameter)
        }
        
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Apply SAM update to a single parameter."""
        if self.first_step:
            # First step: compute gradient at perturbed point
            old_p = parameter.copy()
            state["old_p"] = old_p
            
            # Compute epsilon
            if self.adaptive:
                # Adaptive SAM
                grad_norm = mx.sqrt(mx.sum(gradient * gradient))
                scale = self.rho / (grad_norm + 1e-12)
                epsilon = scale * gradient
            else:
                # Original SAM
                param_norm = mx.sqrt(mx.sum(parameter * parameter))
                epsilon = self.rho * gradient / (mx.sqrt(mx.sum(gradient * gradient)) + 1e-12)
                epsilon = epsilon * param_norm
                
            # Move to worst-case point
            parameter = parameter + epsilon
            
        else:
            # Second step: update using gradient at perturbed point
            parameter = state["old_p"]
            
            # Apply base optimizer update
            parameter, state["base_state"] = self.base_optimizer.apply_single(
                gradient, parameter, state["base_state"]
            )
            
        self.first_step = not self.first_step
        return parameter, state


class LAMB(optim.Optimizer):
    """
    Layer-wise Adaptive Moments optimizer for Batch training (LAMB)
    Particularly effective for large batch training
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        adapt_lr: bool = True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adapt_lr = adapt_lr
        
    def init_single(self, parameter: mx.array) -> Dict[str, Any]:
        """Initialize state for a single parameter."""
        return {
            "m": mlx.zeros_like(parameter),
            "v": mlx.zeros_like(parameter),
            "step": 0
        }
        
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Apply LAMB update to a single parameter."""
        state["step"] += 1
        step = state["step"]
        
        # Exponential moving averages
        m = state["m"]
        v = state["v"]
        
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * gradient * gradient
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** step)
        v_hat = v / (1 - self.beta2 ** step)
        
        # Update with weight decay
        update = m_hat / (mx.sqrt(v_hat) + self.eps) + self.weight_decay * parameter
        
        # Layer adaptation
        if self.adapt_lr:
            param_norm = mx.sqrt(mx.sum(parameter * parameter))
            update_norm = mx.sqrt(mx.sum(update * update))
            
            # Compute adaptive learning rate
            trust_ratio = mlx.where(
                param_norm > 0,
                mlx.where(update_norm > 0, param_norm / update_norm, 1.0),
                1.0
            )
            
            lr = self.learning_rate * trust_ratio
        else:
            lr = self.learning_rate
            
        # Update parameter
        parameter = parameter - lr * update
        
        state["m"] = m
        state["v"] = v
        
        return parameter, state


class NovoGrad(optim.Optimizer):
    """
    NovoGrad optimizer - Normalized gradient with adaptive learning rate
    Performs better than Adam in some NLP tasks
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: tuple = (0.95, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_averaging: bool = False
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_averaging = grad_averaging
        
    def init_single(self, parameter: mx.array) -> Dict[str, Any]:
        """Initialize state for a single parameter."""
        return {
            "m": mlx.zeros_like(parameter),
            "v": mx.array(0.0),
            "step": 0
        }
        
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Apply NovoGrad update to a single parameter."""
        state["step"] += 1
        step = state["step"]
        
        # Compute gradient norm
        grad_norm = mx.sqrt(mx.sum(gradient * gradient))
        
        # Exponential moving average of gradient norm
        v = state["v"]
        v = self.beta2 * v + (1 - self.beta2) * grad_norm
        state["v"] = v
        
        # Normalize gradient
        normalized_grad = gradient / (grad_norm + self.eps)
        
        # Compute first moment
        m = state["m"]
        if self.grad_averaging:
            # Gradient averaging
            m = self.beta1 * m + normalized_grad
        else:
            # Standard momentum
            m = self.beta1 * m + (1 - self.beta1) * normalized_grad
            
        state["m"] = m
        
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** step if not self.grad_averaging else 1.0
        bias_correction2 = 1 - self.beta2 ** step
        
        # Apply weight decay
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)
            
        # Update parameter
        step_size = self.learning_rate / (bias_correction2 ** 0.5)
        parameter = parameter - step_size * m / (v / bias_correction2 + self.eps) / bias_correction1
        
        return parameter, state


def create_optimizer(optimizer_type: str, **kwargs) -> optim.Optimizer:
    """Factory function to create optimizers."""
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "lion": Lion,
        "adafactor": Adafactor,
        "lamb": LAMB,
        "novograd": NovoGrad
    }
    
    if optimizer_type.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    return optimizers[optimizer_type.lower()](**kwargs)
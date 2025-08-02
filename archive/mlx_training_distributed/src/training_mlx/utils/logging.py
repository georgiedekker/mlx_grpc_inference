"""
Logging utilities for Training MLX
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration."""
    
    # Create logger
    logger = logging.getLogger("training_mlx")
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set log level from environment
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if hasattr(logging, env_level):
        logger.setLevel(getattr(logging, env_level))
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"training_mlx.{name}")


class TrainingLogger:
    """Specialized logger for training metrics."""
    
    def __init__(self, experiment_name: str, log_dir: str = "./logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logging(
            level=logging.INFO,
            log_file=str(log_file)
        )
        
        # Metrics log file (CSV format)
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.csv"
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize metrics CSV file."""
        with open(self.metrics_file, 'w') as f:
            f.write("timestamp,epoch,step,loss,learning_rate,grad_norm\n")
    
    def log_metrics(self, epoch: int, step: int, metrics: dict):
        """Log training metrics."""
        timestamp = datetime.now().isoformat()
        
        # Log to file
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{epoch},{step},")
            f.write(f"{metrics.get('loss', 0)},")
            f.write(f"{metrics.get('learning_rate', 0)},")
            f.write(f"{metrics.get('grad_norm', 0)}\n")
        
        # Log to console
        self.logger.info(
            f"Epoch {epoch}, Step {step}: "
            f"loss={metrics.get('loss', 0):.4f}, "
            f"lr={metrics.get('learning_rate', 0):.2e}"
        )
    
    def log_validation(self, epoch: int, metrics: dict):
        """Log validation metrics."""
        self.logger.info(
            f"Validation - Epoch {epoch}: "
            f"loss={metrics.get('loss', 0):.4f}, "
            f"perplexity={metrics.get('perplexity', 0):.2f}"
        )
    
    def log_checkpoint(self, path: str):
        """Log checkpoint save."""
        self.logger.info(f"Saved checkpoint: {path}")
    
    def log_completion(self, total_time: float):
        """Log training completion."""
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        
        self.logger.info(
            f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s"
        )
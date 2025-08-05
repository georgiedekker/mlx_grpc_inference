"""
Enhanced logging configuration for comprehensive diagnostics.
Provides structured logging with performance metrics and debugging info.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import functools
import mlx.core as mx
from contextlib import contextmanager

# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'
}

class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured logs with colors and metadata."""
    
    def format(self, record):
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add color for terminal
        if hasattr(record, 'color') and record.color:
            color = COLORS.get(record.levelname, COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{COLORS['RESET']}"
        
        # Format the base message
        formatted = super().format(record)
        
        # Add structured data if present
        if hasattr(record, 'data'):
            formatted += f"\n  📊 Data: {json.dumps(record.data, indent=2)}"
            
        return formatted

def setup_logging(name: str = None, level: str = "INFO", color: bool = True) -> logging.Logger:
    """
    Set up enhanced logging with structured output.
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level
        color: Whether to use color output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with structured formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Format pattern
    fmt_pattern = "%(timestamp)s [%(name)s] %(levelname)s: %(message)s"
    formatter = StructuredFormatter(fmt_pattern)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # Add color flag to all records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.color = color
        return record
    logging.setLogRecordFactory(record_factory)
    
    return logger

class DiagnosticLogger:
    """Enhanced logger with diagnostic methods for MLX inference."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = setup_logging(name, level)
        self.metrics = {}
        
    def log_tensor_info(self, tensor: mx.array, name: str, level: str = "DEBUG"):
        """Log detailed tensor information."""
        mx.eval(tensor)  # Ensure tensor is evaluated
        
        info = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(mx.min(tensor).item()),
            "max": float(mx.max(tensor).item()),
            "mean": float(mx.mean(tensor).item()),
            "std": float(mx.std(tensor).item()) if tensor.size > 1 else 0,
            "has_nan": bool(mx.any(mx.isnan(tensor)).item()),
            "has_inf": bool(mx.any(mx.isinf(tensor)).item())
        }
        
        # Check for value explosion
        if abs(info["max"]) > 1000 or abs(info["min"]) > 1000:
            level = "WARNING"
            info["alert"] = "⚠️ Large values detected - possible explosion"
        
        getattr(self.logger, level.lower())(f"Tensor '{name}':", extra={"data": info})
        return info
    
    def log_model_info(self, model: Any, tokenizer: Any):
        """Log model architecture and configuration."""
        info = {
            "model_type": str(type(model)),
            "has_layers": hasattr(model, 'layers') or (hasattr(model, 'model') and hasattr(model.model, 'layers')),
            "tokenizer_type": str(type(tokenizer)),
            "vocab_size": len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "unknown"
        }
        
        # Count layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            info["num_layers"] = len(model.model.layers)
        elif hasattr(model, 'layers'):
            info["num_layers"] = len(model.layers)
            
        self.logger.info("Model loaded:", extra={"data": info})
        return info
    
    def log_grpc_message(self, direction: str, service: str, method: str, 
                        size_bytes: int, duration_ms: Optional[float] = None):
        """Log gRPC message with size and timing."""
        info = {
            "direction": direction,  # "send" or "recv"
            "service": service,
            "method": method,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "size_bytes": size_bytes
        }
        
        if duration_ms:
            info["duration_ms"] = round(duration_ms, 2)
            info["throughput_mbps"] = round((size_bytes * 8) / (duration_ms * 1000), 2)
        
        icon = "📤" if direction == "send" else "📥"
        self.logger.info(f"{icon} gRPC {direction}:", extra={"data": info})
        return info
    
    def log_performance_metric(self, operation: str, duration_ms: float, 
                              tokens: Optional[int] = None, **kwargs):
        """Log performance metrics."""
        info = {
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "duration_s": round(duration_ms / 1000, 2)
        }
        
        if tokens:
            info["tokens"] = tokens
            info["tokens_per_second"] = round(tokens / (duration_ms / 1000), 2)
            
        # Add any extra metrics
        info.update(kwargs)
        
        self.logger.info(f"⏱️ Performance - {operation}:", extra={"data": info})
        
        # Store for aggregation
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(info)
        
        return info
    
    def log_memory_usage(self):
        """Log current memory usage."""
        import psutil
        process = psutil.Process()
        
        info = {
            "process_memory_gb": round(process.memory_info().rss / (1024**3), 2),
            "process_memory_percent": round(process.memory_percent(), 2),
            "system_memory_percent": round(psutil.virtual_memory().percent, 2),
            "system_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
        
        # MLX specific memory if available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                info["gpu_memory_used_gb"] = round(used / 1024, 2)
                info["gpu_memory_total_gb"] = round(total / 1024, 2)
                info["gpu_memory_percent"] = round(used / total * 100, 2)
        except:
            pass  # Not NVIDIA GPU
            
        self.logger.info("💾 Memory usage:", extra={"data": info})
        return info
    
    def log_worker_status(self, worker_id: str, status: str, 
                         layers: Optional[list] = None, error: Optional[str] = None):
        """Log worker connection status."""
        info = {
            "worker_id": worker_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if layers:
            info["layers"] = layers
            info["num_layers"] = len(layers)
            
        if error:
            info["error"] = error
            
        icon = "✅" if status == "connected" else "❌"
        level = "INFO" if status == "connected" else "ERROR"
        
        getattr(self.logger, level.lower())(f"{icon} Worker {worker_id}:", extra={"data": info})
        return info
    
    @contextmanager
    def log_operation(self, operation: str):
        """Context manager to time and log an operation."""
        start_time = time.time()
        self.logger.info(f"🚀 Starting: {operation}")
        
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self.log_performance_metric(operation, duration_ms, success=True)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"❌ Failed: {operation} after {duration_ms:.1f}ms - {str(e)}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}
        
        for operation, metrics in self.metrics.items():
            if metrics:
                durations = [m["duration_ms"] for m in metrics]
                summary[operation] = {
                    "count": len(metrics),
                    "avg_ms": round(sum(durations) / len(durations), 2),
                    "min_ms": round(min(durations), 2),
                    "max_ms": round(max(durations), 2)
                }
                
                # Add token metrics if available
                if "tokens_per_second" in metrics[0]:
                    tps_values = [m["tokens_per_second"] for m in metrics]
                    summary[operation]["avg_tokens_per_second"] = round(sum(tps_values) / len(tps_values), 2)
        
        return summary
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        if summary:
            self.logger.info("📈 Performance Summary:", extra={"data": summary})

# Decorator for automatic performance logging
def log_performance(logger: DiagnosticLogger, operation: str = None):
    """Decorator to automatically log function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            with logger.log_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Create a default diagnostic logger
default_logger = DiagnosticLogger("mlx_inference")
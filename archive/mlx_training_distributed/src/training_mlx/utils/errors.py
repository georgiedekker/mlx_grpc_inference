"""
Comprehensive Error Handling System for Distributed MLX Training

This module provides a complete error handling framework with:
- Custom exception hierarchy
- Error logging and monitoring
- Recovery mechanisms
- Graceful degradation strategies
"""

import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    CRITICAL = "critical"    # System-wide failures, immediate intervention required
    HIGH = "high"           # Major functionality broken, affects core operations
    MEDIUM = "medium"       # Partial functionality affected, workarounds available
    LOW = "low"            # Minor issues, no significant impact
    WARNING = "warning"     # Potential issues, monitoring recommended


class ErrorCategory(Enum):
    """Error categories for better organization and handling."""
    DISTRIBUTED_COMM = "distributed_communication"
    MODEL_LOADING = "model_loading"
    DATA_PROCESSING = "data_processing"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_id: str = ""
    category: ErrorCategory = ErrorCategory.TRAINING
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "additional_data": self.additional_data
        }


class BaseDistributedError(Exception):
    """Base exception class for all distributed system errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []
        self.traceback_str = traceback.format_exc()
        
    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        return self.message
        
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
            "recovery_suggestions": self.recovery_suggestions,
            "traceback": self.traceback_str
        }


class DistributedCommunicationError(BaseDistributedError):
    """Errors related to distributed communication failures."""
    
    def __init__(self, message: str, node_id: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DISTRIBUTED_COMM
        context.node_id = node_id
        
        # Add specific recovery suggestions
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Check network connectivity between nodes",
            "Verify firewall settings and port accessibility",
            "Restart distributed communication service",
            "Check if all nodes are running and responsive"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class ModelLoadingError(BaseDistributedError):
    """Errors related to model loading and initialization."""
    
    def __init__(self, message: str, model_path: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.MODEL_LOADING
        if model_path:
            context.additional_data['model_path'] = model_path
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Verify model file exists and is accessible",
            "Check available memory and disk space",
            "Ensure model format is compatible",
            "Try downloading the model again if from remote source"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class DataProcessingError(BaseDistributedError):
    """Errors related to data processing and validation."""
    
    def __init__(self, message: str, data_path: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DATA_PROCESSING
        if data_path:
            context.additional_data['data_path'] = data_path
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Validate data format and structure",
            "Check for corrupted or missing data files",
            "Verify data preprocessing pipeline",
            "Ensure sufficient memory for data loading"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class ResourceError(BaseDistributedError):
    """Errors related to resource constraints (memory, disk, compute)."""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.RESOURCE
        context.severity = ErrorSeverity.HIGH
        if resource_type:
            context.additional_data['resource_type'] = resource_type
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Check available system resources (RAM, disk space, GPU memory)",
            "Reduce batch size or model size if possible",
            "Close unnecessary applications to free resources",
            "Consider using gradient checkpointing to reduce memory usage"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class NetworkError(BaseDistributedError):
    """Errors related to network connectivity and timeouts."""
    
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.NETWORK
        if endpoint:
            context.additional_data['endpoint'] = endpoint
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Check internet connectivity",
            "Verify API endpoints are accessible",
            "Check for firewall or proxy issues",
            "Retry with exponential backoff"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class ConfigurationError(BaseDistributedError):
    """Errors related to configuration and setup issues."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.CONFIGURATION
        if config_key:
            context.additional_data['config_key'] = config_key
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Check configuration file syntax and values",
            "Verify all required configuration parameters are set",
            "Ensure configuration matches expected schema",
            "Review documentation for correct configuration format"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class TrainingError(BaseDistributedError):
    """Errors that occur during the training process."""
    
    def __init__(self, message: str, epoch: int = None, step: int = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.TRAINING
        if epoch is not None:
            context.additional_data['epoch'] = epoch
        if step is not None:
            context.additional_data['step'] = step
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Check training data quality and format",
            "Verify model architecture compatibility",
            "Adjust learning rate or other hyperparameters",
            "Resume from last checkpoint if available"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class InferenceError(BaseDistributedError):
    """Errors that occur during inference/generation."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.INFERENCE
        if model_name:
            context.additional_data['model_name'] = model_name
            
        recovery_suggestions = kwargs.get('recovery_suggestions', [])
        recovery_suggestions.extend([
            "Verify model is loaded and ready for inference",
            "Check input format and size constraints",
            "Ensure sufficient resources for inference",
            "Try with simpler inputs to isolate the issue"
        ])
        kwargs['recovery_suggestions'] = recovery_suggestions
        
        super().__init__(message, context=context, **kwargs)


class ErrorHandler:
    """Centralized error handling and logging system."""
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 log_file: Optional[str] = None,
                 enable_monitoring: bool = True):
        self.logger = logger or self._setup_logger(log_file)
        self.enable_monitoring = enable_monitoring
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
    def _setup_logger(self, log_file: Optional[str] = None) -> logging.Logger:
        """Setup dedicated error logger."""
        logger = logging.getLogger("distributed_error_handler")
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    def handle_error(self, 
                    error: BaseDistributedError,
                    context_updates: Optional[Dict[str, Any]] = None) -> None:
        """Handle and log error with full context."""
        
        # Update context if provided
        if context_updates:
            error.context.additional_data.update(context_updates)
            
        # Generate unique error ID
        error_id = f"{error.context.category.value}_{int(time.time() * 1000)}"
        error.context.error_id = error_id
        
        # Track error counts
        error_type = error.__class__.__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log based on severity
        log_data = {
            "error_id": error_id,
            "error_type": error_type,
            "message": error.message,
            "context": error.context.to_dict(),
            "debug_info": error.get_debug_info()
        }
        
        if error.context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=log_data)
        elif error.context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra=log_data)
        elif error.context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=log_data)
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra=log_data)
            
        # Optional: Send to monitoring system
        if self.enable_monitoring:
            self._send_to_monitoring(error, log_data)
            
    def _send_to_monitoring(self, 
                          error: BaseDistributedError, 
                          log_data: Dict[str, Any]) -> None:
        """Send error to monitoring system (placeholder for actual implementation)."""
        try:
            # This would integrate with actual monitoring systems like:
            # - Sentry
            # - DataDog
            # - New Relic
            # - Custom monitoring endpoints
            pass
        except Exception as e:
            self.logger.warning(f"Failed to send error to monitoring: {e}")
            
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self.error_counts.copy(),
            "recovery_attempts": self.recovery_attempts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }


class RetryManager:
    """Manages retry logic with exponential backoff and circuit breaker pattern."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        
    def execute_with_retry(self,
                          operation: Callable,
                          operation_id: str,
                          *args,
                          **kwargs) -> Any:
        """Execute operation with retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                # Reset failure count on success
                if operation_id in self.failure_counts:
                    del self.failure_counts[operation_id]
                return result
                
            except Exception as e:
                self.failure_counts[operation_id] = self.failure_counts.get(operation_id, 0) + 1
                self.last_failure_time[operation_id] = time.time()
                
                if attempt == self.max_retries:
                    # Convert to appropriate distributed error
                    if isinstance(e, BaseDistributedError):
                        raise e
                    else:
                        raise BaseDistributedError(
                            f"Operation {operation_id} failed after {self.max_retries} retries: {str(e)}",
                            cause=e
                        )
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                
                logging.getLogger(__name__).warning(
                    f"Operation {operation_id} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {str(e)}"
                )
                
                time.sleep(delay)
                
    def should_circuit_break(self, operation_id: str, threshold: int = 5, window: float = 300.0) -> bool:
        """Check if circuit breaker should be triggered."""
        if operation_id not in self.failure_counts:
            return False
            
        failure_count = self.failure_counts[operation_id]
        last_failure = self.last_failure_time.get(operation_id, 0)
        
        # Reset counter if outside time window
        if time.time() - last_failure > window:
            del self.failure_counts[operation_id]
            return False
            
        return failure_count >= threshold


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(error: BaseDistributedError, **context_updates) -> None:
    """Global error handling function."""
    global_error_handler.handle_error(error, context_updates)


def create_error_context(component: str, 
                        operation: str,
                        **additional_data) -> ErrorContext:
    """Convenience function to create error context."""
    context = ErrorContext(
        component=component,
        operation=operation,
        additional_data=additional_data
    )
    return context
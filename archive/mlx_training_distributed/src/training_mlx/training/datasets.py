"""
Dataset utilities for Training MLX

Supports Alpaca and ShareGPT formats with validation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger("datasets")


def detect_dataset_format(file_path: str) -> Dict[str, Any]:
    """Auto-detect dataset format."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Read first few lines to detect format
    with open(path, 'r') as f:
        first_line = f.readline().strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(first_line)
            
            # Check for Alpaca format
            if all(key in data for key in ["instruction", "output"]):
                return {"format": "alpaca", "type": "jsonl"}
            
            # Check for ShareGPT format  
            if "conversations" in data:
                return {"format": "sharegpt", "type": "jsonl"}
                
        except json.JSONDecodeError:
            # Try to load entire file as JSON array
            f.seek(0)
            try:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if all(key in data[0] for key in ["instruction", "output"]):
                        return {"format": "alpaca", "type": "json"}
                    elif "conversations" in data[0]:
                        return {"format": "sharegpt", "type": "json"}
            except:
                pass
    
    raise ValueError(f"Could not detect dataset format for: {file_path}")


def validate_dataset(
    file_path: str, 
    format: Optional[str] = None,
    sample_size: int = 100
) -> Dict[str, Any]:
    """Validate dataset file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Auto-detect format if not provided
    if not format:
        format_info = detect_dataset_format(file_path)
        format = format_info["format"]
        file_type = format_info["type"]
    else:
        file_type = "jsonl"  # Assume JSONL by default
    
    logger.info(f"Validating {format} dataset: {file_path}")
    
    # Validate based on format
    if format == "alpaca":
        return _validate_alpaca(path, file_type, sample_size)
    elif format == "sharegpt":
        return _validate_sharegpt(path, file_type, sample_size)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _validate_alpaca(path: Path, file_type: str, sample_size: int) -> Dict[str, Any]:
    """Validate Alpaca format dataset."""
    issues = []
    warnings = []
    num_samples = 0
    total_length = 0
    sample_data = []
    
    if file_type == "json":
        with open(path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                issues.append("Dataset must be a JSON array")
                return {"valid": False, "issues": issues}
    else:
        data = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON at line {len(data) + 1}: {e}")
    
    # Validate each sample
    for idx, sample in enumerate(data[:sample_size]):
        num_samples += 1
        
        # Check required fields
        if "instruction" not in sample:
            issues.append(f"Sample {idx}: missing 'instruction' field")
        if "output" not in sample:
            issues.append(f"Sample {idx}: missing 'output' field")
        
        # Check field types
        if not isinstance(sample.get("instruction", ""), str):
            issues.append(f"Sample {idx}: 'instruction' must be string")
        if not isinstance(sample.get("output", ""), str):
            issues.append(f"Sample {idx}: 'output' must be string")
        
        # Estimate length
        text_length = len(sample.get("instruction", "")) + len(sample.get("output", ""))
        total_length += text_length
        
        # Warnings for edge cases
        if text_length > 4096:
            warnings.append(f"Sample {idx}: very long text ({text_length} chars)")
        if text_length < 10:
            warnings.append(f"Sample {idx}: very short text ({text_length} chars)")
        
        # Collect sample
        if idx < 3:
            sample_data.append(sample)
    
    avg_length = total_length / max(num_samples, 1)
    
    return {
        "valid": len(issues) == 0,
        "format": "alpaca",
        "num_samples": len(data),
        "validated_samples": num_samples,
        "avg_length": avg_length,
        "issues": issues,
        "warnings": warnings,
        "sample": sample_data
    }


def _validate_sharegpt(path: Path, file_type: str, sample_size: int) -> Dict[str, Any]:
    """Validate ShareGPT format dataset."""
    issues = []
    warnings = []
    num_samples = 0
    total_length = 0
    sample_data = []
    
    if file_type == "json":
        with open(path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                issues.append("Dataset must be a JSON array")
                return {"valid": False, "issues": issues}
    else:
        data = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON at line {len(data) + 1}: {e}")
    
    # Validate each sample
    for idx, sample in enumerate(data[:sample_size]):
        num_samples += 1
        
        # Check required fields
        if "conversations" not in sample:
            issues.append(f"Sample {idx}: missing 'conversations' field")
            continue
        
        conversations = sample["conversations"]
        if not isinstance(conversations, list):
            issues.append(f"Sample {idx}: 'conversations' must be array")
            continue
        
        # Validate conversation structure
        text_length = 0
        for turn_idx, turn in enumerate(conversations):
            if "from" not in turn:
                issues.append(f"Sample {idx}, turn {turn_idx}: missing 'from' field")
            if "value" not in turn:
                issues.append(f"Sample {idx}, turn {turn_idx}: missing 'value' field")
            
            # Check valid roles
            if turn.get("from") not in ["human", "assistant", "user", "gpt"]:
                warnings.append(f"Sample {idx}, turn {turn_idx}: unusual role '{turn.get('from')}'")
            
            text_length += len(turn.get("value", ""))
        
        total_length += text_length
        
        # Warnings
        if len(conversations) < 2:
            warnings.append(f"Sample {idx}: only {len(conversations)} turns")
        if text_length > 8192:
            warnings.append(f"Sample {idx}: very long conversation ({text_length} chars)")
        
        # Collect sample
        if idx < 3:
            sample_data.append(sample)
    
    avg_length = total_length / max(num_samples, 1)
    
    return {
        "valid": len(issues) == 0,
        "format": "sharegpt",
        "num_samples": len(data),
        "validated_samples": num_samples,
        "avg_length": avg_length,
        "issues": issues,
        "warnings": warnings,
        "sample": sample_data
    }


class DatasetValidator:
    """High-level dataset validator."""
    
    def validate(
        self,
        file_path: str,
        format: str = "auto",
        split_ratio: float = 0.1,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate dataset with comprehensive checks."""
        
        # Auto-detect format if needed
        if format == "auto":
            format_info = detect_dataset_format(file_path)
            format = format_info["format"]
        
        # Run validation
        result = validate_dataset(
            file_path,
            format=format,
            sample_size=max_samples or 1000
        )
        
        # Add split information
        if result["valid"] and split_ratio > 0:
            total_samples = result["num_samples"]
            val_samples = int(total_samples * split_ratio)
            train_samples = total_samples - val_samples
            
            result["split_info"] = {
                "train_samples": train_samples,
                "validation_samples": val_samples,
                "split_ratio": split_ratio
            }
        
        return result
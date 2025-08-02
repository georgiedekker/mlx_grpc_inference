"""
Dataset Format Integration for MLX Training API

This module provides integration for Alpaca and ShareGPT dataset formats
into the existing training API.
"""

import json
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    format_type: str  # "alpaca", "sharegpt", or "unknown"
    total_samples: int
    errors: List[str]
    warnings: List[str]
    sample_preview: Optional[Dict[str, Any]] = None


def detect_dataset_format(file_path: str) -> str:
    """
    Detect the format of a dataset file.
    
    Returns: "alpaca", "sharegpt", or "unknown"
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to read first few lines
            first_line = f.readline().strip()
            
            if not first_line:
                return "unknown"
            
            # Try to parse as JSON
            try:
                if first_line.startswith('['):
                    # JSON array format
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return _detect_format_from_sample(data[0])
                else:
                    # JSONL format
                    sample = json.loads(first_line)
                    return _detect_format_from_sample(sample)
            except json.JSONDecodeError:
                return "unknown"
                
    except Exception as e:
        logger.error(f"Error detecting dataset format: {e}")
        return "unknown"


def _detect_format_from_sample(sample: Dict[str, Any]) -> str:
    """Detect format from a single sample."""
    # Check for Alpaca format
    alpaca_keys = {"instruction", "output"}
    alpaca_optional_keys = {"input", "system"}
    
    # Check for ShareGPT format
    sharegpt_keys = {"conversations"}
    sharegpt_alt_keys = {"messages", "dialogue", "chat", "turns"}
    
    sample_keys = set(sample.keys())
    
    # Check Alpaca format
    if alpaca_keys.issubset(sample_keys):
        return "alpaca"
    
    # Check ShareGPT format
    if sharegpt_keys.intersection(sample_keys) or sharegpt_alt_keys.intersection(sample_keys):
        # Verify it has conversation structure
        conv_key = None
        for key in ["conversations", "messages", "dialogue", "chat", "turns"]:
            if key in sample:
                conv_key = key
                break
        
        if conv_key and isinstance(sample[conv_key], list) and len(sample[conv_key]) > 0:
            # Check if it has role/content structure
            first_turn = sample[conv_key][0]
            if isinstance(first_turn, dict) and any(k in first_turn for k in ["from", "role"]):
                return "sharegpt"
    
    return "unknown"


def validate_alpaca_dataset(file_path: str, max_samples: int = None) -> DatasetValidationResult:
    """
    Validate an Alpaca format dataset.
    """
    errors = []
    warnings = []
    total_samples = 0
    valid_samples = 0
    sample_preview = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if JSON or JSONL
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON array format
                data = json.load(f)
                samples = data if max_samples is None else data[:max_samples]
            else:
                # JSONL format
                samples = []
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: Invalid JSON - {e}")
        
        total_samples = len(samples)
        
        for i, sample in enumerate(samples):
            sample_errors = []
            
            # Check required fields
            if "instruction" not in sample:
                sample_errors.append(f"Sample {i+1}: Missing 'instruction' field")
            elif not isinstance(sample["instruction"], str) or not sample["instruction"].strip():
                sample_errors.append(f"Sample {i+1}: 'instruction' must be non-empty string")
            
            if "output" not in sample:
                sample_errors.append(f"Sample {i+1}: Missing 'output' field")
            elif not isinstance(sample["output"], str) or not sample["output"].strip():
                sample_errors.append(f"Sample {i+1}: 'output' must be non-empty string")
            
            # Check optional fields
            if "input" in sample and not isinstance(sample["input"], str):
                sample_errors.append(f"Sample {i+1}: 'input' must be string")
            
            if "system" in sample and not isinstance(sample["system"], str):
                sample_errors.append(f"Sample {i+1}: 'system' must be string")
            
            # Warnings for quality
            if len(sample_errors) == 0:
                valid_samples += 1
                
                if i == 0:
                    sample_preview = sample
                
                # Quality warnings
                instruction_len = len(sample.get("instruction", "").split())
                output_len = len(sample.get("output", "").split())
                
                if instruction_len < 3:
                    warnings.append(f"Sample {i+1}: Very short instruction ({instruction_len} words)")
                if output_len < 3:
                    warnings.append(f"Sample {i+1}: Very short output ({output_len} words)")
                if instruction_len > 500:
                    warnings.append(f"Sample {i+1}: Very long instruction ({instruction_len} words)")
                if output_len > 1000:
                    warnings.append(f"Sample {i+1}: Very long output ({output_len} words)")
            
            errors.extend(sample_errors)
        
        is_valid = len(errors) == 0 and valid_samples > 0
        
        return DatasetValidationResult(
            is_valid=is_valid,
            format_type="alpaca",
            total_samples=total_samples,
            errors=errors[:10],  # Limit errors shown
            warnings=warnings[:10],  # Limit warnings shown
            sample_preview=sample_preview
        )
        
    except Exception as e:
        return DatasetValidationResult(
            is_valid=False,
            format_type="alpaca",
            total_samples=0,
            errors=[f"Failed to read dataset: {e}"],
            warnings=[]
        )


def validate_sharegpt_dataset(file_path: str, max_samples: int = None) -> DatasetValidationResult:
    """
    Validate a ShareGPT format dataset.
    """
    errors = []
    warnings = []
    total_samples = 0
    valid_samples = 0
    sample_preview = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if JSON or JSONL
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON array format
                data = json.load(f)
                samples = data if max_samples is None else data[:max_samples]
            else:
                # JSONL format
                samples = []
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: Invalid JSON - {e}")
        
        total_samples = len(samples)
        
        for i, sample in enumerate(samples):
            sample_errors = []
            
            # Find conversations key
            conv_key = None
            for key in ["conversations", "messages", "dialogue", "chat", "turns"]:
                if key in sample:
                    conv_key = key
                    break
            
            if not conv_key:
                sample_errors.append(f"Sample {i+1}: Missing conversations field")
                errors.extend(sample_errors)
                continue
            
            conversations = sample[conv_key]
            
            if not isinstance(conversations, list):
                sample_errors.append(f"Sample {i+1}: '{conv_key}' must be a list")
            elif len(conversations) < 2:
                sample_errors.append(f"Sample {i+1}: Conversation must have at least 2 turns")
            else:
                # Validate conversation structure
                has_human = False
                has_assistant = False
                
                for j, turn in enumerate(conversations):
                    if not isinstance(turn, dict):
                        sample_errors.append(f"Sample {i+1}, Turn {j+1}: Must be a dictionary")
                        continue
                    
                    # Check for role field
                    role_key = None
                    for rk in ["from", "role", "speaker"]:
                        if rk in turn:
                            role_key = rk
                            break
                    
                    if not role_key:
                        sample_errors.append(f"Sample {i+1}, Turn {j+1}: Missing role field")
                        continue
                    
                    # Check for content field
                    content_key = None
                    for ck in ["value", "content", "text", "message"]:
                        if ck in turn:
                            content_key = ck
                            break
                    
                    if not content_key:
                        sample_errors.append(f"Sample {i+1}, Turn {j+1}: Missing content field")
                        continue
                    
                    role = turn[role_key].lower()
                    content = turn[content_key]
                    
                    if role in ["human", "user"]:
                        has_human = True
                    elif role in ["assistant", "gpt"]:
                        has_assistant = True
                    
                    if not isinstance(content, str) or not content.strip():
                        sample_errors.append(f"Sample {i+1}, Turn {j+1}: Content must be non-empty string")
                
                if not has_human:
                    warnings.append(f"Sample {i+1}: No human/user turns found")
                if not has_assistant:
                    warnings.append(f"Sample {i+1}: No assistant/gpt turns found")
            
            if len(sample_errors) == 0:
                valid_samples += 1
                if i == 0:
                    sample_preview = sample
            
            errors.extend(sample_errors)
        
        is_valid = len(errors) == 0 and valid_samples > 0
        
        return DatasetValidationResult(
            is_valid=is_valid,
            format_type="sharegpt",
            total_samples=total_samples,
            errors=errors[:10],  # Limit errors shown
            warnings=warnings[:10],  # Limit warnings shown
            sample_preview=sample_preview
        )
        
    except Exception as e:
        return DatasetValidationResult(
            is_valid=False,
            format_type="sharegpt",
            total_samples=0,
            errors=[f"Failed to read dataset: {e}"],
            warnings=[]
        )


def validate_dataset(file_path: str) -> DatasetValidationResult:
    """
    Validate a dataset file and detect its format.
    """
    # Detect format
    format_type = detect_dataset_format(file_path)
    
    if format_type == "alpaca":
        return validate_alpaca_dataset(file_path)
    elif format_type == "sharegpt":
        return validate_sharegpt_dataset(file_path)
    else:
        return DatasetValidationResult(
            is_valid=False,
            format_type="unknown",
            total_samples=0,
            errors=["Unknown dataset format. Expected Alpaca or ShareGPT format."],
            warnings=[]
        )


def convert_sharegpt_to_alpaca(sharegpt_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single ShareGPT sample to Alpaca format.
    """
    # Find conversations
    conv_key = None
    for key in ["conversations", "messages", "dialogue", "chat", "turns"]:
        if key in sharegpt_sample:
            conv_key = key
            break
    
    if not conv_key:
        return None
    
    conversations = sharegpt_sample[conv_key]
    
    # Extract system message, first user input, and first assistant response
    system_message = ""
    user_messages = []
    assistant_messages = []
    
    for turn in conversations:
        # Find role and content
        role = None
        content = None
        
        for rk in ["from", "role", "speaker"]:
            if rk in turn:
                role = turn[rk].lower()
                break
        
        for ck in ["value", "content", "text", "message"]:
            if ck in turn:
                content = turn[ck]
                break
        
        if not role or not content:
            continue
        
        if role == "system":
            system_message = content
        elif role in ["human", "user"]:
            user_messages.append(content)
        elif role in ["assistant", "gpt"]:
            assistant_messages.append(content)
    
    if not user_messages or not assistant_messages:
        return None
    
    # Create Alpaca format
    alpaca_sample = {
        "instruction": user_messages[0],
        "output": assistant_messages[0]
    }
    
    # Add input if there are multiple user messages
    if len(user_messages) > 1:
        alpaca_sample["input"] = "\n\n".join(user_messages[1:])
    
    # Add system message if present
    if system_message:
        alpaca_sample["system"] = system_message
    
    return alpaca_sample


def create_dataset_loader_config(
    dataset_path: str,
    format_type: str,
    batch_size: int = 8,
    max_seq_length: int = 2048,
    shuffle: bool = True
) -> Dict[str, Any]:
    """
    Create configuration for dataset loading in the training API.
    """
    config = {
        "dataset_path": dataset_path,
        "format_type": format_type,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "shuffle": shuffle,
        "preprocessing": {
            "tokenizer": "auto",  # Use model's tokenizer
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "mlx"
        }
    }
    
    if format_type == "alpaca":
        config["format_config"] = {
            "instruction_key": "instruction",
            "input_key": "input",
            "output_key": "output",
            "system_key": "system",
            "use_input_template": True,
            "input_template": "Input: {input}"
        }
    elif format_type == "sharegpt":
        config["format_config"] = {
            "conversations_key": "conversations",
            "role_key": "from",
            "content_key": "value",
            "single_turn_mode": False,
            "conversation_template": "sharegpt",
            "max_turns": None,
            "min_turns": 2
        }
    
    return config


def integrate_dataset_validation_into_api():
    """
    Instructions for integrating dataset validation into the Training API.
    """
    integration_code = '''
# Add to your FastAPI training endpoint:

from training_mlx.training.datasets import (
    validate_dataset, create_dataset_loader_config
)

@app.post("/v1/fine-tuning/validate-dataset")
async def validate_training_dataset(
    file_path: str = Body(..., description="Path to the dataset file")
):
    """Validate a training dataset and detect its format."""
    
    # Validate dataset
    validation_result = validate_dataset(file_path)
    
    return {
        "valid": validation_result.is_valid,
        "format": validation_result.format_type,
        "total_samples": validation_result.total_samples,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
        "sample": validation_result.sample_preview
    }

@app.post("/v1/fine-tuning/jobs")
async def create_fine_tuning_job(
    model: str = Body(...),
    training_file: str = Body(...),
    hyperparameters: Dict[str, Any] = Body(default_factory=dict),
    format: str = Body(None, description="Dataset format: 'alpaca' or 'sharegpt'")
):
    """Create a fine-tuning job with automatic format detection."""
    
    # Detect format if not specified
    if not format:
        detected_format = detect_dataset_format(training_file)
        if detected_format == "unknown":
            raise HTTPException(
                status_code=400,
                detail="Unable to detect dataset format. Please specify 'alpaca' or 'sharegpt'."
            )
        format = detected_format
    
    # Validate dataset
    validation_result = validate_dataset(training_file)
    
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid dataset",
                "errors": validation_result.errors[:5],
                "format": validation_result.format_type
            }
        )
    
    # Create dataset loader config
    dataset_config = create_dataset_loader_config(
        dataset_path=training_file,
        format_type=format,
        batch_size=hyperparameters.get("batch_size", 8),
        max_seq_length=hyperparameters.get("max_seq_length", 2048),
        shuffle=hyperparameters.get("shuffle", True)
    )
    
    # Continue with job creation...
    '''
    
    return integration_code


def example_dataset_usage():
    """
    Examples of using different dataset formats.
    """
    examples = {
        "alpaca_example": {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "input": "",  # Optional
            "system": "You are a helpful geography assistant."  # Optional
        },
        
        "sharegpt_example": {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a helpful geography assistant."
                },
                {
                    "from": "human",
                    "value": "What is the capital of France?"
                },
                {
                    "from": "gpt",
                    "value": "The capital of France is Paris."
                },
                {
                    "from": "human",
                    "value": "What about Germany?"
                },
                {
                    "from": "gpt",
                    "value": "The capital of Germany is Berlin."
                }
            ]
        },
        
        "training_request_alpaca": {
            "model": "mlx-community/Qwen2.5-1.5B-4bit",
            "training_file": "alpaca_data.json",
            "format": "alpaca",
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5
            }
        },
        
        "training_request_sharegpt": {
            "model": "mlx-community/Qwen2.5-1.5B-4bit",
            "training_file": "sharegpt_data.json",
            "format": "sharegpt",
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "max_turns": 10  # Limit conversation length
            }
        }
    }
    
    return examples


if __name__ == "__main__":
    # Test dataset detection
    print("Dataset Integration for Training MLX")
    print("=" * 50)
    
    # Show integration instructions
    print("\nIntegration code:")
    print(integrate_dataset_validation_into_api())
    
    # Show examples
    print("\nDataset format examples:")
    import json
    examples = example_dataset_usage()
    for name, example in examples.items():
        print(f"\n{name}:")
        print(json.dumps(example, indent=2))
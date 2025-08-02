#!/usr/bin/env python3
"""
Dataset handling for various formats
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import numpy as np
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    format: str
    num_examples: int
    columns: List[str]
    sample: Dict[str, Any]
    

class DatasetHandler:
    """Handles loading and processing of various dataset formats."""
    
    SUPPORTED_FORMATS = ["alpaca", "sharegpt", "jsonl", "parquet", "csv", "text"]
    
    @staticmethod
    def detect_format(file_path: str, sample_data: Optional[Dict] = None) -> str:
        """Detect dataset format from file or sample."""
        path = Path(file_path)
        
        # Check file extension
        if path.suffix == ".parquet":
            return "parquet"
        elif path.suffix == ".csv":
            return "csv"
        elif path.suffix == ".txt":
            return "text"
            
        # For JSON/JSONL, check content structure
        if sample_data:
            if "instruction" in sample_data and "output" in sample_data:
                return "alpaca"
            elif "messages" in sample_data or "conversations" in sample_data:
                return "sharegpt"
                
        return "jsonl"  # Default for JSON files
        
    @staticmethod
    def load_dataset(file_path: str, format: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Auto-detect format if not specified
        if format is None:
            # Load sample to detect format
            sample = None
            if path.suffix in [".json", ".jsonl"]:
                with open(path, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        sample = json.loads(first_line)
            format = DatasetHandler.detect_format(file_path, sample)
            
        # Load based on format
        if format == "alpaca":
            return DatasetHandler._load_alpaca(path)
        elif format == "sharegpt":
            return DatasetHandler._load_sharegpt(path)
        elif format == "jsonl":
            return DatasetHandler._load_jsonl(path)
        elif format == "parquet":
            return DatasetHandler._load_parquet(path)
        elif format == "csv":
            return DatasetHandler._load_csv(path)
        elif format == "text":
            return DatasetHandler._load_text(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    @staticmethod
    def _load_alpaca(path: Path) -> List[Dict[str, Any]]:
        """Load Alpaca format dataset."""
        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                
        # Validate Alpaca format
        required_fields = ["instruction", "output"]
        for item in data[:10]:  # Check first 10 items
            if not all(field in item for field in required_fields):
                raise ValueError("Invalid Alpaca format: missing required fields")
                
        return data
        
    @staticmethod
    def _load_sharegpt(path: Path) -> List[Dict[str, Any]]:
        """Load ShareGPT format dataset."""
        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                
        # Convert conversations to messages if needed
        processed_data = []
        for item in data:
            if "conversations" in item:
                # ShareGPT format with conversations
                processed_item = {
                    "messages": [
                        {"role": conv.get("from", "user"), "content": conv.get("value", "")}
                        for conv in item["conversations"]
                    ]
                }
            elif "messages" in item:
                # Already in messages format
                processed_item = item
            else:
                raise ValueError("Invalid ShareGPT format: missing conversations or messages")
                
            processed_data.append(processed_item)
            
        return processed_data
        
    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Load JSONL format dataset."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
        
    @staticmethod
    def _load_parquet(path: Path) -> List[Dict[str, Any]]:
        """Load Parquet format dataset."""
        df = pd.read_parquet(path)
        return df.to_dict('records')
        
    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, Any]]:
        """Load CSV format dataset."""
        df = pd.read_csv(path)
        return df.to_dict('records')
        
    @staticmethod
    def _load_text(path: Path) -> List[Dict[str, Any]]:
        """Load plain text dataset."""
        with open(path, 'r') as f:
            text = f.read()
            
        # Split into chunks (simple approach)
        chunk_size = 1024
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append({"text": text[i:i+chunk_size]})
            
        return chunks
        
    @staticmethod
    def validate_dataset(file_path: str, format: Optional[str] = None) -> DatasetInfo:
        """Validate dataset and return info."""
        try:
            # Load sample
            data = DatasetHandler.load_dataset(file_path, format)
            
            if not data:
                raise ValueError("Dataset is empty")
                
            # Get format
            detected_format = format or DatasetHandler.detect_format(file_path, data[0] if data else None)
            
            # Get columns
            columns = list(data[0].keys()) if data else []
            
            return DatasetInfo(
                format=detected_format,
                num_examples=len(data),
                columns=columns,
                sample=data[0] if data else {}
            )
            
        except Exception as e:
            raise ValueError(f"Dataset validation failed: {str(e)}")
            
    @staticmethod
    def convert_format(
        input_path: str,
        output_path: str,
        input_format: str,
        output_format: str
    ) -> int:
        """Convert dataset from one format to another."""
        # Load data
        data = DatasetHandler.load_dataset(input_path, input_format)
        
        # Convert to target format
        if output_format == "alpaca":
            converted_data = DatasetHandler._convert_to_alpaca(data, input_format)
        elif output_format == "sharegpt":
            converted_data = DatasetHandler._convert_to_sharegpt(data, input_format)
        elif output_format == "jsonl":
            converted_data = data  # JSONL can store any format
        else:
            raise ValueError(f"Conversion to {output_format} not supported")
            
        # Save converted data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "jsonl" or output_path.suffix == ".jsonl":
            with open(output_path, 'w') as f:
                for item in converted_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(output_path, 'w') as f:
                json.dump(converted_data, f, indent=2)
                
        return len(converted_data)
        
    @staticmethod
    def _convert_to_alpaca(data: List[Dict], source_format: str) -> List[Dict]:
        """Convert data to Alpaca format."""
        converted = []
        
        for item in data:
            if source_format == "sharegpt":
                # Extract instruction and response from messages
                messages = item.get("messages", [])
                if len(messages) >= 2:
                    alpaca_item = {
                        "instruction": messages[0].get("content", ""),
                        "input": "",
                        "output": messages[1].get("content", "") if len(messages) > 1 else ""
                    }
                    converted.append(alpaca_item)
            elif "text" in item:
                # Convert plain text to instruction format
                alpaca_item = {
                    "instruction": "Continue the following text:",
                    "input": item["text"][:500],
                    "output": item["text"][500:] if len(item["text"]) > 500 else ""
                }
                converted.append(alpaca_item)
            else:
                # Assume it's already close to Alpaca format
                converted.append(item)
                
        return converted
        
    @staticmethod
    def _convert_to_sharegpt(data: List[Dict], source_format: str) -> List[Dict]:
        """Convert data to ShareGPT format."""
        converted = []
        
        for item in data:
            if source_format == "alpaca":
                # Convert Alpaca to messages
                messages = [
                    {"role": "user", "content": item.get("instruction", "")},
                    {"role": "assistant", "content": item.get("output", "")}
                ]
                if item.get("input"):
                    messages[0]["content"] += f"\n{item['input']}"
                    
                converted.append({"messages": messages})
            elif "text" in item:
                # Convert plain text to conversation
                messages = [
                    {"role": "user", "content": "Please analyze the following text:"},
                    {"role": "assistant", "content": item["text"]}
                ]
                converted.append({"messages": messages})
            else:
                # Assume it's already close to ShareGPT format
                converted.append(item)
                
        return converted


class DatasetIterator:
    """Iterator for efficient dataset loading."""
    
    def __init__(self, file_path: str, batch_size: int = 32, format: Optional[str] = None):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.format = format or DatasetHandler.detect_format(file_path)
        self._file = None
        self._position = 0
        
    def __enter__(self):
        self._file = open(self.file_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over batches."""
        batch = []
        
        for line in self._file:
            if line.strip():
                try:
                    item = json.loads(line)
                    batch.append(item)
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError:
                    continue
                    
        # Yield remaining items
        if batch:
            yield batch
            
    def reset(self):
        """Reset iterator to beginning."""
        if self._file:
            self._file.seek(0)
        self._position = 0


def create_preference_dataset(
    chosen_data: List[Dict],
    rejected_data: List[Dict],
    output_path: str
) -> int:
    """Create preference dataset for RLHF training."""
    preference_data = []
    
    # Pair chosen and rejected examples
    for chosen, rejected in zip(chosen_data, rejected_data):
        if "instruction" in chosen:
            # Alpaca format
            pref_item = {
                "prompt": chosen["instruction"],
                "chosen": chosen["output"],
                "rejected": rejected["output"]
            }
        elif "messages" in chosen:
            # ShareGPT format
            pref_item = {
                "prompt": chosen["messages"][0]["content"],
                "chosen": chosen["messages"][1]["content"] if len(chosen["messages"]) > 1 else "",
                "rejected": rejected["messages"][1]["content"] if len(rejected["messages"]) > 1 else ""
            }
        else:
            continue
            
        preference_data.append(pref_item)
        
    # Save preference dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in preference_data:
            f.write(json.dumps(item) + "\n")
            
    return len(preference_data)
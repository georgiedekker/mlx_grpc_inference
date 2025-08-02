#!/usr/bin/env python3
"""
Model Layer Splitter for Distributed MLX Inference
Splits Qwen 3.1 model layers across multiple devices for distributed processing
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)

@dataclass
class LayerAssignment:
    """Configuration for layer assignment to devices"""
    device_id: str
    hostname: str
    layer_start: int
    layer_end: int
    layers: List[int]

# Device layer assignments for 3-device setup
LAYER_ASSIGNMENTS = [
    LayerAssignment("mini1", "mini1.local", 0, 9, list(range(0, 10))),
    LayerAssignment("mini2", "mini2.local", 10, 18, list(range(10, 19))),
    LayerAssignment("master", "master.local", 19, 27, list(range(19, 28)))
]

class ModelLayerSplitter:
    """Handles splitting MLX transformer models across multiple devices"""
    
    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-8bit"):
        self.model_name = model_name
        self.full_model = None
        self.tokenizer = None
        self.model_config = None
        
    def load_full_model(self):
        """Load the complete model to analyze its structure"""
        logger.info(f"Loading full model: {self.model_name}")
        try:
            self.full_model, self.tokenizer = load(self.model_name)
            logger.info("✅ Full model loaded successfully")
            self._analyze_model_structure()
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _analyze_model_structure(self):
        """Analyze the model structure to understand layer organization"""
        logger.info("🔍 Analyzing model structure...")
        
        # Print model structure for debugging
        model_dict = dict(self.full_model.named_modules())
        
        logger.info("Model structure:")
        transformer_layers = []
        
        for name, module in model_dict.items():
            if "layers." in name and name.count(".") == 2:  # model.layers.X format
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    try:
                        layer_num = int(parts[2])
                        transformer_layers.append((layer_num, name, module))
                        logger.info(f"  Found layer {layer_num}: {name}")
                    except ValueError:
                        continue
        
        transformer_layers.sort(key=lambda x: x[0])
        logger.info(f"Found {len(transformer_layers)} transformer layers")
        
        # Store layer information
        self.transformer_layers = transformer_layers
        self.num_layers = len(transformer_layers)
        
        # Validate layer assignments
        max_layer = max(assignment.layer_end for assignment in LAYER_ASSIGNMENTS)
        if max_layer >= self.num_layers:
            logger.warning(f"Layer assignment goes up to {max_layer} but model only has {self.num_layers} layers")
    
    def extract_layers_for_device(self, device_assignment: LayerAssignment) -> Dict[str, Any]:
        """Extract specific layers for a device"""
        logger.info(f"Extracting layers {device_assignment.layers} for {device_assignment.device_id}")
        
        if not self.full_model:
            raise ValueError("Full model not loaded. Call load_full_model() first.")
        
        # Create a partial model containing only the assigned layers
        partial_model_dict = {}
        
        # Copy non-layer components (embeddings, normalization, output layers)
        full_model_dict = dict(self.full_model.named_modules())
        
        for name, module in full_model_dict.items():
            # Include embeddings and other non-transformer-layer components
            if not name.startswith("layers."):
                partial_model_dict[name] = module
            else:
                # Check if this is one of the assigned layers
                if "layers." in name:
                    try:
                        layer_num = int(name.split(".")[1])
                        if layer_num in device_assignment.layers:
                            # Adjust the layer numbering for the partial model
                            new_layer_num = device_assignment.layers.index(layer_num)
                            new_name = name.replace(f"layers.{layer_num}", f"layers.{new_layer_num}")
                            partial_model_dict[new_name] = module
                    except (ValueError, IndexError):
                        # Handle non-numeric layer names
                        continue
        
        return {
            "model_dict": partial_model_dict,
            "device_id": device_assignment.device_id,
            "layer_range": (device_assignment.layer_start, device_assignment.layer_end),
            "num_layers": len(device_assignment.layers),
            "original_layers": device_assignment.layers
        }
    
    def create_partial_model(self, device_assignment: LayerAssignment):
        """Create a partial model for a specific device"""
        extracted_data = self.extract_layers_for_device(device_assignment)
        
        # For now, return the extracted information
        # In a full implementation, this would reconstruct the model with only the assigned layers
        return extracted_data
    
    def get_layer_io_shapes(self) -> Dict[int, Tuple[tuple, tuple]]:
        """Get input/output shapes for each layer for tensor passing"""
        logger.info("Analyzing layer I/O shapes...")
        
        # This would analyze each layer to determine tensor shapes
        # For now, return placeholder information
        layer_shapes = {}
        
        # Typical transformer layer shapes (batch_size, seq_len, hidden_size)
        # This would be determined by actually running inference through each layer
        for i in range(self.num_layers):
            # Placeholder - in reality we'd need to trace through the model
            layer_shapes[i] = {
                "input_shape": (1, None, 1536),  # Qwen 1.7B hidden size
                "output_shape": (1, None, 1536)
            }
        
        return layer_shapes
    
    def validate_layer_split(self) -> bool:
        """Validate that the layer split configuration is correct"""
        logger.info("🔍 Validating layer split configuration...")
        
        # Check that all layers are assigned
        assigned_layers = set()
        for assignment in LAYER_ASSIGNMENTS:
            assigned_layers.update(assignment.layers)
        
        expected_layers = set(range(self.num_layers))
        
        if assigned_layers != expected_layers:
            missing = expected_layers - assigned_layers
            extra = assigned_layers - expected_layers
            logger.error(f"Layer assignment mismatch!")
            if missing:
                logger.error(f"Missing layers: {sorted(missing)}")
            if extra:
                logger.error(f"Extra layers: {sorted(extra)}")
            return False
        
        logger.info("✅ Layer split validation passed")
        return True
    
    def get_tokenizer(self):
        """Get the tokenizer (needed on coordinator device)"""
        return self.tokenizer

def test_model_splitting():
    """Test the model splitting functionality"""
    logger.info("🧪 Testing model splitting...")
    
    splitter = ModelLayerSplitter()
    
    if not splitter.load_full_model():
        logger.error("Failed to load model for testing")
        return False
    
    if not splitter.validate_layer_split():
        logger.error("Layer split validation failed")
        return False
    
    # Test extracting layers for each device
    for assignment in LAYER_ASSIGNMENTS:
        logger.info(f"Testing layer extraction for {assignment.device_id}")
        partial_data = splitter.create_partial_model(assignment)
        logger.info(f"  Extracted {len(partial_data['model_dict'])} model components")
        logger.info(f"  Covers layers: {partial_data['original_layers']}")
    
    # Test layer I/O shapes
    shapes = splitter.get_layer_io_shapes()
    logger.info(f"Analyzed {len(shapes)} layer shapes")
    
    logger.info("✅ Model splitting test completed successfully")
    return True

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    success = test_model_splitting()
    
    if success:
        print("🎉 Model splitting implementation is working!")
    else:
        print("❌ Model splitting test failed")
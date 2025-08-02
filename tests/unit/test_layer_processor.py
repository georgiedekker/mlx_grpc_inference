"""
Unit tests for the LayerProcessor class.
"""

import pytest
from unittest.mock import MagicMock, patch
import mlx.core as mx
import mlx.nn as nn

from src.model.inference import LayerProcessor


class TestLayerProcessor:
    """Test cases for LayerProcessor."""
    
    def test_init(self, mock_mlx_model):
        """Test LayerProcessor initialization."""
        device_id = "test_device"
        assigned_layers = [0, 1, 2]
        
        processor = LayerProcessor(mock_mlx_model, device_id, assigned_layers)
        
        assert processor.model == mock_mlx_model
        assert processor.device_id == device_id
        assert processor.assigned_layers == set(assigned_layers)
        assert isinstance(processor.cache, dict)
    
    def test_process_valid_layers(self, mock_mlx_model, sample_hidden_states):
        """Test processing with valid assigned layers."""
        assigned_layers = [0, 1, 2]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        # Mock the layer processing
        with patch.object(processor, '_process_single_layer') as mock_process:
            mock_process.return_value = sample_hidden_states
            
            result = processor.process(sample_hidden_states, [0, 1], {})
            
            assert result.shape == sample_hidden_states.shape
            assert mock_process.call_count == 2  # Called for layers 0 and 1
    
    def test_process_invalid_layer(self, mock_mlx_model, sample_hidden_states):
        """Test processing with layer not assigned to device."""
        assigned_layers = [0, 1, 2]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        with pytest.raises(ValueError, match="Layer 5 not assigned to device test_device"):
            processor.process(sample_hidden_states, [0, 5], {})
    
    def test_process_empty_layers(self, mock_mlx_model, sample_hidden_states):
        """Test processing with empty layer list."""
        assigned_layers = [0, 1, 2]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        result = processor.process(sample_hidden_states, [], {})
        
        # Should return input unchanged
        assert mx.array_equal(result, sample_hidden_states)
    
    def test_process_layers_in_order(self, mock_mlx_model, sample_hidden_states):
        """Test that layers are processed in sorted order."""
        assigned_layers = [0, 1, 2, 3]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        layer_call_order = []
        
        def mock_process_single_layer(layer, hidden_states, layer_idx, context):
            layer_call_order.append(layer_idx)
            return hidden_states
        
        with patch.object(processor, '_process_single_layer', side_effect=mock_process_single_layer):
            processor.process(sample_hidden_states, [3, 1, 2], {})
            
            # Should be called in order: 1, 2, 3
            assert layer_call_order == [1, 2, 3]
    
    def test_process_single_layer_complete(self, mock_mlx_model, sample_hidden_states):
        """Test processing a single layer with all components."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        # Get the first layer from mock model
        layer = mock_mlx_model.model.layers[0]
        
        # Ensure layer has all required attributes
        layer.input_layernorm.return_value = sample_hidden_states
        layer.self_attn.return_value = sample_hidden_states
        layer.post_attention_layernorm.return_value = sample_hidden_states
        layer.mlp.return_value = sample_hidden_states
        
        result = processor._process_single_layer(layer, sample_hidden_states, 0, {})
        
        # Verify all components were called
        layer.input_layernorm.assert_called_once_with(sample_hidden_states)
        layer.self_attn.assert_called_once()
        layer.post_attention_layernorm.assert_called_once()
        layer.mlp.assert_called_once()
        
        assert result.shape == sample_hidden_states.shape
    
    def test_process_single_layer_attention_tuple_output(self, mock_mlx_model, sample_hidden_states):
        """Test processing layer when attention returns tuple."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        layer = mock_mlx_model.model.layers[0]
        
        # Mock attention to return tuple (common in some models)
        layer.self_attn.return_value = (sample_hidden_states, mx.ones((1, 5, 5)))  # (output, attention_weights)
        layer.input_layernorm.return_value = sample_hidden_states
        layer.post_attention_layernorm.return_value = sample_hidden_states
        layer.mlp.return_value = sample_hidden_states
        
        result = processor._process_single_layer(layer, sample_hidden_states, 0, {})
        
        assert result.shape == sample_hidden_states.shape
    
    def test_process_single_layer_missing_components(self, sample_hidden_states):
        """Test processing layer with missing components."""
        assigned_layers = [0]
        # Create minimal model without all components
        model = MagicMock()
        processor = LayerProcessor(model, "test_device", assigned_layers)
        
        # Create layer without some components
        layer = MagicMock()
        del layer.input_layernorm  # Remove attribute
        del layer.post_attention_layernorm
        layer.self_attn.return_value = sample_hidden_states
        layer.mlp.return_value = sample_hidden_states
        
        # Should still work with missing components
        result = processor._process_single_layer(layer, sample_hidden_states, 0, {})
        
        # Verify only available components were called
        layer.self_attn.assert_called_once()
        layer.mlp.assert_called_once()
        assert result.shape == sample_hidden_states.shape
    
    def test_process_embedding_success(self, mock_mlx_model, sample_input_tensor):
        """Test processing input embeddings."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        result = processor.process_embedding(sample_input_tensor)
        
        mock_mlx_model.model.embed_tokens.assert_called_once_with(sample_input_tensor)
        assert result.shape == (1, 10, 512)  # From mock
    
    def test_process_embedding_missing_embed_tokens(self, sample_input_tensor):
        """Test processing embeddings when model lacks embed_tokens."""
        assigned_layers = [0]
        model = MagicMock()
        del model.model.embed_tokens  # Remove attribute
        
        processor = LayerProcessor(model, "test_device", assigned_layers)
        
        with pytest.raises(ValueError, match="Model does not have embed_tokens layer"):
            processor.process_embedding(sample_input_tensor)
    
    def test_process_output_success(self, mock_mlx_model, sample_hidden_states):
        """Test processing final output layers."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        result = processor.process_output(sample_hidden_states)
        
        mock_mlx_model.model.norm.assert_called_once_with(sample_hidden_states)
        mock_mlx_model.lm_head.assert_called_once()
        assert result.shape == (1, 10, 32000)  # From mock
    
    def test_process_output_missing_norm(self, sample_hidden_states):
        """Test processing output when model lacks norm layer."""
        assigned_layers = [0]
        model = MagicMock()
        del model.model.norm  # Remove attribute
        model.lm_head.return_value = mx.ones((1, 5, 32000))
        
        processor = LayerProcessor(model, "test_device", assigned_layers)
        
        # Should still work without norm
        result = processor.process_output(sample_hidden_states)
        
        model.lm_head.assert_called_once_with(sample_hidden_states)
        assert result.shape == (1, 5, 32000)
    
    def test_process_output_missing_lm_head(self, mock_mlx_model, sample_hidden_states):
        """Test processing output when model lacks lm_head."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        del mock_mlx_model.lm_head  # Remove attribute
        
        with pytest.raises(ValueError, match="Model does not have lm_head"):
            processor.process_output(sample_hidden_states)
    
    def test_get_memory_usage(self, mock_mlx_model):
        """Test getting memory usage statistics."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        memory_stats = processor.get_memory_usage()
        
        assert isinstance(memory_stats, dict)
        assert 'allocated_gb' in memory_stats
        assert 'cached_gb' in memory_stats
        assert 'reserved_gb' in memory_stats
        
        # All values should be non-negative numbers
        for key, value in memory_stats.items():
            assert isinstance(value, (int, float))
            assert value >= 0
    
    def test_layer_not_found_warning(self, mock_mlx_model, sample_hidden_states, caplog):
        """Test warning when requested layer doesn't exist in model."""
        assigned_layers = [0, 1, 2, 10]  # Layer 10 doesn't exist in mock (only 0-8)
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        # Try to process non-existent layer
        result = processor.process(sample_hidden_states, [10], {})
        
        # Should log warning and return input unchanged
        assert "Layer 10 not found in model" in caplog.text
        assert mx.array_equal(result, sample_hidden_states)
    
    def test_process_with_context(self, mock_mlx_model, sample_hidden_states):
        """Test processing layers with context information."""
        assigned_layers = [0]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        context = {"attention_mask": mx.ones((1, 5)), "position_ids": mx.arange(5)}
        
        with patch.object(processor, '_process_single_layer') as mock_process:
            mock_process.return_value = sample_hidden_states
            
            processor.process(sample_hidden_states, [0], context)
            
            # Verify context was passed to layer processing
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[0][3] == context  # context is 4th argument
    
    @pytest.mark.parametrize("layer_idx,expected_layers", [
        ([0], [0]),
        ([2, 1, 0], [0, 1, 2]),
        ([1], [1]),
        ([0, 2], [0, 2])
    ])
    def test_layer_processing_order(self, mock_mlx_model, sample_hidden_states, 
                                   layer_idx, expected_layers):
        """Test that layers are always processed in ascending order."""
        assigned_layers = [0, 1, 2]
        processor = LayerProcessor(mock_mlx_model, "test_device", assigned_layers)
        
        processed_layers = []
        
        def mock_process_single_layer(layer, hidden_states, idx, context):
            processed_layers.append(idx)
            return hidden_states
        
        with patch.object(processor, '_process_single_layer', side_effect=mock_process_single_layer):
            processor.process(sample_hidden_states, layer_idx, {})
            
            assert processed_layers == expected_layers
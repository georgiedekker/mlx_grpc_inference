#!/usr/bin/env python3
"""
Comprehensive debugging script for coordinator-only inference.
Instruments the full pipeline to identify where gibberish is introduced.
"""

import sys
import logging
import hashlib
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceDebugger:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self, model_name="mlx-community/Qwen3-1.7B-8bit"):
        """Load model and log details about it."""
        logger.info(f"Loading model: {model_name}")
        
        # Load model
        self.model, self.tokenizer = load(model_name)
        
        # Log model details
        logger.info(f"Model type: {type(self.model)}")
        logger.info(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:20]}")
        
        # Check for model structure
        if hasattr(self.model, 'model'):
            logger.info("Model has nested 'model' attribute")
            logger.info(f"Inner model type: {type(self.model.model)}")
            logger.info(f"Inner model attributes: {[attr for attr in dir(self.model.model) if not attr.startswith('_')][:20]}")
        
        # Log tokenizer details
        logger.info(f"Tokenizer type: {type(self.tokenizer)}")
        logger.info(f"Vocabulary size: {len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 'unknown'}")
        logger.info(f"EOS token ID: {self.tokenizer.eos_token_id}")
        
        # Check model weights
        self._log_weight_checksums()
        
    def _log_weight_checksums(self):
        """Log checksums of key model weights."""
        logger.info("Computing weight checksums...")
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Check first layer weights
            first_layer = self.model.model.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                weight = first_layer.self_attn.q_proj.weight
                mx.eval(weight)
                # Convert to numpy for checksum
                weight_np = np.array(weight.astype(mx.float32))
                checksum = hashlib.md5(weight_np.tobytes()).hexdigest()[:8]
                logger.info(f"First layer q_proj weight checksum: {checksum}")
                logger.info(f"Weight shape: {weight.shape}, dtype: {weight.dtype}")
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embed_weight = self.model.model.embed_tokens.weight
            mx.eval(embed_weight)
            embed_np = np.array(embed_weight.astype(mx.float32))
            checksum = hashlib.md5(embed_np.tobytes()).hexdigest()[:8]
            logger.info(f"Embedding weight checksum: {checksum}")
            logger.info(f"Embedding shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")
    
    def test_tokenization(self, prompt="Hello"):
        """Test tokenization and log details."""
        logger.info(f"\n=== Testing tokenization for prompt: '{prompt}' ===")
        
        # Encode
        input_ids = self.tokenizer.encode(prompt)
        logger.info(f"Token IDs: {input_ids}")
        logger.info(f"Number of tokens: {len(input_ids)}")
        
        # Decode back
        decoded = self.tokenizer.decode(input_ids)
        logger.info(f"Decoded text: '{decoded}'")
        logger.info(f"Decode matches original: {decoded == prompt}")
        
        # Convert to tensor
        input_tensor = mx.array(input_ids)
        logger.info(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        
        return input_tensor
    
    def test_embeddings(self, input_ids):
        """Test embedding layer and log details."""
        logger.info("\n=== Testing embeddings ===")
        
        # Add batch dimension if needed
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
            logger.info(f"Added batch dimension: {input_ids.shape}")
        
        # Get embeddings
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embeddings = self.model.model.embed_tokens(input_ids)
            mx.eval(embeddings)
            
            logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            logger.info(f"Embeddings stats - min: {mx.min(embeddings).item():.4f}, "
                       f"max: {mx.max(embeddings).item():.4f}, "
                       f"mean: {mx.mean(embeddings).item():.4f}")
            
            # Check for NaN/Inf
            has_nan = mx.any(mx.isnan(embeddings)).item()
            has_inf = mx.any(mx.isinf(embeddings)).item()
            logger.info(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            return embeddings
        else:
            logger.error("Model doesn't have embed_tokens!")
            return None
    
    def test_forward_pass(self, input_ids):
        """Test full forward pass with detailed logging."""
        logger.info("\n=== Testing forward pass ===")
        
        # Get embeddings
        embeddings = self.test_embeddings(input_ids)
        if embeddings is None:
            return None
        
        hidden_states = embeddings
        
        # Process through layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers[:5]):  # Test first 5 layers
                logger.info(f"\nLayer {i}:")
                logger.info(f"  Input shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
                
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
                
                mx.eval(hidden_states)
                logger.info(f"  Output shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
                logger.info(f"  Output stats - min: {mx.min(hidden_states).item():.4f}, "
                           f"max: {mx.max(hidden_states).item():.4f}, "
                           f"mean: {mx.mean(hidden_states).item():.4f}")
        
        # Final norm
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            hidden_states = self.model.model.norm(hidden_states)
            mx.eval(hidden_states)
            logger.info(f"\nAfter norm - shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        
        # Get logits
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden_states)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head'):
            logits = self.model.model.lm_head(hidden_states)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            logits = self.model.model.embed_tokens.as_linear(hidden_states)
            logger.info("Using tied embeddings for output projection")
        else:
            logger.error("No output projection found!")
            return None
        
        mx.eval(logits)
        logger.info(f"\nLogits shape: {logits.shape}, dtype: {logits.dtype}")
        logger.info(f"Logits stats - min: {mx.min(logits).item():.4f}, "
                   f"max: {mx.max(logits).item():.4f}, "
                   f"mean: {mx.mean(logits).item():.4f}")
        
        # Check top tokens
        last_logits = logits[0, -1, :]  # Last position
        top_k = 10
        top_indices = mx.argpartition(-last_logits, kth=top_k)[:top_k]
        top_values = last_logits[top_indices]
        
        logger.info(f"\nTop {top_k} token predictions:")
        for i in range(top_k):
            token_id = top_indices[i].item()
            logit_value = top_values[i].item()
            token_text = self.tokenizer.decode([token_id])
            logger.info(f"  {i+1}. Token {token_id} ('{token_text}'): logit={logit_value:.4f}")
        
        return logits
    
    def test_sampling(self, logits, temperature=0.7, top_p=1.0):
        """Test sampling from logits."""
        logger.info(f"\n=== Testing sampling (temp={temperature}, top_p={top_p}) ===")
        
        # Create sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        
        # Sample multiple times to check consistency
        samples = []
        for i in range(5):
            next_token = sampler(logits[:, -1:, :])
            token_id = next_token.item()
            samples.append(token_id)
        
        logger.info(f"Sampled tokens (5 samples): {samples}")
        logger.info(f"Decoded samples: {[self.tokenizer.decode([t]) for t in samples]}")
        
        # Check if all samples are the same (low temperature)
        if len(set(samples)) == 1:
            logger.warning("All samples are identical - might indicate deterministic sampling")
        
        return samples[0]
    
    def test_generation_methods(self, prompt="Hello"):
        """Test different generation methods."""
        logger.info(f"\n=== Testing generation methods for '{prompt}' ===")
        
        # Method 1: Using mlx_lm.generate
        logger.info("\nMethod 1: mlx_lm.generate()")
        response1 = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=20,
            temp=0.7,
            verbose=True
        )
        logger.info(f"Generated: '{response1}'")
        
        # Method 2: Manual generation
        logger.info("\nMethod 2: Manual generation")
        input_ids = mx.array(self.tokenizer.encode(prompt))
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        generated_ids = []
        current_ids = input_ids
        
        for i in range(20):
            # Forward pass
            embeddings = self.model.model.embed_tokens(current_ids)
            hidden_states = embeddings
            
            # Only process through layers (simplified)
            for layer in self.model.model.layers:
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            
            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.model.embed_tokens.as_linear(hidden_states)
            
            # Sample
            sampler = make_sampler(temp=0.7)
            next_token = sampler(logits[:, -1:, :])
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            if token_id == self.tokenizer.eos_token_id:
                break
            
            # Append for next iteration
            new_token_tensor = mx.array([[token_id]])
            current_ids = mx.concatenate([current_ids, new_token_tensor], axis=1)
        
        response2 = self.tokenizer.decode(generated_ids)
        logger.info(f"Generated: '{response2}'")
        
        return response1, response2
    
    def run_all_tests(self):
        """Run all debugging tests."""
        logger.info("=" * 80)
        logger.info("Starting comprehensive inference debugging")
        logger.info("=" * 80)
        
        # Load model
        self.load_model()
        
        # Test simple prompt
        prompt = "Hello"
        input_ids = self.test_tokenization(prompt)
        
        # Test forward pass
        logits = self.test_forward_pass(input_ids)
        if logits is not None:
            # Test sampling
            self.test_sampling(logits)
        
        # Test generation methods
        self.test_generation_methods(prompt)
        
        # Test with chat format
        chat_prompt = "user: Hello\nassistant: "
        logger.info(f"\n=== Testing with chat format: '{chat_prompt}' ===")
        self.test_generation_methods(chat_prompt)
        
        logger.info("\n" + "=" * 80)
        logger.info("Debugging complete")
        logger.info("=" * 80)

if __name__ == "__main__":
    debugger = InferenceDebugger()
    debugger.run_all_tests()
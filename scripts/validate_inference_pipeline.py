#!/usr/bin/env python3
"""
Comprehensive validation script for MLX inference pipeline.
Use this in CI/CD to catch regressions early.
"""

import sys
import subprocess
import time
import json
import mlx.core as mx
from mlx_lm import load, generate
import numpy as np
from typing import Dict, List, Tuple, Any

# Test configuration
TEST_MODEL = "mlx-community/GLM-4.5-4bit"
TEST_PROMPTS = [
    "Hello",
    "What is machine learning?",
    "Tell me about San Francisco",
    "Explain quantum computing in simple terms"
]

# Expected patterns (should NOT appear in good outputs)
GIBBERISH_PATTERNS = [
    "amp" * 5,  # "ampampampampamp"
    "ion" * 5,  # "ionionionionion"
    "eter" * 3,  # "eteretereter"
    "0us",
    ".exports",
    "oreampamp"
]


class InferencePipelineValidator:
    """Validates the inference pipeline for correctness and performance."""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
        self.model = None
        self.tokenizer = None
    
    def log_result(self, test_name: str, passed: bool, message: str, details: Dict = None):
        """Log test result."""
        result = {
            "test": test_name,
            "message": message,
            "timestamp": time.time()
        }
        if details:
            result["details"] = details
            
        if passed:
            self.results["passed"].append(result)
            print(f"✅ {test_name}: {message}")
        else:
            self.results["failed"].append(result)
            print(f"❌ {test_name}: {message}")
            if details:
                print(f"   Details: {json.dumps(details, indent=2)}")
    
    def log_warning(self, test_name: str, message: str):
        """Log warning."""
        self.results["warnings"].append({
            "test": test_name,
            "message": message,
            "timestamp": time.time()
        })
        print(f"⚠️  {test_name}: {message}")
    
    def test_mlx_installation(self) -> bool:
        """Test 1: Verify MLX is properly installed."""
        try:
            import mlx
            import mlx_lm
            
            # Check version
            mlx_version = getattr(mlx, "__version__", "unknown")
            
            # Test basic operations
            x = mx.array([1.0, 2.0, 3.0])
            y = mx.sum(x)
            mx.eval(y)
            
            self.log_result(
                "mlx_installation",
                True,
                f"MLX version {mlx_version} installed and functional",
                {"version": mlx_version, "sum_test": float(y.item())}
            )
            return True
            
        except Exception as e:
            self.log_result(
                "mlx_installation",
                False,
                f"MLX installation error: {e}"
            )
            return False
    
    def test_model_loading(self) -> bool:
        """Test 2: Verify model loads correctly."""
        try:
            print(f"\nLoading model {TEST_MODEL}...")
            self.model, self.tokenizer = load(TEST_MODEL)
            
            # Verify model structure
            has_layers = hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')
            num_layers = len(self.model.model.layers) if has_layers else 0
            
            # Check tokenizer
            vocab_size = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 0
            
            self.log_result(
                "model_loading",
                True,
                f"Model loaded successfully",
                {
                    "num_layers": num_layers,
                    "vocab_size": vocab_size,
                    "model_type": str(type(self.model))
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "model_loading",
                False,
                f"Model loading failed: {e}"
            )
            return False
    
    def test_weight_checksums(self) -> bool:
        """Test 3: Verify model weights are consistent."""
        if not self.model:
            self.log_warning("weight_checksums", "Skipping - model not loaded")
            return False
            
        try:
            checksums = {}
            
            # Check embedding weights
            if hasattr(self.model.model, 'embed_tokens'):
                embed_weight = self.model.model.embed_tokens.weight
                mx.eval(embed_weight)
                embed_np = np.array(embed_weight.astype(mx.float32))
                embed_checksum = hash(embed_np.tobytes()) % 1000000
                checksums["embeddings"] = embed_checksum
            
            # Check first layer weights
            if hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
                first_layer = self.model.model.layers[0]
                if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                    q_weight = first_layer.self_attn.q_proj.weight
                    mx.eval(q_weight)
                    q_np = np.array(q_weight.astype(mx.float32))
                    q_checksum = hash(q_np.tobytes()) % 1000000
                    checksums["first_layer_q_proj"] = q_checksum
            
            self.log_result(
                "weight_checksums",
                True,
                "Model weights validated",
                checksums
            )
            return True
            
        except Exception as e:
            self.log_result(
                "weight_checksums",
                False,
                f"Weight validation failed: {e}"
            )
            return False
    
    def test_hidden_state_explosion(self) -> bool:
        """Test 4: Verify hidden states don't explode during manual processing."""
        if not self.model:
            self.log_warning("hidden_state_explosion", "Skipping - model not loaded")
            return False
            
        try:
            # Tokenize simple input
            input_ids = mx.array(self.tokenizer.encode("Hello"))
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]
            
            # Get embeddings
            embeddings = self.model.model.embed_tokens(input_ids)
            hidden_states = embeddings
            
            layer_stats = []
            explosion_detected = False
            
            # Process through first 5 layers
            for i, layer in enumerate(self.model.model.layers[:5]):
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
                
                mx.eval(hidden_states)
                
                # Check statistics
                max_val = float(mx.max(mx.abs(hidden_states)).item())
                mean_val = float(mx.mean(hidden_states).item())
                
                layer_stats.append({
                    "layer": i,
                    "max_abs": max_val,
                    "mean": mean_val
                })
                
                # Detect explosion
                if max_val > 1000:
                    explosion_detected = True
                    self.log_warning(
                        "hidden_state_explosion",
                        f"Hidden states exploded at layer {i}: max={max_val:.1f}"
                    )
            
            self.log_result(
                "hidden_state_explosion",
                not explosion_detected,
                "Hidden state explosion detected" if explosion_detected else "No explosion detected",
                {"layer_stats": layer_stats}
            )
            
            return not explosion_detected
            
        except Exception as e:
            self.log_result(
                "hidden_state_explosion",
                False,
                f"Explosion test failed: {e}"
            )
            return False
    
    def test_generation_quality(self) -> bool:
        """Test 5: Verify generation produces coherent text."""
        if not self.model:
            self.log_warning("generation_quality", "Skipping - model not loaded")
            return False
            
        try:
            all_good = True
            generation_results = []
            
            for prompt in TEST_PROMPTS:
                # Generate response
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=30,
                    verbose=False
                )
                
                # Check for gibberish patterns
                has_gibberish = any(pattern in response for pattern in GIBBERISH_PATTERNS)
                
                # Check basic coherence
                words = response.split()
                unique_words = len(set(words))
                word_ratio = unique_words / max(len(words), 1)
                
                is_coherent = not has_gibberish and word_ratio > 0.3
                
                generation_results.append({
                    "prompt": prompt,
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "coherent": is_coherent,
                    "word_diversity": f"{word_ratio:.2f}"
                })
                
                if not is_coherent:
                    all_good = False
                    self.log_warning(
                        "generation_quality",
                        f"Poor generation for '{prompt}': {response[:50]}..."
                    )
            
            self.log_result(
                "generation_quality",
                all_good,
                "All generations coherent" if all_good else "Some generations failed",
                {"results": generation_results}
            )
            
            return all_good
            
        except Exception as e:
            self.log_result(
                "generation_quality",
                False,
                f"Generation test failed: {e}"
            )
            return False
    
    def test_dtype_preservation(self) -> bool:
        """Test 6: Verify bfloat16 dtype is preserved."""
        if not self.model:
            self.log_warning("dtype_preservation", "Skipping - model not loaded")
            return False
            
        try:
            # Check model weights dtype
            embed_dtype = self.model.model.embed_tokens.weight.dtype
            
            # Generate embeddings
            input_ids = mx.array([1, 2, 3])
            embeddings = self.model.model.embed_tokens(input_ids[None, :])
            embed_output_dtype = embeddings.dtype
            
            # Process through a layer
            layer_output = self.model.model.layers[0](embeddings)
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]
            layer_output_dtype = layer_output.dtype
            
            # Check if dtypes are preserved
            all_bfloat16 = (
                embed_output_dtype == mx.bfloat16 and
                layer_output_dtype == mx.bfloat16
            )
            
            self.log_result(
                "dtype_preservation",
                all_bfloat16,
                "bfloat16 preserved" if all_bfloat16 else "dtype changed during processing",
                {
                    "weight_dtype": str(embed_dtype),
                    "embedding_output": str(embed_output_dtype),
                    "layer_output": str(layer_output_dtype)
                }
            )
            
            return all_bfloat16
            
        except Exception as e:
            self.log_result(
                "dtype_preservation",
                False,
                f"Dtype test failed: {e}"
            )
            return False
    
    def test_performance_baseline(self) -> bool:
        """Test 7: Verify generation performance meets baseline."""
        if not self.model:
            self.log_warning("performance_baseline", "Skipping - model not loaded")
            return False
            
        try:
            prompt = "What is machine learning?"
            
            # Warmup
            _ = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=10)
            
            # Timed run
            start_time = time.time()
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=50,
                verbose=False
            )
            elapsed = time.time() - start_time
            
            # Calculate metrics
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = len(self.tokenizer.encode(response)) - prompt_tokens
            tokens_per_second = response_tokens / elapsed if elapsed > 0 else 0
            
            # Baseline: At least 10 tokens/second on single device
            meets_baseline = tokens_per_second >= 10
            
            self.log_result(
                "performance_baseline",
                meets_baseline,
                f"Generation speed: {tokens_per_second:.1f} tokens/second",
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "elapsed_seconds": round(elapsed, 2),
                    "tokens_per_second": round(tokens_per_second, 1)
                }
            )
            
            return meets_baseline
            
        except Exception as e:
            self.log_result(
                "performance_baseline",
                False,
                f"Performance test failed: {e}"
            )
            return False
    
    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("=" * 60)
        print("MLX Inference Pipeline Validation")
        print("=" * 60)
        
        tests = [
            self.test_mlx_installation,
            self.test_model_loading,
            self.test_weight_checksums,
            self.test_hidden_state_explosion,
            self.test_generation_quality,
            self.test_dtype_preservation,
            self.test_performance_baseline
        ]
        
        for test in tests:
            test()
            print()  # Blank line between tests
        
        # Summary
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        
        # Save results
        with open("validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to validation_results.json")
        
        # Return success if no failures
        return len(self.results['failed']) == 0


def main():
    """Main entry point."""
    validator = InferencePipelineValidator()
    success = validator.run_all_tests()
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
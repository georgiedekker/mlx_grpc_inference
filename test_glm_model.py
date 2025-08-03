#!/usr/bin/env python3
"""
Test GLM-4.5-4bit model loading and structure.
"""

from mlx_lm import load, generate
import mlx.core as mx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_glm_model():
    """Test GLM-4.5-4bit model."""
    
    # Load model
    logger.info("Loading GLM-4.5-Air-4bit model...")
    model, tokenizer = load("mlx-community/GLM-4.5-Air-4bit")
    
    # Check model structure
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
    
    # Check for layers
    if hasattr(model, 'model'):
        logger.info(f"Inner model type: {type(model.model)}")
        if hasattr(model.model, 'layers'):
            logger.info(f"Number of layers: {len(model.model.layers)}")
        else:
            logger.info("No 'layers' attribute in model.model")
    elif hasattr(model, 'layers'):
        logger.info(f"Number of layers: {len(model.layers)}")
    else:
        logger.info("No 'layers' attribute found")
    
    # Check tokenizer
    logger.info(f"Tokenizer type: {type(tokenizer)}")
    if hasattr(tokenizer, 'vocab_size'):
        logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    elif hasattr(tokenizer, 'vocab'):
        logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Test tokenization
    test_prompt = "Hello, how are you?"
    logger.info(f"\nTesting tokenization for: '{test_prompt}'")
    tokens = tokenizer.encode(test_prompt)
    logger.info(f"Token IDs: {tokens}")
    logger.info(f"Number of tokens: {len(tokens)}")
    
    # Test generation
    logger.info("\nTesting generation...")
    response = generate(
        model,
        tokenizer,
        prompt=test_prompt,
        max_tokens=30,
        verbose=True
    )
    logger.info(f"Generated response: '{response}'")
    
    # Check model weights dtype
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_weight = model.model.embed_tokens.weight
        logger.info(f"\nEmbedding weight dtype: {embed_weight.dtype}")
        logger.info(f"Embedding weight shape: {embed_weight.shape}")
    
    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = test_glm_model()
        print("\n✅ GLM-4.5-4bit model loaded and tested successfully!")
    except Exception as e:
        print(f"\n❌ Error testing GLM model: {e}")
        import traceback
        traceback.print_exc()
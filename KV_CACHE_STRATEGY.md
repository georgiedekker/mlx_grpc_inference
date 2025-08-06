# KV Cache Strategy for Distributed MLX Inference

## Current Implementation

After analyzing the experimental branch and its complex distributed KV cache attempts, we've implemented a hybrid approach on the main branch:

### 1. Automatic Strategy Selection

The system automatically chooses between distributed and single-device inference based on:

- **Prompt length**: Longer prompts (>100 tokens) use single-device
- **Generation length**: Longer generations (>50 tokens) use single-device
- **Environment variable**: `USE_DISTRIBUTED_INFERENCE` can be set to:
  - `'auto'` (default): System decides based on sequence length
  - `'always'`: Always use distributed inference (no KV cache)
  - `'never'`: Always use single-device (with KV cache)

### 2. Why This Approach?

**Distributed inference without KV cache has quadratic complexity:**
- Token 1: Process 1 token through all layers
- Token 2: Process 2 tokens through all layers
- Token N: Process N tokens through all layers

This results in O(N²) complexity instead of O(N) with proper KV caching.

**Single-device with KV cache:**
- Uses MLX's built-in `generate()` function
- Maintains KV cache across tokens
- Linear complexity O(N)

### 3. Performance Characteristics

| Scenario | Strategy | Performance |
|----------|----------|-------------|
| Short prompt (<100 tokens), short generation (<50) | Distributed | Good - network overhead acceptable |
| Long prompt or generation | Single-device | Better - KV cache crucial |
| Very large models (>10B params) | Distributed required | Slower but enables larger models |

### 4. Usage Examples

```bash
# Auto mode (default)
./launch.sh start

# Force distributed (no KV cache)
USE_DISTRIBUTED_INFERENCE=always ./launch.sh start

# Force single-device (with KV cache)
USE_DISTRIBUTED_INFERENCE=never ./launch.sh start
```

### 5. Future Improvements

1. **Chunked generation**: Process multiple tokens at once in distributed mode
2. **Partial KV cache**: Cache only attention keys, not values
3. **Ring communication**: Pass KV cache in a ring between devices
4. **Speculative decoding**: Use a small model to generate candidates

## Why Not Full Distributed KV Cache?

The experimental branch attempted full distributed KV caching but encountered:

1. **Serialization overhead**: Sending large cache tensors between devices
2. **Synchronization complexity**: Keeping cache states perfectly aligned
3. **MLX limitations**: KV cache objects not designed for distribution
4. **Network bandwidth**: Cache updates can exceed Thunderbolt bandwidth

For most use cases, the hybrid approach provides better performance than attempting to distribute the KV cache.
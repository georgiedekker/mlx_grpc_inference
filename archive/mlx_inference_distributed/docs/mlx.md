mlx-community/Qwen3-1.7B-8bit
This model mlx-community/Qwen3-1.7B-8bit was converted to MLX format from Qwen/Qwen3-1.7B using mlx-lm version 0.24.0.

Use with mlx
uv install mlx-lm

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-1.7B-8bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
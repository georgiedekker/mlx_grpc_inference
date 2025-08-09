#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.
"""
Official MLX pipeline_generate.py from Awni's gist
Run with:
mlx.launch --hostfile hosts.json --backend mpi pipeline_generate.py --prompt "hello"
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model, load_tokenizer


def download(repo: str, allow_patterns: list[str]) -> Path:
    return Path(
        snapshot_download(
            repo,
            allow_patterns=allow_patterns,
        )
    )


def shard_and_load(repo):
    # Get model path with everything but weight safetensors
    model_path = download(
        repo,
        allow_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
    )

    # Lazy load and shard model to figure out which weights we need
    model, _ = load_model(model_path, lazy=True, strict=False)

    group = mx.distributed.init()
    rank = group.rank()
    
    # This is the key - the model MUST have a pipeline() method!
    model.model.pipeline(group)

    # Figure out which files we need for the local shard
    with open(model_path / "model.safetensors.index.json", "r") as fid:
        weight_index = json.load(fid)["weight_map"]

    local_files = set()
    for k, _ in tree_flatten(model.parameters()):
        local_files.add(weight_index[k])

    # Download weights for local shard
    download(repo, allow_patterns=list(local_files))

    # Load and shard the model, and load the weights
    tokenizer = load_tokenizer(model_path)
    model, _ = load_model(model_path, lazy=True, strict=False)
    model.model.pipeline(group)
    mx.eval(model.parameters())

    # Synchronize processes before generation
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM pipelined inference example")
    parser.add_argument(
        "--model",
        default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        help="HF repo or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="What is 2+2?",
        help="Message to be processed by the model",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    rprint(f"Loading model: {args.model}")
    model, tokenizer = shard_and_load(args.model)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    rprint("Generating...")
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")
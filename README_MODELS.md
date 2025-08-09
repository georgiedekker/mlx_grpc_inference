# Model Setup Instructions

## Pre-downloading Models (Required)

Models must be downloaded BEFORE running the distributed server. The system will NOT download models automatically to avoid repeated downloads.

### Step 1: Download Model on Server Machine (mini1)

#### Option A: Using Hugging Face CLI (Recommended)

```bash
# Install huggingface-hub if not already installed
pip install -U "huggingface-hub[cli]"

# Download your chosen model (it automatically uses the right cache directory)
hf download mlx-community/Qwen2.5-7B-Instruct-4bit
```

#### Option B: Using MLX

```bash
# Activate the virtual environment
source .venv/bin/activate

# Download your chosen model
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

### Available Models by Memory Size

- **16GB cluster**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- **32GB cluster**: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- **48GB cluster**: `mlx-community/Qwen2.5-14B-Instruct-4bit`
- **64GB+ cluster**: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`

### Step 2: Model Location

Models are stored in: `~/.cache/huggingface/hub/models--{model-name}`

The launch script will automatically sync models from the server to worker nodes.

### Step 3: Run the Server

After downloading the model:

```bash
./launch.sh start
```

The system will:
1. Detect the model exists locally
2. Sync it to worker nodes automatically
3. Start distributed inference

### Troubleshooting

If you see "Model not found" errors:
1. Check the model is downloaded: `ls ~/.cache/huggingface/hub/`
2. Ensure the model name matches exactly
3. Re-download if needed using the command above
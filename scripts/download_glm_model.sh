#!/bin/bash
# Download GLM-4.5-4bit model efficiently using HuggingFace transfer

set -e

MODEL_ID="mlx-community/GLM-4.5-4bit"
CACHE_DIR="$HOME/.cache/huggingface/hub"

echo "=== Downloading GLM-4.5-4bit model ==="
echo "Model: $MODEL_ID"
echo "Cache directory: $CACHE_DIR"
echo ""

# Enable HuggingFace transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install hf_transfer if not already installed
if ! python -c "import hf_transfer" 2>/dev/null; then
    echo "Installing hf_transfer for faster downloads..."
    uv pip install hf_transfer
fi

# Create Python script to download model
cat > download_model.py << 'EOF'
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download
import time

model_id = "mlx-community/GLM-4.5-4bit"
print(f"Starting download of {model_id}...")
start_time = time.time()

try:
    # Download all model files
    local_dir = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Download completed in {elapsed/60:.1f} minutes")
    print(f"Model downloaded to: {local_dir}")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    exit(1)
EOF

# Run the download
uv run python download_model.py

# Verify model files
echo ""
echo "=== Verifying model files ==="
echo "Computing checksums..."

# Find the model directory
MODEL_DIR=$(find "$CACHE_DIR" -name "*GLM-4.5-4bit*" -type d | head -1)
if [ -n "$MODEL_DIR" ]; then
    cd "$MODEL_DIR"
    echo "Model directory: $MODEL_DIR"
    echo ""
    
    # Compute checksums for all safetensor files
    if ls model-*.safetensors 1> /dev/null 2>&1; then
        sha256sum model-*.safetensors > model_checksums.txt
        OVERALL_CHECKSUM=$(sha256sum model-*.safetensors | sha256sum | cut -d' ' -f1)
        echo "Overall model checksum: $OVERALL_CHECKSUM"
        echo "Individual file checksums saved to: $MODEL_DIR/model_checksums.txt"
    else
        echo "No safetensor files found!"
    fi
else
    echo "Model directory not found!"
fi

echo ""
echo "=== Next steps ==="
echo "1. To sync to other machines, run:"
echo "   rsync -avz --progress $MODEL_DIR/ mini2.local:$MODEL_DIR/"
echo "   rsync -avz --progress $MODEL_DIR/ master.local:$MODEL_DIR/"
echo ""
echo "2. Verify checksums on each machine:"
echo "   ssh mini2.local 'cd $MODEL_DIR && sha256sum model-*.safetensors | sha256sum'"
echo "   ssh master.local 'cd $MODEL_DIR && sha256sum model-*.safetensors | sha256sum'"
echo ""
echo "3. The overall checksum should match: $OVERALL_CHECKSUM"

# Clean up
rm -f download_model.py
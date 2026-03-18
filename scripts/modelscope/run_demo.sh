#!/bin/bash

# Configuration
# On ModelScope, /mnt/workspace/ is persistent.
TARGET_DIR="/mnt/workspace/gca_persistent"
PYTHON_CMD="python"

# Check for GCA directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: GCA directory '$TARGET_DIR' does not exist."
    echo "Please run 'scripts/modelscope/setup_env.sh' first."
    exit 1
fi

# Navigate into GCA directory
cd "$TARGET_DIR" || { echo "Error: Failed to enter $TARGET_DIR"; exit 1; }

echo "Running GCA on MMSI Benchmark..."
echo "Current directory: $(pwd)"

# Optional: Set VLM Provider (Set to 'vllm' for local deployment, 'openai' for API)
# export VLM_PROVIDER="openai" 

# Check if OPENAI_API_KEY is set (required for reasoning)
if [ -z "$OPENAI_API_KEY" ] && [ "$VLM_PROVIDER" != "vllm" ]; then
    echo "Warning: OPENAI_API_KEY is not set. The demo might fail if the model requires it."
    echo "Please edit this script or export your API key before running."
    # read -p "Press Enter to continue anyway or Ctrl+C to abort..."
fi

$PYTHON_CMD -m entrypoints.agent --benchmark mmsi --concurrency 16

echo "Demo complete!"

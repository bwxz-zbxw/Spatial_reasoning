#!/bin/bash

# Configuration
# On ModelScope, /mnt/workspace/ is persistent.
TARGET_DIR="/mnt/workspace/gca_persistent"
MMSI_DATA_DIR="data/mmsi"

# Check for GCA directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: GCA directory '$TARGET_DIR' does not exist."
    echo "Please run 'scripts/modelscope/setup_env.sh' first."
    exit 1
fi

# Navigate into GCA directory
cd "$TARGET_DIR" || { echo "Error: Failed to enter $TARGET_DIR"; exit 1; }

echo "Setting up MMSI Data..."
echo "Current directory: $(pwd)"

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Creating data directory '$MMSI_DATA_DIR'..."
mkdir -p "$MMSI_DATA_DIR"

echo "Downloading MMSI-Bench from ModelScope..."
# Use HuggingFace mirror for China
export HF_ENDPOINT="https://hf-mirror.com"

# Install hf_transfer for acceleration
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download command
huggingface-cli download RunsenXu/MMSI-Bench --local-dir "$MMSI_DATA_DIR" --repo-type dataset --resume-download

echo "MMSI Data setup complete!"
echo "Check '$TARGET_DIR/$MMSI_DATA_DIR' for downloaded files."

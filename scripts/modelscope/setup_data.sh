#!/bin/bash

# Configuration
TARGET_DIR="gca"
MMSI_DATA_DIR="data/mmsi"

# Ensure we are in the project root
cd "$(dirname "$0")/../.." || { echo "Error: Could not navigate to project root"; exit 1; }

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

# Using ModelScope SDK or HuggingFace CLI depending on availability
# If ModelScope runs HuggingFace models, we can use hf-cli, or we can use `modelscope download` if available.
# Let's try HuggingFace CLI first as it's more universal.
huggingface-cli download RunsenXu/MMSI-Bench --local-dir "$MMSI_DATA_DIR" --repo-type dataset --resume-download

echo "MMSI Data setup complete!"
echo "Check '$TARGET_DIR/$MMSI_DATA_DIR' for downloaded files."

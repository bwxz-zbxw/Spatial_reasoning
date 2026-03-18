#!/bin/bash

# Configuration
GCA_REPO_URL="https://github.com/Zx55/GCA.git"
TARGET_DIR="gca"

# Ensure we are in the project root
cd "$(dirname "$0")/../.." || { echo "Error: Could not navigate to project root"; exit 1; }

echo "Current directory: $(pwd)"

# 1. Clone GCA if not present
if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning GCA repository into $TARGET_DIR..."
    git clone "$GCA_REPO_URL" "$TARGET_DIR"
else
    echo "GCA directory '$TARGET_DIR' already exists. Pulling latest changes..."
    cd "$TARGET_DIR" && git pull && cd ..
fi

# Navigate into GCA directory
cd "$TARGET_DIR" || { echo "Error: Failed to enter $TARGET_DIR"; exit 1; }

echo "Setting up GCA environment in $(pwd)..."

# 2. Python Environment Setup
echo "Installing PyTorch 2.5.1+..."
pip install torch>=2.5.1 torchvision>=0.20.1 torchaudio>=2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "Installing GCA requirements..."
pip install -r requirements/gca.txt

# 3. Third-party Tools Setup
echo "Setting up third-party tools..."
mkdir -p tools/third_party
cd tools/third_party

# 3.1 VGGT
if [ ! -d "vggt" ]; then
    echo "Cloning VGGT..."
    git clone --depth=1 https://github.com/facebookresearch/vggt.git
    cd vggt
    pip install .
    cd ..
else
    echo "VGGT already exists."
fi

# 3.2 SAM2
if [ ! -d "sam2" ]; then
    echo "Cloning SAM2..."
    git clone --depth=1 https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install .
    
    echo "Downloading SAM2 checkpoints..."
    cd checkpoints
    ./download_ckpts.sh
    cd ../..
else
    echo "SAM2 already exists."
fi

# 3.3 Orient-Anything
if [ ! -d "Orient-Anything" ]; then
    echo "Cloning Orient-Anything..."
    git clone --depth=1 https://github.com/SpatialVision/Orient-Anything.git
else
    echo "Orient-Anything already exists."
fi

# 3.4 MoGe
if [ ! -d "MoGe" ]; then
    echo "Cloning MoGe..."
    git clone --depth=1 https://github.com/microsoft/MoGe.git
else
    echo "MoGe already exists."
fi

cd ../.. # Back to gca root

echo "GCA Environment Setup Complete!"

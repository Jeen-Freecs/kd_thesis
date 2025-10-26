#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Installing Confidence-Aware Knowledge Distillation Environment ==="

# Check if conda is already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
else
    echo "Conda already installed, skipping..."
fi

# Create & activate the Conda environment
echo "Creating conda environment 'kd-env' with Python 3.11..."
conda create -n kd-env python=3.11 -y

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kd-env

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "Warning: nvidia-smi not found. Defaulting to CUDA 11.8"
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
# For CUDA 11.8 (most common on vast.ai)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1, uncomment below and comment above:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Install ipykernel for Jupyter notebook support
echo "Setting up Jupyter kernel..."
conda install ipykernel --update-deps --force-reinstall -y
python -m ipykernel install --user --name kd-env --display-name "Python (KD-Env)"

conda deactivate

echo ""
echo "=== Installation Complete! ==="
echo "To activate the environment, run:"
echo "  conda activate kd-env"
echo ""
echo "To verify CUDA is available in PyTorch, run:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
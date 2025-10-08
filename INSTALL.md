# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- 10GB disk space

### Recommended Requirements
- Python 3.9+
- CUDA-capable GPU with 8GB+ VRAM
- 16GB RAM
- 20GB disk space
- CUDA 11.7+ and cuDNN

## Installation Methods

### Method 1: pip (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation.git
cd Confidence-Aware-Ensemble-Knowledge-Distillation
```

2. **Create virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n kd-env python=3.9
conda activate kd-env
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install in development mode** (optional):
```bash
pip install -e .
```

### Method 2: Conda

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation.git
cd Confidence-Aware-Ensemble-Knowledge-Distillation
```

2. **Create conda environment**:
```bash
conda create -n kd-env python=3.9
conda activate kd-env
```

3. **Install PyTorch** (GPU version):
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

4. **Install other dependencies**:
```bash
pip install -r requirements.txt
```

### Method 3: Docker (Coming Soon)

```bash
# Build image
docker build -t kd-cifar100 .

# Run container
docker run --gpus all -it kd-cifar100
```

## Verify Installation

### Test imports:
```python
python -c "import torch; import pytorch_lightning; print('Success!')"
```

### Check CUDA availability:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run a quick test:
```bash
python -c "
from src.data import CIFAR100DataModule
from src.models import create_student_model
print('All imports successful!')
"
```

## Setup Weights & Biases

1. **Create account**: Go to [wandb.ai](https://wandb.ai) and sign up

2. **Login**:
```bash
wandb login
```

3. **Enter your API key** when prompted

4. **Test connection**:
```bash
python -c "import wandb; wandb.init(project='test'); wandb.finish()"
```

## Download CIFAR-100

The dataset will be automatically downloaded on first run. To pre-download:

```python
from torchvision import datasets
datasets.CIFAR100(root='./data', train=True, download=True)
datasets.CIFAR100(root='./data', train=False, download=True)
```

## GPU Setup

### NVIDIA GPU

1. **Check CUDA version**:
```bash
nvidia-smi
```

2. **Install correct PyTorch**:
Visit [pytorch.org](https://pytorch.org) and select your CUDA version

### AMD GPU (ROCm)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### Apple Silicon (M1/M2)

PyTorch with MPS (Metal Performance Shaders) acceleration:
```bash
pip install torch torchvision
```

The code automatically detects and uses MPS when available.

## Common Installation Issues

### Issue 1: CUDA version mismatch

**Error**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**: Install PyTorch for your specific CUDA version:
```bash
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Out of memory during installation

**Error**: `MemoryError` during pip install

**Solution**: Install packages one by one:
```bash
pip install torch torchvision
pip install pytorch-lightning
pip install timm wandb
pip install -r requirements.txt
```

### Issue 3: timm installation fails

**Solution**: Update pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install timm
```

### Issue 4: WandB login fails

**Solution**: Use offline mode:
```bash
export WANDB_MODE=offline
# Or add to config: wandb.init(mode="offline")
```

## Post-Installation

### 1. Verify GPU is detected:
```bash
python scripts/train.py --config configs/baseline_config.yaml --max-epochs 1
```

### 2. Test all components:
```bash
# Test data loading
python -c "from src.data import CIFAR100DataModule; dm = CIFAR100DataModule(); print('Data OK')"

# Test models
python -c "from src.models import create_student_model; m = create_student_model(); print('Models OK')"

# Test training
python -c "from src.training import train_kd_model; print('Training OK')"
```

### 3. Run demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Deactivate environment
deactivate  # or: conda deactivate

# Remove virtual environment
rm -rf venv  # or: conda env remove -n kd-env

# Remove repository
cd ..
rm -rf Confidence-Aware-Ensemble-Knowledge-Distillation
```

## Getting Help

If you encounter issues:

1. Check [GitHub Issues](https://github.com/yourusername/repo/issues)
2. Review [Troubleshooting Guide](README.md#troubleshooting)
3. Join our [Discord/Slack] (if available)
4. Email: your.email@example.com

## Next Steps

After installation:

1. Read the [Quick Start Guide](USAGE.md)
2. Review [Configuration Options](README.md#configuration)
3. Run your first experiment: `python scripts/train.py --config configs/baseline_config.yaml`
4. Explore the [demo notebook](notebooks/demo.ipynb)

Happy Training! 🚀


# Confidence-Aware Ensemble Knowledge Distillation

Complete implementation of multi-teacher knowledge distillation methods for CIFAR-100 image classification.

## 🎯 Overview

This project implements **three knowledge distillation methods** plus baseline training, achieving up to **74.24% accuracy** on CIFAR-100:

- **Baseline**: Student (MobileNetV2) from scratch → 69.84%
- **Method 1**: CA-WKD (Confidence-Aware Weighted KD) → 73.92%
- **Method 2**: α-Guided CA-WKD (Dynamic gating) → **74.24%** ⭐ (best overall)
- **Method 3**: Adaptive α-Guided KD → 73.26%
- **Single Teacher ViT**: → 73.19%

### Key Features

- ✅ **12 Pre-configured Experiments** - All experiments from research paper
- ✅ **Automated Setup** - One-command installation for vast.ai/Linux
- ✅ **Weights & Biases Integration** - Real-time experiment tracking
- ✅ **Multiple Teacher Support** - ResNet, DenseNet, ViT ensembles
- ✅ **PAT Framework** - Heterogeneous KD for CNN ↔ ViT
- ✅ **PyTorch Lightning** - Modern training framework
- ✅ **Production Ready** - Tested on vast.ai GPU instances

---

## 🚀 Quick Start (5 Minutes)

### On vast.ai or Linux with CUDA

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Confidence-Aware-Ensemble-Knowledge-Distillation

# 2. Install everything (takes 5 minutes)
bash install.sh

# 3. Activate environment
conda activate kd-env

# 4. Login to wandb (get API key from https://wandb.ai/authorize)
wandb login

# 5. Start training best model!
python scripts/train.py --config configs/method2_diverse_ensemble.yaml
```

**That's it!** Training starts immediately. View results at https://wandb.ai

### Run All 12 Experiments Automatically

```bash
# Runs all experiments sequentially (70-90 hours)
bash run_all_experiments.sh

# Or run in background
nohup ./run_all_experiments.sh > experiments.log 2>&1 &
tail -f experiments.log
```

---

## 📊 All Experiments

### Complete Experiment List (Evaluated on Test Set)

| Config File | Method | Teacher(s) | Accuracy | Training Time |
|-------------|--------|------------|----------|---------------|
| `baseline_config.yaml` | Baseline | None | 69.84% | 2-3h |
| `single_teacher_densenet.yaml` | Single | DenseNet-121 | 71.99% | 4-6h |
| `single_teacher_vit.yaml` | Single | ViT | 73.19% | 6-8h |
| `single_teacher_resnet50.yaml` | Single | ResNet-50 | 72.81% | 4-6h |
| `method1_ca_wkd.yaml` | Method 1 | R+D+V | 73.92% | 8-10h |
| `method2_diverse_ensemble.yaml` | Method 2 | R+D+V (γ=10) | **74.24%** ⭐ | 8-10h |
| `method2_dynamic_kd.yaml` | Method 2 | 3 ResNets (γ=5) | 72.53% | 6-8h |
| `method3_densenet_resnet_temp8.yaml` | Method 3 | D+R (T=8) | 72.72% | 6-8h |
| `method3_densenet_resnet_temp16.yaml` | Method 3 | D+R (T=16) | 73.26% | 6-8h |
| `method3_adaptive_alpha.yaml` | Method 3 | 3 ResNets + CE | 71.60% | 6-8h |
| `method3_diverse_ensemble.yaml` | Method 3 | R+D+V | 72.34% | 8-10h |
| `method3_3resnets_no_ce.yaml` | Method 3 | 3 ResNets (no CE) | 72.12% | 6-8h |

**Legend**: R=ResNet, D=DenseNet, V=ViT, T=Temperature, γ=Gamma

### Individual Experiment Commands

```bash
# Baseline
python scripts/train.py --config configs/baseline_config.yaml

# Best performers
python scripts/train.py --config configs/method2_diverse_ensemble.yaml  # 74.24% ⭐
python scripts/train.py --config configs/method1_ca_wkd.yaml             # 73.92%
python scripts/train.py --config configs/method3_densenet_resnet_temp16.yaml  # 73.26%

# All single teachers
python scripts/train.py --config configs/single_teacher_vit.yaml        # 73.19%
python scripts/train.py --config configs/single_teacher_resnet50.yaml   # 72.81%
python scripts/train.py --config configs/single_teacher_densenet.yaml   # 71.99%

# All methods
python scripts/train.py --config configs/method2_dynamic_kd.yaml
python scripts/train.py --config configs/method3_densenet_resnet_temp8.yaml
python scripts/train.py --config configs/method3_adaptive_alpha.yaml
python scripts/train.py --config configs/method3_diverse_ensemble.yaml
python scripts/train.py --config configs/method3_3resnets_no_ce.yaml
```

---

## 💻 Installation

### Prerequisites

- Python 3.11
- CUDA-capable GPU (≥12GB VRAM recommended)
- 50GB disk space
- Internet connection

### Automated Installation (Recommended)

```bash
bash install.sh
```

**What this installs**:
- Miniconda (if not present)
- Python 3.11 environment (`kd-env`)
- PyTorch with CUDA 11.8 support
- All dependencies
- Jupyter kernel

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# 1. Create conda environment
conda create -n kd-env python=3.11 -y
conda activate kd-env

# 2. Install PyTorch with CUDA
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup wandb
wandb login
```
</details>

---

## 🌐 Running on vast.ai

### Step 1: Rent GPU Instance

1. Go to [vast.ai](https://vast.ai)
2. Select instance with:
   - GPU: RTX 3090/4090 or A6000
   - VRAM: ≥12GB
   - CUDA: 11.8 or 12.1
3. Connect via SSH

### Step 2: Setup & Train

```bash
# Upload project
cd /workspace
git clone <your-repo-url>
cd Confidence-Aware-Ensemble-Knowledge-Distillation

# Install (5 minutes)
bash install.sh

# Activate & login
conda activate kd-env
wandb login

# Start training (best model)
python scripts/train.py --config configs/method2_diverse_ensemble.yaml
```

### Keep Training Running (If Connection Drops)

```bash
# Use tmux to keep training alive
tmux new -s training
python scripts/train.py --config configs/...

# Detach: Press Ctrl+B, then D

# Reconnect later
tmux attach -t training
```

### Cost Estimate

| GPU | Price/Hour | All 12 Experiments |
|-----|------------|--------------------|
| RTX 3090 | $0.20-0.40 | $15-30 |
| RTX 4090 | $0.40-0.60 | $30-50 |

Total time: 70-90 hours on single GPU

---

## 📈 Weights & Biases Integration

### Setup

```bash
# Get API key from https://wandb.ai/authorize
wandb login
```

### What Gets Logged

- ✅ Training/validation loss curves
- ✅ Accuracy and AUROC metrics
- ✅ Learning rate schedule
- ✅ Teacher confidence scores
- ✅ Model checkpoints
- ✅ Hyperparameters

### View Results

Go to https://wandb.ai/your-username to see:
- Real-time training progress
- Comparison across experiments
- Interactive charts
- Model artifacts

### Offline Mode

If no internet:
```bash
export WANDB_MODE=offline
python scripts/train.py --config configs/...

# Sync later
wandb sync --sync-all
```

### Custom Project Name

Edit any config file:
```yaml
wandb:
  project: "My-Custom-Project"
  name: "Experiment-1"
```

---

## ⚙️ Configuration

All experiments are controlled via YAML files in `configs/` directory.

### Config Structure

```yaml
# Data configuration
data:
  data_dir: "./data"
  batch_size: 128  # Reduce to 64 if out of memory
  num_workers: 4
  val_size: 5000
  seed: 42

# Model configuration
model:
  num_classes: 100
  student_name: "mobilenetv2_100"
  teacher_names:
    - "resnet50_cifar100"
    - "densenet121_cifar100"

# Knowledge Distillation configuration
kd:
  type: "confidence"  # or "ca_weighted", "dynamic", "baseline"
  temperature: 4.0
  learning_rate: 0.01

# Training configuration
training:
  max_epochs: 150
  patience: 30  # Early stopping

# Weights & Biases configuration
wandb:
  project: "KD-CIFAR100"
  name: "experiment-name"
```

### Adjusting for Your GPU

#### Small GPU (8GB VRAM)
```yaml
data:
  batch_size: 64
  num_workers: 2
```

#### Large GPU (24GB+ VRAM)
```yaml
data:
  batch_size: 256
  num_workers: 8
```

---

## 🧠 Knowledge Distillation Methods

### Method 1: CA-WKD (Confidence-Aware Weighted KD)

**How it works**:
- Weights teachers by their performance (lower loss → higher weight)
- Equal contribution from KD and ground truth
- Simple, no hyperparameter tuning needed

**Formula**: `L_total = L_KL + L_CE`

**Use when**: You want a simple multi-teacher baseline

```bash
python scripts/train.py --config configs/method1_ca_wkd.yaml
```

### Method 2: α-Guided CA-WKD (Dynamic Gating)

**How it works**:
- Dynamic weighting like Method 1
- Adaptive gating: balances KD vs ground truth based on teacher confidence
- High teacher confidence → More KD loss
- Low teacher confidence → More ground truth loss

**Formula**: `L_total = α * L_KL + (1-α) * L_CE` where α = sigmoid(γ * (conf - θ))

**Use when**: You have time for hyperparameter tuning (γ and θ)

```bash
python scripts/train.py --config configs/method2_diverse_ensemble.yaml
```

### Method 3: Adaptive α-Guided KD (Most Confident Teacher)

**How it works**:
- Selects the most confident teacher for each sample
- Uses teacher confidence as α (no manual tuning!)
- Different teachers selected for different samples

**Formula**: `L_total = α * L_KL* + (1-α) * L_CE` where α = mean(max_k p_Tk(y|x))

**Use when**: You want automatic teacher selection without tuning

```bash
python scripts/train.py --config configs/method3_densenet_resnet_temp16.yaml  # 73.26%
```

### Single Teacher (Simplest)

**How it works**:
- One strong teacher (ViT, ResNet, or DenseNet)
- Fixed α = 0.25-0.75 (tuned per teacher)
- Simpler than multi-teacher

**Use when**: You want simplicity with good accuracy

```bash
python scripts/train.py --config configs/single_teacher_vit.yaml  # 73.19%
```

---

## 📊 Results & Analysis

### Key Findings

1. **Method 2 (Dynamic α-Guided) is best overall**: 74.24% with diverse ensemble (γ=10)
2. **Method 1 (CA-WKD) is second best**: 73.92% with R+D+V ensemble
3. **Single ViT outperforms DenseNet**: ViT (73.19%) > ResNet (72.81%) > DenseNet (71.99%)
4. **Temperature 16 better than 8**: T=16 (73.26%) > T=8 (72.72%) for Method 3
5. **Dynamic gating (Method 2) > Static confidence (Method 3)**: Adaptive α helps

### Performance Comparison

| Approach | Accuracy | Complexity | Training Time |
|----------|----------|------------|---------------|
| Baseline | 69.84% | Simplest | 2-3h |
| Single Teacher (ViT) | 73.19% | Simple | 6-8h |
| Multi-Teacher (Method 2) | **74.24%** ⭐ | Medium | 8-10h |

**Recommendation**: Use Method 2 with diverse ensemble (γ=10) for best accuracy

---

## 🔧 Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
data:
  batch_size: 64  # or 32
```

**Solution 2**: Use gradient accumulation (add to trainer.py)
```python
accumulate_grad_batches=2
```

### Teacher Download Fails

**Solution**: Pre-download teachers
```bash
python -c "
import torch
urls = [
    'https://huggingface.co/edadaltocg/resnet50_cifar100/resolve/main/pytorch_model.bin',
    'https://huggingface.co/edadaltocg/densenet121_cifar100/resolve/main/pytorch_model.bin',
]
for url in urls:
    name = url.split('/')[-3]
    torch.hub.load_state_dict_from_url(url, file_name=f'{name}.pth')
    print(f'Downloaded {name}')
"
```

### wandb Authentication Fails

**Solution**:
```bash
wandb login --relogin
# Or use offline mode
export WANDB_MODE=offline
```

### Training is Slow

**Check**:
```bash
nvidia-smi  # Verify GPU is being used
```

**If on CPU**: Make sure PyTorch with CUDA is installed
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Connection Lost to vast.ai

**Solution**: Always use `tmux`
```bash
tmux new -s training
python scripts/train.py --config configs/...
# Detach: Ctrl+B then D
```

---

## 📁 Project Structure

```
Confidence-Aware-Ensemble-Knowledge-Distillation/
├── README.md                    # This file
├── install.sh                   # Automated installation
├── run_all_experiments.sh       # Run all 12 experiments
├── requirements.txt             # Python dependencies
│
├── configs/                     # 12 experiment configs
│   ├── baseline_config.yaml
│   ├── single_teacher_*.yaml (3 files)
│   ├── method1_ca_wkd.yaml
│   ├── method2_*.yaml (2 files)
│   └── method3_*.yaml (5 files)
│
├── src/                         # Source code
│   ├── data/                    # Data loading
│   ├── models/                  # Student, Teacher, KD modules
│   ├── training/                # Training utilities
│   ├── evaluation/              # Evaluation
│   └── utils/                   # Config, logging
│
├── scripts/                     # Executable scripts
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── experiment.py            # Experiment runner
│
├── notebooks/                   # Jupyter notebooks
└── logs/                       # Training logs (auto-created)
```

---

## 🎯 Recommended Workflow

### Day 1: Setup & Baseline (3-4 hours)

```bash
# Setup
bash install.sh
conda activate kd-env
wandb login

# Test with baseline
python scripts/train.py --config configs/baseline_config.yaml
```

### Day 2: Best Experiments (10-14 hours)

```bash
# Best overall (Method 2)
python scripts/train.py --config configs/method2_diverse_ensemble.yaml  # 74.24%

# Best single teacher
python scripts/train.py --config configs/single_teacher_vit.yaml  # 73.19%
```

### Day 3-4: Full Ablation Study

```bash
# All experiments
bash run_all_experiments.sh
```

### Day 5: Analysis

- Download checkpoints from `lightning_logs/`
- Analyze results on wandb.ai
- Compare methods
- Write report

---

## 📈 Monitoring Training

### Terminal Output

```
Epoch 15/150: 100%|████████| 390/390 [02:15<00:00]
train/loss=1.234, val/acc=0.723, val/auroc=0.896
```

### Weights & Biases Dashboard

Open https://wandb.ai to see:
- Live loss curves
- Accuracy progression
- Learning rate schedule
- Teacher confidence scores
- GPU utilization

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Check GPU is being used
nvidia-smi
```

---

## 🚀 Advanced Usage

### Resume Training from Checkpoint

```bash
python scripts/train.py \
    --config configs/single_teacher_densenet.yaml \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt
```

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
    --config configs/single_teacher_densenet.yaml \
    --split test
```

### Custom Experiment

```bash
python scripts/experiment.py \
    --config configs/method3_densenet_resnet_temp8.yaml \
    --exp-name "my-custom-experiment"
```

### Modify Hyperparameters

Create new config file or edit existing:
```yaml
kd:
  temperature: 8.0  # Try different temperatures
  learning_rate: 0.005  # Try different learning rates

training:
  max_epochs: 200  # Train longer
  patience: 40  # More patience
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{confidence-aware-kd-2024,
  title={Confidence-Aware Ensemble Knowledge Distillation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation}
}
```

---

## 📧 Support

For issues or questions:
- Check troubleshooting section above
- Open an issue on GitHub
- Email: your.email@example.com

---

## ✅ Quick Command Reference

```bash
# Setup
bash install.sh && conda activate kd-env && wandb login

# Best overall (Method 2 - 74.24%)
python scripts/train.py --config configs/method2_diverse_ensemble.yaml

# Best single teacher (ViT - 73.19%)
python scripts/train.py --config configs/single_teacher_vit.yaml

# All experiments
bash run_all_experiments.sh

# Monitor
tail -f logs/train.log
watch -n 1 nvidia-smi

# Evaluate
python scripts/evaluate.py --checkpoint <path> --config <config>
```

---

## 📈 Final Results Summary

| Rank | Method | Config | Test Accuracy |
|------|--------|--------|---------------|
| 🥇 1 | Method 2 (Dynamic, γ=10) | `method2_diverse_ensemble.yaml` | **74.24%** |
| 🥈 2 | Method 1 (CA-WKD) | `method1_ca_wkd.yaml` | 73.92% |
| 🥉 3 | Method 3 (T=16) | `method3_densenet_resnet_temp16.yaml` | 73.26% |
| 4 | Single ViT (α=0.25) | `single_teacher_vit.yaml` | 73.19% |
| 5 | Single ResNet (α=0.75) | `single_teacher_resnet50.yaml` | 72.81% |
| 6 | Method 3 (T=8) | `method3_densenet_resnet_temp8.yaml` | 72.72% |
| 7 | Method 2 (γ=5) | `method2_dynamic_kd.yaml` | 72.53% |
| 8 | Method 3 (Diverse) | `method3_diverse_ensemble.yaml` | 72.34% |
| 9 | Method 3 (NoCE) | `method3_3resnets_no_ce.yaml` | 72.12% |
| 10 | Single DenseNet (α=0.50) | `single_teacher_densenet.yaml` | 71.99% |
| 11 | Method 3 (WithCE) | `method3_adaptive_alpha.yaml` | 71.60% |
| 12 | Baseline | `baseline_config.yaml` | 69.84% |

---

**🎯 You're ready to train! Start with:**
```bash
bash install.sh
conda activate kd-env
wandb login
python scripts/train.py --config configs/method2_diverse_ensemble.yaml
```

**Happy Training! 🚀**

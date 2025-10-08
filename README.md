# Confidence-Aware Ensemble Knowledge Distillation

A professional implementation of **Confidence-Aware Ensemble Knowledge Distillation** for image classification on CIFAR-100. This project implements two novel knowledge distillation strategies that dynamically adjust the learning process based on teacher model confidence.

## 🎯 Overview

This project implements advanced knowledge distillation techniques where multiple teacher models guide a lightweight student model. The key innovation is the dynamic adjustment of knowledge transfer based on teacher confidence and performance.

### Key Features

- **Two KD Strategies**:
  - **Dynamic KD with Weighted Ensemble**: Dynamically weights teachers based on their performance and uses confidence-based gating
  - **Confidence-Based KD**: Selects the most confident teacher per sample and uses confidence as the mixing weight
  
- **Modular Architecture**: Professional, maintainable codebase with clear separation of concerns
- **Multiple Teacher Support**: Ensemble of ResNet, DenseNet, ViT, and other models
- **Experiment Tracking**: Full integration with Weights & Biases
- **Flexible Configuration**: YAML-based configuration management
- **PyTorch Lightning**: Modern training framework with automatic optimization

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Model Architectures](#model-architectures)
- [Knowledge Distillation Methods](#knowledge-distillation-methods)
- [Results](#results)
- [Advanced Usage](#advanced-usage)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation.git
cd Confidence-Aware-Ensemble-Knowledge-Distillation
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install the package** (optional):
```bash
pip install -e .
```

5. **Setup Weights & Biases** (for experiment tracking):
```bash
wandb login
```

## 📁 Project Structure

```
Confidence-Aware-Ensemble-Knowledge-Distillation/
├── configs/                      # Configuration files
│   ├── config.yaml              # Main KD configuration
│   ├── baseline_config.yaml     # Baseline (no KD) configuration
│   └── confidence_config.yaml   # Confidence-based KD configuration
│
├── src/                         # Source code
│   ├── data/                    # Data loading and preprocessing
│   │   ├── datamodule.py       # PyTorch Lightning DataModule
│   │   └── transforms.py       # Custom transforms
│   │
│   ├── models/                  # Model architectures
│   │   ├── student.py          # Student model creation
│   │   ├── teacher.py          # Teacher model loading
│   │   └── kd_module.py        # KD Lightning modules
│   │
│   ├── training/               # Training utilities
│   │   └── trainer.py          # Training logic
│   │
│   ├── evaluation/             # Evaluation utilities
│   │   └── evaluator.py        # Model evaluation
│   │
│   └── utils/                  # Helper utilities
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
│
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── experiment.py          # Experiment runner
│
├── notebooks/                 # Jupyter notebooks (optional)
├── logs/                     # Training logs
├── data/                     # Dataset directory (auto-created)
└── requirements.txt          # Python dependencies
```

## 🏃 Quick Start

### 1. Train Baseline Model (No Knowledge Distillation)

Train a student model from scratch without knowledge distillation:

```bash
python scripts/train.py --config configs/baseline_config.yaml
```

### 2. Train with Dynamic KD

Train with weighted ensemble and dynamic gating:

```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Train with Confidence-Based KD

Train with most-confident teacher selection:

```bash
python scripts/train.py --config configs/confidence_config.yaml
```

### 4. Evaluate a Trained Model

```bash
python scripts/evaluate.py \
    --checkpoint path/to/checkpoint.ckpt \
    --config configs/config.yaml \
    --split test
```

### 5. Run a Custom Experiment

```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name my_experiment
```

## ⚙️ Configuration

All experiments are controlled via YAML configuration files. Here's a breakdown of the main sections:

### Data Configuration

```yaml
data:
  data_dir: "./data"              # Dataset directory
  batch_size: 128                 # Training batch size
  num_workers: 4                  # Data loader workers
  val_size: 5000                  # Validation set size
  seed: 42                        # Random seed
  teacher_model_types: ["resnet", "resnet", "resnet"]  # Teacher transform types
  student_model_type: "mobilenet" # Student transform type
```

### Model Configuration

```yaml
model:
  num_classes: 100
  student_name: "mobilenetv2_100"  # Student architecture
  student_pretrained: false
  teacher_names:                    # Teacher models from timm
    - "resnet50_cifar100"
    - "resnet18_cifar100"
    - "resnet34_cifar100"
```

### Knowledge Distillation Configuration

```yaml
kd:
  type: "dynamic"                  # "dynamic" or "confidence"
  temperature: 4.0                 # Softmax temperature
  gamma: 10.0                      # Gating function scaling
  threshold: 0.5                   # Confidence threshold
  alpha: 0.5                       # Base mixing weight
  learning_rate: 0.01
  use_soft_loss: true             # Enable KD loss
  use_hard_loss: false            # Enable CE loss
```

### Training Configuration

```yaml
training:
  max_epochs: 150
  patience: 30                     # Early stopping patience
  log_every_n_steps: 50
```

## 🔬 Running Experiments

### Experiment 1: Baseline Performance

Measure student performance without distillation:

```bash
python scripts/experiment.py \
    --config configs/baseline_config.yaml \
    --exp-name baseline_mobilenet
```

### Experiment 2: Single Teacher KD

Modify config to use one teacher:

```yaml
model:
  teacher_names:
    - "resnet50_cifar100"
```

```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name single_teacher_kd
```

### Experiment 3: Multi-Teacher Ensemble

Use multiple teachers with dynamic weighting:

```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name multi_teacher_ensemble
```

### Experiment 4: Confidence-Based Selection

Use the most confident teacher:

```bash
python scripts/experiment.py \
    --config configs/confidence_config.yaml \
    --exp-name confidence_based_kd
```

### Resume Training from Checkpoint

```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name resumed_experiment \
    --checkpoint path/to/checkpoint.ckpt
```

## 🏗️ Model Architectures

### Student Models

- **MobileNetV2**: Lightweight CNN (Default)
- **EfficientNet-B0**: Efficient scaled architecture
- Custom architectures via `timm` library

### Teacher Models (Pretrained on CIFAR-100)

- **ResNet-18/34/50**: Deep residual networks
- **DenseNet-121**: Densely connected networks
- **Vision Transformer (ViT)**: Transformer-based architecture

All teacher models are automatically frozen during training.

## 📚 Knowledge Distillation Methods

### 1. Dynamic KD with Weighted Ensemble

**Key Components**:

1. **Dynamic Teacher Weighting**: Each teacher receives a weight based on its cross-entropy loss:
   ```
   w_k = (1/(K-1)) * [1 - exp(L_CE^k) / Σ_j exp(L_CE^j)]
   ```

2. **Confidence-Based Gating**: Automatically balances soft (KD) and hard (CE) losses:
   ```
   gate = sigmoid(γ * (conf - threshold))
   L_total = gate * L_KD + (1-gate) * α * L_CE
   ```

3. **Temperature Scaling**: Softens probability distributions for knowledge transfer

**When to Use**: 
- Multiple diverse teachers available
- Want automatic balancing of KD and CE losses
- Need per-sample adaptive learning

### 2. Confidence-Based KD

**Key Components**:

1. **Most Confident Teacher Selection**: For each sample, selects the teacher with highest probability on the true class

2. **Dynamic Alpha**: Uses teacher confidence as mixing weight:
   ```
   α = mean(max_teacher_confidence)
   L_total = α * L_KD + (1-α) * L_CE
   ```

3. **Teacher Usage Tracking**: Logs which teacher is selected for each sample

**When to Use**:
- Teachers have varying expertise across different classes
- Want to leverage the best teacher for each sample
- Need interpretable teacher selection

### Loss Functions

- **Soft Loss (L_KD)**: KL Divergence between student and teacher distributions
- **Hard Loss (L_CE)**: Cross-entropy with ground truth labels
- **Temperature**: Controls smoothness of probability distributions (typically 3-5)

## 📊 Results

### Expected Performance on CIFAR-100

| Method | Accuracy | Parameters |
|--------|----------|------------|
| MobileNetV2 (Baseline) | ~68-70% | 3.5M |
| Single Teacher KD | ~70-72% | 3.5M |
| Multi-Teacher Dynamic KD | ~72-74% | 3.5M |
| Confidence-Based KD | ~71-73% | 3.5M |
| ResNet-50 Teacher | ~80-81% | 25M |

*Note: Results may vary based on hyperparameters and training duration*

### Monitoring Training

View real-time metrics in Weights & Biases:
- Training/validation loss
- Accuracy and AUROC
- Teacher confidence and gate values
- Teacher usage distribution (confidence-based)
- Learning rate schedule

## 🔧 Advanced Usage

### Custom Teacher Models

Add your own pretrained teacher:

```python
from src.models.teacher import create_teacher_models

# Define custom teacher loading
def load_custom_teacher(num_classes, device):
    model = YourCustomModel(num_classes=num_classes)
    model.load_state_dict(torch.load('path/to/weights.pth'))
    return model

# Modify teacher.py to include your model
```

### Custom Student Architecture

```python
from src.models.student import create_student_model

student = create_student_model(
    model_name='efficientnet_b0',  # Any timm model
    num_classes=100,
    pretrained=False
)
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Temperature** (1-10): Higher = softer distributions
2. **Gamma** (1-20): Controls gating sensitivity
3. **Threshold** (0.3-0.7): Confidence threshold for gating
4. **Learning Rate** (1e-4 to 1e-2): Adjust based on model size
5. **Alpha** (0.1-0.9): Base mixing weight

### Multi-GPU Training

Modify trainer configuration:

```python
trainer = pl.Trainer(
    devices=4,  # Number of GPUs
    accelerator='gpu',
    strategy='ddp'  # Distributed Data Parallel
)
```

### Mixed Precision Training

Automatically enabled for GPU training with `precision='16-mixed'` in the trainer.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Reduce number of teachers

2. **Teacher Download Fails**:
   - Check internet connection
   - Manually download models to `~/.cache/torch/hub/checkpoints/`

3. **WandB Login Issues**:
   - Run `wandb login` and enter your API key
   - Or set `WANDB_MODE=offline` for local logging

### Debug Mode

Enable verbose logging:

```python
python scripts/train.py --config configs/config.yaml --log-level DEBUG
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{confidence-aware-kd-2024,
  title={Confidence-Aware Ensemble Knowledge Distillation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## 🙏 Acknowledgments

- PyTorch Lightning for the training framework
- TIMM for pretrained models
- Weights & Biases for experiment tracking
- CIFAR-100 dataset creators

---

**Happy Training! 🚀**

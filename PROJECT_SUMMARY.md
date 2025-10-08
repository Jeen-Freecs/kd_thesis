# Project Summary: Confidence-Aware Ensemble Knowledge Distillation

## Overview

This project implements two novel knowledge distillation strategies for training lightweight student models on CIFAR-100:

1. **Dynamic KD with Weighted Ensemble**: Dynamically weights multiple teachers and uses confidence-based gating
2. **Confidence-Based KD**: Selects the most confident teacher per sample

## Architecture Highlights

### Modular Design
- **Data Module**: Handles CIFAR-100 with custom transforms for multiple teachers/students
- **Model Zoo**: Flexible teacher/student creation with TIMM integration
- **KD Modules**: Two Lightning modules implementing different KD strategies
- **Training Pipeline**: Automated training with early stopping and checkpointing
- **Evaluation Tools**: Comprehensive model evaluation utilities

### Key Components

#### 1. Data Processing (`src/data/`)
- `DualTransformDataset`: Applies different transforms to teachers and students
- `CIFAR100DataModule`: PyTorch Lightning DataModule with stratified splits
- Model-specific transforms (ResNet, DenseNet, ViT, MobileNet)

#### 2. Models (`src/models/`)
- **Student Models**: MobileNetV2, EfficientNet, etc. (via TIMM)
- **Teacher Models**: Pretrained ResNet/DenseNet/ViT on CIFAR-100
- **DynamicKDLitModule**: Weighted ensemble with gating mechanism
- **ConfidenceBasedKDLitModule**: Best teacher selection with dynamic alpha

#### 3. Training (`src/training/`)
- Automated training pipeline with WandB logging
- Mixed precision training support
- Early stopping and model checkpointing
- Flexible configuration system

#### 4. Evaluation (`src/evaluation/`)
- Model evaluation on test/validation sets
- Checkpoint loading utilities
- Metrics: Accuracy, AUROC, Loss

## KD Strategies Explained

### Dynamic KD (Weighted Ensemble + Gating)

**Teacher Weighting**:
```
w_k = (1/(K-1)) * [1 - exp(L_CE^k) / Σ_j exp(L_CE^j)]
```
- Teachers with lower CE loss get higher weights
- Weights are computed per-sample

**Confidence Gating**:
```
gate = sigmoid(γ * (avg_conf - threshold))
L_total = gate * L_KD + (1-gate) * α * L_CE
```
- High teacher confidence → More KD loss
- Low teacher confidence → More hard label loss

### Confidence-Based KD (Best Teacher)

**Teacher Selection**:
- For each sample, select teacher with highest probability on true class
- Use that teacher's logits for KD

**Dynamic Alpha**:
```
α = mean(max_teacher_conf)
L_total = α * L_KD + (1-α) * L_CE
```
- Higher confidence → More KD
- Lower confidence → More hard labels

## Configuration System

Three main configs provided:

1. **baseline_config.yaml**: Student only (no KD)
2. **config.yaml**: Dynamic KD with multiple teachers
3. **confidence_config.yaml**: Confidence-based KD

Key parameters:
- `temperature`: Softmax temperature (3-5 typical)
- `gamma`: Gating sensitivity (5-15 typical)
- `threshold`: Confidence threshold (0.3-0.7 typical)
- `alpha`: Base mixing weight (0.3-0.7 typical)

## Usage Patterns

### Basic Training
```bash
# Baseline
python scripts/train.py --config configs/baseline_config.yaml

# Dynamic KD
python scripts/train.py --config configs/config.yaml

# Confidence KD
python scripts/train.py --config configs/confidence_config.yaml
```

### Running Experiments
```bash
python scripts/experiment.py --config configs/config.yaml --exp-name exp1
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint model.ckpt --config configs/config.yaml
```

## Project Structure

```
├── configs/              # YAML configurations
├── src/
│   ├── data/            # Data loading & transforms
│   ├── models/          # Model architectures & KD modules
│   ├── training/        # Training utilities
│   ├── evaluation/      # Evaluation utilities
│   └── utils/           # Config & logging helpers
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter demos
├── requirements.txt     # Dependencies
├── setup.py            # Package setup
└── README.md           # Full documentation
```

## Key Features

✅ **Modular & Professional**: Clean architecture, easy to extend  
✅ **Multiple KD Strategies**: Two novel approaches implemented  
✅ **Flexible Configuration**: YAML-based, easy to modify  
✅ **Experiment Tracking**: Full WandB integration  
✅ **PyTorch Lightning**: Modern training framework  
✅ **Pretrained Teachers**: Automatic download from HuggingFace  
✅ **Mixed Precision**: Automatic FP16 training on GPU  
✅ **Early Stopping**: Prevent overfitting automatically  

## Expected Results

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| MobileNetV2 Baseline | ~68-70% | - |
| Single Teacher KD | ~70-72% | +2-4% |
| Multi-Teacher Dynamic KD | ~72-74% | +4-6% |
| Confidence-Based KD | ~71-73% | +3-5% |

## Extensions & Future Work

Potential improvements:
- Add more diverse teachers (ConvNext, Swin Transformer)
- Implement attention-based KD
- Add feature distillation alongside logit distillation
- Experiment with different student architectures
- Add uncertainty quantification
- Implement progressive KD (curriculum learning)

## Citation

```bibtex
@misc{confidence-aware-kd-2024,
  title={Confidence-Aware Ensemble Knowledge Distillation},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - Free for academic and commercial use


# Quick Usage Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

## Basic Training

### 1. Train Baseline (No KD)
```bash
python scripts/train.py --config configs/baseline_config.yaml
```

### 2. Train with Knowledge Distillation
```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Train with Confidence-Based KD
```bash
python scripts/train.py --config configs/confidence_config.yaml
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint path/to/checkpoint.ckpt \
    --config configs/config.yaml \
    --split test
```

## Running Experiments

### Simple Experiment
```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name my_experiment
```

### Resume from Checkpoint
```bash
python scripts/experiment.py \
    --config configs/config.yaml \
    --exp-name resumed_exp \
    --checkpoint path/to/checkpoint.ckpt
```

## Configuration Options

### Key Parameters in `configs/config.yaml`:

- **KD Type**: `type: "dynamic"` or `type: "confidence"`
- **Temperature**: `temperature: 4.0` (1-10)
- **Learning Rate**: `learning_rate: 0.01`
- **Teachers**: List in `model.teacher_names`
- **Epochs**: `max_epochs: 150`

### Example Custom Config

```yaml
kd:
  type: "confidence"
  temperature: 5.0
  learning_rate: 0.001
  use_soft_loss: true
  use_hard_loss: true

model:
  teacher_names:
    - "resnet50_cifar100"
    - "densenet121_cifar100"
```

## Monitoring

### Weights & Biases
```bash
wandb login
# Then run any training script
```

### View Metrics
- Training/validation loss
- Accuracy and AUROC
- Teacher confidence
- Gate values (dynamic KD)
- Teacher usage (confidence KD)

## Common Tasks

### Change Batch Size
Edit `configs/config.yaml`:
```yaml
data:
  batch_size: 256  # Increase/decrease as needed
```

### Use Different Student
```yaml
model:
  student_name: "efficientnet_b0"
```

### Adjust Early Stopping
```yaml
training:
  patience: 20  # Number of epochs to wait
```

### Multi-GPU Training
The framework automatically detects and uses available GPUs.

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use fewer teachers
- Enable gradient checkpointing

### Slow Training
- Increase `num_workers` in data config
- Use mixed precision (automatic on GPU)
- Reduce validation frequency

### Teacher Loading Issues
- Ensure internet connection for downloading pretrained models
- Check teacher names are correct (from timm)

## Tips

1. **Start with baseline**: Train without KD first to get baseline performance
2. **Tune temperature**: Try values between 3-5 for soft loss
3. **Monitor gate values**: In dynamic KD, check if gate is working properly
4. **Check teacher usage**: In confidence KD, ensure all teachers are being used
5. **Use early stopping**: Set appropriate patience to avoid overfitting


# WandB Logging Documentation

## Current WandB Integration Status

### ✓ Training Logging (Automatic via PyTorch Lightning)

**Location:** `src/training/trainer.py` + `src/models/kd_module.py`

**Configuration:**
```python
WandbLogger(
    project='Knowledge-Distillation-CIFAR100',  # From config
    name=experiment_name,                        # From config
    log_model='all',                             # Uploads best + latest checkpoints
    resume='allow'                               # Can resume interrupted runs
)
```

**Metrics Logged:**

#### All Methods:
- `train/loss_total` - Total training loss
- `train/accuracy` - Training accuracy (per epoch)
- `val/accuracy` - Validation accuracy ⭐ (monitored for early stopping)
- `val/loss_total` - Validation loss
- `val/auroc` - Validation AUROC (macro average)
- `test/accuracy` - Test accuracy (if test split evaluated)
- `test/auroc` - Test AUROC
- `epoch` - Current epoch number
- `learning_rate` - Current LR (from scheduler)

#### Method-Specific Metrics:

**CA-Weighted KD (Method 1):**
- `train/loss_kl` - KL divergence loss (KD component)
- `train/loss_ce` - Cross-entropy loss (ground truth component)

**Dynamic KD (Method 2):**
- `train/loss_soft` - Soft loss (KD from teachers)
- `train/loss_hard` - Hard loss (ground truth CE)
- `train/gate` - Dynamic gate value (α)
- `train/avg_teacher_conf` - Average teacher confidence

**Confidence-Based KD (Method 3):**
- `train/loss_soft` - KD loss from most confident teacher
- `train/loss_hard` - CE loss
- `train/loss_total` - Combined loss
- `train/dynamic_alpha` - Adaptive α value
- `train/max_teacher_conf_mean` - Max teacher confidence
- `train/teacher_{i}_usage_fraction` - % of samples using teacher i

**Baseline (No KD):**
- `train/loss_total` - CE loss only

### ✓ Evaluation Logging

**Location:** `scripts/evaluate.py`

**What's Logged:**
```python
{
    f"{split}/loss": results['loss'],              # test/loss or val/loss
    f"{split}/accuracy": results['accuracy'],       # test/accuracy
    f"{split}/accuracy_percent": accuracy * 100,   # For readability
}
```

**Summary Metrics:**
```python
wandb.summary[f"{split}_loss"]
wandb.summary[f"{split}_accuracy"]
wandb.summary[f"{split}_accuracy_percent"]
wandb.summary["checkpoint_path"]
```

### ✓ Model Artifacts

**Automatically Uploaded:**
- Best checkpoint (based on val/accuracy)
- Latest checkpoint (most recent)
- Stored as WandB artifacts: `model-{run_id}:best`, `model-{run_id}:latest`

## Issues & Improvements

### ⚠️ Issues Found

1. **Missing Hyperparameters Logging**
   - Temperature, alpha, gamma not logged
   - Teacher model names not logged
   - Config not automatically saved

2. **Inconsistent Metric Naming**
   - Training: `train/accuracy` 
   - Evaluation: `test/accuracy` (good)
   - But could add `final_` prefix for clarity

3. **No System Metrics**
   - GPU usage not tracked
   - Training time per epoch not logged
   - Data loading time not tracked

### 🔧 Recommended Improvements

See section below for implementation.

---

## Improvements to Implement

### 1. Add Hyperparameter Logging

Add to `src/training/trainer.py` after WandB logger initialization:

```python
# Log all hyperparameters to WandB
wandb_logger.experiment.config.update({
    # Model config
    "student_model": config['model']['student_name'],
    "teacher_models": config['model'].get('teacher_names', []),
    "num_classes": config['model']['num_classes'],
    
    # KD config
    "kd_type": config['kd']['type'],
    "temperature": config['kd'].get('temperature', None),
    "alpha": config['kd'].get('alpha', None),
    "gamma": config['kd'].get('gamma', None),
    "threshold": config['kd'].get('threshold', None),
    
    # Training config
    "max_epochs": train_config['max_epochs'],
    "batch_size": config['data']['batch_size'],
    "learning_rate": config['kd']['learning_rate'],
    "patience": train_config['patience'],
    
    # Data config
    "val_size": config['data']['val_size'],
    "num_workers": config['data']['num_workers'],
})
```

### 2. Add Training Time Metrics

Add to each KD module's `training_step` and `validation_step`:

```python
import time

# In __init__:
self.step_start_time = None

# At start of training_step:
self.step_start_time = time.time()

# At end of training_step (before return):
step_time = time.time() - self.step_start_time
self.log('train/step_time', step_time, on_step=True, on_epoch=False)
```

### 3. Add Learning Rate Logging

Already done by PyTorch Lightning automatically when using a scheduler ✓

### 4. Improve Evaluation Logging

Add model metadata to evaluation:

```python
# In evaluate.py, add:
wandb.summary.update({
    "model_name": config['model']['student_name'],
    "teachers": config['model'].get('teacher_names', []),
    "kd_method": config['kd']['type'],
    "temperature": config['kd'].get('temperature', None),
})
```

### 5. Add Config File as Artifact

```python
# In trainer.py, after fit:
wandb_logger.experiment.save(args.config)  # Upload config file
```

---

## Current Status: ✅ GOOD

The WandB integration is **functional and comprehensive**:
- ✅ All important metrics logged
- ✅ Checkpoints automatically uploaded
- ✅ Evaluation creates separate runs
- ✅ Can resume interrupted training
- ✅ Progress bar shows key metrics

**Minor improvements recommended but not critical for reproducing results.**

---

## How to Use

### View Training Progress
```bash
# Training automatically logs to WandB
python scripts/train.py --config configs/method3_adaptive_alpha.yaml

# View at: https://wandb.ai/YOUR_USERNAME/Knowledge-Distillation-CIFAR100
```

### Evaluate and Log
```bash
# Logs evaluation metrics to WandB
python scripts/evaluate.py \
    --checkpoint checkpoints/path/to/best.ckpt \
    --config configs/method3_adaptive_alpha.yaml \
    --split test

# Disable WandB for evaluation:
python scripts/evaluate.py ... --no-wandb
```

### Download Artifacts
```bash
# Evaluate from WandB artifact
python scripts/evaluate.py \
    --wandb-artifact username/project/model-abc123:best \
    --config configs/method3_adaptive_alpha.yaml
```

---

## Metrics Comparison: Notebook vs Code

| Metric | Notebook | Current Code | Status |
|--------|----------|--------------|--------|
| Training loss | ✓ | ✓ | ✅ Same |
| Validation accuracy | ✓ | ✓ | ✅ Same |
| AUROC | ✓ | ✓ | ✅ Same |
| Teacher confidence | ✓ | ✓ | ✅ Same |
| Gate/alpha values | ✓ | ✓ | ✅ Same |
| Learning rate | ✓ | ✓ | ✅ Same |
| Epoch time | ❌ | ❌ | ⚠️ Could add |
| GPU usage | ❌ | ❌ | ⚠️ Could add |

**Conclusion:** Logging is consistent with notebook implementation ✅


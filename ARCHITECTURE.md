# Architecture Documentation

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ train.py   │  │evaluate.py │  │  experiment.py       │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  YAML Configs (baseline/dynamic/confidence)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │  Trainer         │  │  PyTorch Lightning           │    │
│  │  - train_kd_     │  │  - Auto optimization         │    │
│  │    model()       │  │  - Mixed precision           │    │
│  │  - Callbacks     │  │  - Distributed training      │    │
│  └──────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Distillation Layer              │
│  ┌───────────────────────┐  ┌────────────────────────────┐ │
│  │ DynamicKDLitModule    │  │ ConfidenceBasedKDLitModule │ │
│  │                       │  │                            │ │
│  │ - Weighted ensemble   │  │ - Best teacher selection  │ │
│  │ - Confidence gating   │  │ - Dynamic alpha           │ │
│  │ - Per-sample weights  │  │ - Teacher usage tracking  │ │
│  └───────────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Model Layer                           │
│  ┌─────────────┐  ┌──────────────────────────────────────┐ │
│  │  Teachers   │  │  Student                             │ │
│  │             │  │                                      │ │
│  │ - ResNet    │  │ - MobileNetV2 (default)             │ │
│  │ - DenseNet  │  │ - EfficientNet                      │ │
│  │ - ViT       │  │ - Custom (via TIMM)                 │ │
│  │ (Frozen)    │  │ (Trainable)                         │ │
│  └─────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Data Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CIFAR100DataModule                                 │   │
│  │                                                     │   │
│  │  ┌──────────────┐  ┌─────────────────────────┐    │   │
│  │  │ Dual         │  │  Model-specific         │    │   │
│  │  │ Transform    │  │  Transforms             │    │   │
│  │  │ Dataset      │  │  - ResNet transforms    │    │   │
│  │  │              │  │  - DenseNet transforms  │    │   │
│  │  │              │  │  - ViT transforms       │    │   │
│  │  │              │  │  - MobileNet transforms │    │   │
│  │  └──────────────┘  └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Utilities & Logging                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Config      │  │  Logger      │  │  WandB          │   │
│  │  Management  │  │  Setup       │  │  Integration    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Flow

```
CIFAR-100 Dataset
    │
    ├─→ Train Split (45K samples)
    │       │
    │       ├─→ DualTransformDataset
    │       │       │
    │       │       ├─→ Teacher Transform 1 (ResNet)
    │       │       ├─→ Teacher Transform 2 (DenseNet)
    │       │       ├─→ Teacher Transform 3 (ViT)
    │       │       └─→ Student Transform (MobileNet)
    │       │
    │       └─→ DataLoader (batch_size=128)
    │
    ├─→ Val Split (5K samples)
    │       └─→ Same transform pipeline
    │
    └─→ Test Split (10K samples)
            └─→ Same transform pipeline
```

### 2. Knowledge Distillation Flow

#### Dynamic KD with Weighted Ensemble

```
Input Batch
    │
    ├─→ Teachers (frozen)
    │   ├─→ Teacher 1: logits_1 ──┐
    │   ├─→ Teacher 2: logits_2 ──┼─→ Compute Weights
    │   └─→ Teacher 3: logits_3 ──┘   (per-sample, based on CE loss)
    │                                         │
    ├─→ Student (trainable)                   │
    │   └─→ student_logits                    │
    │                                         ▼
    └─→ Ground Truth Labels ──→ Compute Teacher Confidence
                                            │
                                            ▼
                                  Confidence-based Gate
                                  gate = σ(γ*(conf-θ))
                                            │
                ┌───────────────────────────┴────────────────────────┐
                │                                                    │
                ▼                                                    ▼
        Weighted KD Loss                                      Hard CE Loss
    L_KD = Σ_k w_k * KL(S||T_k)                          L_CE = CE(S, y)
                │                                                    │
                └────────────────┬───────────────────────────────────┘
                                 ▼
                    Total Loss = gate * L_KD + (1-gate) * α * L_CE
                                 │
                                 ▼
                          Backpropagation
                    (only through student)
```

#### Confidence-Based KD

```
Input Batch
    │
    ├─→ Teachers (frozen)
    │   ├─→ Teacher 1: logits_1, conf_1 ──┐
    │   ├─→ Teacher 2: logits_2, conf_2 ──┼─→ Select Max Confidence
    │   └─→ Teacher 3: logits_3, conf_3 ──┘   Teacher per Sample
    │                                                  │
    ├─→ Student (trainable)                           │
    │   └─→ student_logits                            │
    │                                                  ▼
    └─→ Ground Truth ──→ best_teacher_logits, max_conf
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
            KD Loss from Best                   Hard CE Loss
         L_KD = KL(S||T_best)                 L_CE = CE(S, y)
                    │                                │
                    └────────┬────────────────────────┘
                             ▼
                  α = mean(max_conf)
            L_total = α * L_KD + (1-α) * L_CE
                             │
                             ▼
                      Backpropagation
```

### 3. Training Loop

```
For each epoch:
    │
    ├─→ Training Phase
    │   │
    │   For each batch:
    │   ├─→ Forward pass (teachers + student)
    │   ├─→ Compute KD loss
    │   ├─→ Compute hard loss (optional)
    │   ├─→ Combine losses (with gate/alpha)
    │   ├─→ Backward pass
    │   ├─→ Optimizer step
    │   └─→ Log metrics to WandB
    │
    ├─→ Validation Phase
    │   │
    │   For each batch:
    │   ├─→ Forward pass (no gradients)
    │   ├─→ Compute metrics (accuracy, AUROC, loss)
    │   └─→ Log validation metrics
    │
    ├─→ Callbacks
    │   ├─→ Early Stopping (monitor val/accuracy)
    │   ├─→ Model Checkpoint (save best models)
    │   └─→ Learning Rate Scheduler
    │
    └─→ Check stopping criteria
```

### 4. Class Hierarchy

```
pl.LightningModule
    │
    ├─→ DynamicKDLitModule
    │   ├─→ __init__(teachers, student, temperature, gamma, threshold, ...)
    │   ├─→ forward(x) → student(x)
    │   ├─→ configure_optimizers() → AdamW + CosineAnnealingLR
    │   ├─→ compute_weights(teacher_logits, labels) → weights
    │   ├─→ compute_average_teacher_confidence() → confidence
    │   ├─→ compute_losses() → total_loss
    │   ├─→ training_step(batch) → loss
    │   ├─→ validation_step(batch) → metrics
    │   └─→ test_step(batch) → metrics
    │
    └─→ ConfidenceBasedKDLitModule
        ├─→ __init__(teachers, student, temperature, ...)
        ├─→ forward(x) → student(x)
        ├─→ get_most_confident_teacher_logits() → best_logits
        ├─→ compute_max_teacher_confidence() → max_conf
        ├─→ compute_losses() → total_loss (with dynamic alpha)
        ├─→ training_step(batch) → loss
        ├─→ validation_step(batch) → metrics
        └─→ test_step(batch) → metrics
```

### 5. Configuration Structure

```yaml
config.yaml
├─→ data:
│   ├─→ data_dir
│   ├─→ batch_size
│   ├─→ num_workers
│   ├─→ teacher_model_types  # For transforms
│   └─→ student_model_type
│
├─→ model:
│   ├─→ num_classes
│   ├─→ student_name
│   └─→ teacher_names  # List of pretrained models
│
├─→ kd:
│   ├─→ type  # 'dynamic' or 'confidence'
│   ├─→ temperature
│   ├─→ gamma (for dynamic)
│   ├─→ threshold (for dynamic)
│   ├─→ alpha
│   ├─→ learning_rate
│   ├─→ use_soft_loss
│   └─→ use_hard_loss
│
├─→ training:
│   ├─→ max_epochs
│   ├─→ patience
│   └─→ log_every_n_steps
│
└─→ wandb:
    ├─→ project
    ├─→ name
    ├─→ log_model
    └─→ resume
```

### 6. Module Dependencies

```
scripts/
├─→ train.py
│   ├─→ src.data.CIFAR100DataModule
│   ├─→ src.models.create_student_model
│   ├─→ src.models.create_teacher_models
│   ├─→ src.training.create_kd_module_from_config
│   ├─→ src.training.train_kd_model
│   └─→ src.utils.load_config
│
├─→ evaluate.py
│   ├─→ src.data.CIFAR100DataModule
│   ├─→ src.models.create_student_model
│   ├─→ src.evaluation.evaluate_model
│   └─→ src.utils.load_config
│
└─→ experiment.py
    └─→ (same as train.py + wandb integration)
```

## Design Patterns Used

### 1. **Strategy Pattern**
- Different KD strategies (Dynamic vs Confidence-based)
- Interchangeable via configuration

### 2. **Factory Pattern**
- `create_student_model()`: Creates student models
- `create_teacher_models()`: Creates teacher ensembles
- `create_kd_module_from_config()`: Creates KD modules from config

### 3. **Template Method Pattern**
- `shared_eval_step()`: Shared validation/test logic
- Reduces code duplication

### 4. **Dependency Injection**
- Models and configs injected into trainers
- Facilitates testing and flexibility

### 5. **Observer Pattern**
- PyTorch Lightning callbacks
- WandB logging

## Performance Considerations

### Memory Optimization
- Teachers frozen → No gradient storage
- Mixed precision training → 50% memory reduction
- Gradient accumulation → Effective larger batch sizes

### Computational Efficiency
- `@torch.no_grad()` for teacher inference
- Batch processing for all operations
- Efficient tensor operations (no loops where possible)

### Scalability
- DataLoader with multiple workers
- Pin memory for GPU transfer
- Distributed training ready (via PyTorch Lightning)

## Extension Points

### Adding New KD Strategy

```python
class MyCustomKDModule(pl.LightningModule):
    def __init__(self, teachers, student, ...):
        super().__init__()
        # Custom initialization
    
    def compute_losses(self, student_logits, labels, teacher_logits):
        # Custom loss computation
        pass
    
    # Implement required methods
```

### Adding New Model Architecture

```python
# In src/models/student.py
def create_custom_student(num_classes):
    model = YourCustomArchitecture(num_classes)
    return model

# In config.yaml
model:
  student_name: "custom"
```

### Adding New Dataset

```python
# In src/data/
class CustomDataModule(pl.LightningDataModule):
    # Implement required methods
    pass
```

## Testing Strategy

### Unit Tests (Future)
- Test individual components
- Mock dependencies
- Test edge cases

### Integration Tests (Future)
- Test end-to-end pipeline
- Test with small dataset
- Verify outputs

### Performance Tests (Future)
- Benchmark training speed
- Memory profiling
- GPU utilization

---

**Note**: This architecture is designed for extensibility, maintainability, and performance. Each component is loosely coupled and can be modified independently.


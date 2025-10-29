# ✅ WandB Configuration - STANDARDIZED

## Changes Made

### Before ❌
- **7 different projects** - experiments scattered across multiple WandB projects
- Inconsistent naming (some "KD-", some "Knowledge-Distillation-", some "Method3-")
- Generic names like "KD-Experiment" hard to identify

### After ✅
- **Single project: `KD-CIFAR100`** - all experiments in one place for easy comparison
- Consistent, descriptive naming scheme
- Clear identification of method, architecture, and key parameters

---

## Complete Configuration

| Config File | Experiment Name | Description |
|-------------|-----------------|-------------|
| `baseline_config.yaml` | `Baseline-MobileNetV2` | Student trained from scratch (no KD) |
| `single_teacher_densenet.yaml` | `Single-DenseNet121-alpha0.50-BEST` | Best single teacher (75.38%) ⭐ |
| `single_teacher_resnet50.yaml` | `Single-ResNet50-alpha0.75` | Single ResNet50 teacher |
| `single_teacher_vit.yaml` | `Single-ViT-alpha0.25` | Single ViT teacher |
| `method1_ca_wkd.yaml` | `Method1-CA-WKD-ResNet-DenseNet-ViT` | Confidence-aware weighted KD |
| `method2_dynamic_kd.yaml` | `Method2-Dynamic-3xResNet-gamma5` | Dynamic α with 3 ResNets |
| `method2_diverse_ensemble.yaml` | `Method2-Dynamic-Diverse-gamma10` | Dynamic α with diverse teachers |
| `method3_adaptive_alpha.yaml` | `Method3-Adaptive-3xResNet-WithCE-temp4` | Adaptive α with CE loss |
| `method3_3resnets_no_ce.yaml` | `Method3-Adaptive-3xResNet-NoCE-temp4` | Adaptive α without CE loss |
| `method3_densenet_resnet_temp8.yaml` | `Method3-Adaptive-DenseNet-ResNet-temp8-BEST` | Best multi-teacher (74.00%) ⭐ |
| `method3_densenet_resnet_temp16.yaml` | `Method3-Adaptive-DenseNet-ResNet-temp16` | Same as above, temp=16 |
| `method3_diverse_ensemble.yaml` | `Method3-Adaptive-Diverse-Ensemble-temp4` | Adaptive α with R+D+V |

---

## Naming Convention

### Format: `{Category}-{Details}-{Parameters}`

**Categories:**
- `Baseline` - No knowledge distillation
- `Single` - One teacher
- `Method1` - CA-WKD (Confidence-Aware Weighted KD)
- `Method2` - Dynamic α-Guided KD
- `Method3` - Adaptive α-Guided KD (best method)

**Details:**
- Teacher architectures (e.g., `DenseNet121`, `3xResNet`, `Diverse`)
- Special variants (`NoCE` = no cross-entropy, `WithCE` = with CE)

**Parameters:**
- `alpha{value}` - Fixed alpha value for single teacher
- `temp{value}` - Temperature value
- `gamma{value}` - Gamma value for gating
- `BEST` - Best performing configuration

---

## Benefits

### 1. Easy Comparison
All experiments in one project → side-by-side comparison in WandB dashboard

### 2. Clear Identification
Experiment names include:
- Method type
- Architecture details  
- Key hyperparameters
- Performance indicators (BEST)

### 3. Filtering & Search
Can filter by:
- Method (Method1, Method2, Method3, Single, Baseline)
- Temperature (temp4, temp8, temp16)
- Loss type (NoCE, WithCE)

### 4. Reproducibility
Name clearly indicates configuration → easy to map to config file

---

## WandB Dashboard Organization

### View in WandB:
```
Project: KD-CIFAR100
├── Baseline-MobileNetV2                          (66.29%)
├── Single-DenseNet121-alpha0.50-BEST             (75.38%) ⭐ Best
├── Single-ResNet50-alpha0.75                     (73.75%)
├── Single-ViT-alpha0.25                          (74.00%)
├── Method1-CA-WKD-ResNet-DenseNet-ViT           (73.33%)
├── Method2-Dynamic-3xResNet-gamma5              (~72%)
├── Method2-Dynamic-Diverse-gamma10              (71.38%)
├── Method3-Adaptive-3xResNet-WithCE-temp4       (~72%)
├── Method3-Adaptive-3xResNet-NoCE-temp4         (71.89%)
├── Method3-Adaptive-DenseNet-ResNet-temp8-BEST  (74.00%) ⭐ Best Multi
├── Method3-Adaptive-DenseNet-ResNet-temp16      (~72%)
└── Method3-Adaptive-Diverse-Ensemble-temp4      (~72%)
```

### Tags for Each Run:
- Method type: `baseline`, `single-teacher`, `method1`, `method2`, `method3`
- Architecture: `mobilenet`, `resnet`, `densenet`, `vit`
- KD type: `ca-wkd`, `dynamic`, `adaptive`, `confidence`

---

## Usage

### Run Single Experiment:
```bash
python scripts/train.py --config configs/single_teacher_densenet.yaml
# Will appear in WandB as: Single-DenseNet121-alpha0.50-BEST
```

### Run All Experiments:
```bash
bash run_all_experiments.sh
# All will appear under project: KD-CIFAR100
```

### View Results:
```
https://wandb.ai/YOUR_USERNAME/KD-CIFAR100
```

---

## Hyperparameters Logged

For each run, WandB automatically logs:

**Model Config:**
- student_model, teacher_models, num_classes, pretrained

**KD Config:**
- kd_type, temperature, alpha, gamma, threshold, learning_rate, use_soft/hard_loss

**Training Config:**
- max_epochs, patience, batch_size, val_size, num_workers, seed

**Metrics:**
- All losses (KL, CE, total, soft, hard)
- Accuracy (train, val, test)
- AUROC, teacher confidence, gate values

---

## ✅ Status: PRODUCTION READY

All configurations standardized and consistent with:
- ✅ Single unified project
- ✅ Descriptive experiment names
- ✅ Complete hyperparameter logging
- ✅ Comprehensive metrics tracking
- ✅ Clear documentation
- ✅ Easy comparison and filtering

**Ready for full experiment runs!** 🚀


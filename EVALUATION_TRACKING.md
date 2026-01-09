# Evaluation Tracking Guide

## Problem Solved

Previously, evaluation runs in WandB had cryptic names like `eval-test-i9skbi4e`, making it difficult to identify which model checkpoint was evaluated.

**Now fixed:** Evaluation runs use descriptive names like `eval-test-Method1-CA-WKD-ResNet-DenseNet-ViT-i9skbi4e` ✅

## 🔍 Identify Your Existing 9 Evaluation Runs

Run this script to analyze and match your existing evaluation runs to their training runs:

```bash
python scripts/identify_evaluations.py
```

### Output Example

```
================================================================================
EVALUATION RUNS SUMMARY
================================================================================

Total evaluation runs: 9
  - Matched to training runs: 8
  - Unmatched: 1

================================================================================
MATCHED EVALUATIONS
================================================================================

1. Evaluation Run: eval-test-i9skbi4e
   ──────────────────────────────────────────────────────────────────────
   Eval Run ID:       uiqhw7mp
   Training Run ID:   i9skbi4e
   Training Run Name: Method1-CA-WKD-ResNet-DenseNet-ViT
   KD Method:         ca_weighted
   Teachers:          resnet50_cifar100, densenet121_cifar100, vit
   Test Accuracy:     73.92%
   Checkpoint:        /workspace/kd_thesis/artifacts/model-i9skbi4e:v73/model.ckpt...

2. Evaluation Run: eval-test-wvhuxac6
   ──────────────────────────────────────────────────────────────────────
   Eval Run ID:       qcsvcace
   Training Run ID:   wvhuxac6
   Training Run Name: Method3-Adaptive-3xResNet-WithCE-temp4
   KD Method:         confidence
   Teachers:          resnet50_cifar100, resnet18_cifar100, resnet34_cifar100
   Test Accuracy:     71.60%
   Checkpoint:        /workspace/kd_thesis/artifacts/model-wvhuxac6:v76/model.ckpt...

...
```

### Export to CSV

The script also exports results to `logs/evaluation_matches.csv` for easy filtering in Excel/Google Sheets:

```bash
python scripts/identify_evaluations.py --output logs/my_evaluations.csv
```

### Advanced Options

```bash
# Specify WandB entity (if not using default)
python scripts/identify_evaluations.py --entity your-username

# Specify different project
python scripts/identify_evaluations.py --project "My-Other-Project"

# Skip CSV export
python scripts/identify_evaluations.py --no-csv
```

## 🆕 Future Evaluations (Improved Naming)

After the fix to `scripts/evaluate.py`, new evaluation runs will automatically have descriptive names.

### Naming Format

```
eval-{split}-{experiment_name}-{run_id}
```

**Examples:**
- `eval-test-Baseline-MobileNetV2-gc7agbej`
- `eval-test-Method1-CA-WKD-ResNet-DenseNet-ViT-i9skbi4e`
- `eval-test-Method2-Dynamic-3xResNet-gamma5-ns4nmxno`
- `eval-test-Method3-Adaptive-DenseNet-ResNet-temp8-BEST-av6eqh8e`

### How It Works

The evaluation script now:

1. **Uses experiment name from config** (`wandb.name` field)
2. **Fallback to auto-generated name** if not in config:
   - Baseline: `Baseline`
   - With teachers: `{method}-{teachers}` (e.g., `Confidence-ResNet-DenseNet-ViT`)
3. **Appends run ID** for uniqueness
4. **Adds to tags** for easy filtering in WandB

## 📊 WandB Organization

### Filtering Evaluation Runs

In WandB UI, you can now filter by:

**By Job Type:**
- `job_type: evaluation` - All evaluation runs
- `job_type: train` - All training runs (default)

**By Tags:**
- `evaluation` - All evaluation runs
- `test` or `val` - Specific data split
- `baseline`, `ca_weighted`, `dynamic`, `confidence` - By KD method
- Experiment name tags (e.g., `Method1-CA-WKD-ResNet-DenseNet-ViT`)

**By Summary Fields:**
- `kd_method: confidence` - Specific KD methods
- `num_teachers: 3` - Number of teachers
- `test_accuracy > 0.73` - Performance threshold

### Linking Training and Evaluation

Each evaluation run includes:
- `config.run_id` - Training run ID
- `config.experiment_name` - Experiment name
- `summary.checkpoint_path` - Full path to checkpoint used
- `tags` - Includes training run ID

## 🔗 Matching Evaluations to Training Runs

### Method 1: By Run Name (After Fix)

Training run: `Method1-CA-WKD-ResNet-DenseNet-ViT`  
Evaluation run: `eval-test-Method1-CA-WKD-ResNet-DenseNet-ViT-i9skbi4e`

→ Clear match by name! ✅

### Method 2: By Run ID in Tags

1. Find training run ID (e.g., `i9skbi4e`)
2. Search evaluation runs with tag `i9skbi4e`
3. Find matching evaluation run

### Method 3: By Checkpoint Path

1. Look at evaluation run's `checkpoint_path` in summary
2. Extract run ID from path: `artifacts/model-{run_id}:v{version}/model.ckpt`
3. Find training run with that ID

### Method 4: Use the Identification Script

```bash
python scripts/identify_evaluations.py
```

This automatically matches everything for you! 🎉

## 📁 File Organization Reference

### Training Outputs

```
checkpoints/{kd_type}/{experiment_name}/{run_id}/
├── best.ckpt   # Best validation accuracy
└── last.ckpt   # Latest checkpoint
```

### WandB Artifacts (Downloaded)

```
artifacts/model-{run_id}:v{version}/
└── model.ckpt  # Downloaded checkpoint
```

### Logs

```
logs/
├── training/          # Training logs
├── evaluation/        # Evaluation logs
├── results/           # Result summaries
└── evaluation_matches.csv  # Matching results from identify_evaluations.py
```

## 🎯 Best Practices

### 1. Use Descriptive Config Names

In your config files, set meaningful names:

```yaml
wandb:
  project: "KD-CIFAR100"
  name: "Method3-Adaptive-DenseNet-ResNet-temp8-BEST"  # Descriptive!
  log_model: "all"
  resume: "allow"
```

### 2. Run Evaluation Immediately After Training

The `run_all_experiments.sh` script already does this automatically:

```bash
# Train → Evaluate automatically
./run_all_experiments.sh
```

### 3. Tag Important Runs

Add custom tags when evaluating manually:

```bash
python scripts/evaluate.py \
    --checkpoint path/to/best.ckpt \
    --config configs/method3_adaptive_alpha.yaml \
    --split test
```

The script automatically adds tags:
- Data split (`test` or `val`)
- `evaluation`
- Training run ID
- KD method
- Experiment name

## 🐛 Troubleshooting

### "Cannot find training run for evaluation"

**Cause:** Training run ID not found in checkpoint path or config

**Solution:** Use the identification script to analyze:
```bash
python scripts/identify_evaluations.py
```

### "Multiple evaluations for same training run"

**Cause:** Model was evaluated multiple times (normal if re-evaluating)

**Solution:** Check the `created_at` timestamp to find the latest evaluation

### "Evaluation run name still shows old format"

**Cause:** You're looking at old evaluation runs created before the fix

**Solution:** 
1. Use `identify_evaluations.py` to understand existing runs
2. Future evaluations will use the new naming format

## 📈 Quick Reference Commands

```bash
# Identify all evaluation runs
python scripts/identify_evaluations.py

# Export to custom location
python scripts/identify_evaluations.py --output my_results.csv

# Evaluate with new naming (automatic)
python scripts/evaluate.py --wandb-run-id i9skbi4e --config configs/method1_ca_wkd.yaml

# Run full pipeline (train + evaluate)
./run_all_experiments.sh

# View specific evaluation in WandB
# → Go to WandB UI → Filter by job_type: evaluation
```

## ✅ Summary

**Before:**
- ❌ Cryptic names: `eval-test-i9skbi4e`
- ❌ Hard to identify which model was evaluated
- ❌ No easy way to match evaluations to training runs

**After:**
- ✅ Descriptive names: `eval-test-Method1-CA-WKD-ResNet-DenseNet-ViT-i9skbi4e`
- ✅ Clear matching between training and evaluation
- ✅ Script to analyze existing runs: `identify_evaluations.py`
- ✅ Better tags and metadata for filtering
- ✅ CSV export for easy analysis


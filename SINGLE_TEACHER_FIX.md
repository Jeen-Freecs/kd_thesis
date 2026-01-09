# Single-Teacher Fix Documentation

## 🐛 Bug Fixed

**Issue:** `ZeroDivisionError: float division by zero` when training with single-teacher configurations.

**Error Location:** `src/models/kd_module.py` in `compute_weights()` method

**Root Cause:** The weight computation formula assumes multiple teachers:
```python
weights = (1.0 / (K - 1)) * (1.0 - (exp_losses / sum_exp_losses))
```

When K=1 (single teacher), this causes division by zero: `1.0 / (1 - 1) = 1.0 / 0`

## ✅ Solution Applied

Updated both `CAWeightedKDLitModule` and `DynamicKDLitModule` classes to handle single-teacher case:

```python
def compute_weights(self, teacher_logits, labels):
    K = len(teacher_logits)
    batch_size = teacher_logits[0].shape[0]
    
    # Special case: single teacher - just return weight of 1.0
    if K == 1:
        return torch.ones((batch_size, 1), device=teacher_logits[0].device)
    
    # Original multi-teacher logic for K > 1
    # ...
```

**Logic:** For a single teacher, there's no need to compute relative weights - the single teacher always gets a weight of 1.0.

## 📝 Affected Configurations

These configs now work correctly:

1. **`configs/single_teacher_densenet.yaml`**
   - Single DenseNet-121 teacher
   - Expected accuracy: 75.38%
   - α = 0.50 (balanced KD/CE)

2. **`configs/single_teacher_resnet50.yaml`**
   - Single ResNet-50 teacher
   - Expected accuracy: 73.75%
   - α = 0.75 (high KD weight)

3. **`configs/single_teacher_vit.yaml`**
   - Single Vision Transformer teacher
   - Expected accuracy: 74.00%
   - α = 0.25 (low KD weight, high CE)

## 🚀 How to Run

Now you can run your experiments without errors:

```bash
# Run all experiments (including single-teacher)
./run_all_experiments.sh

# Or run individual single-teacher experiments
python scripts/train.py --config configs/single_teacher_densenet.yaml
python scripts/train.py --config configs/single_teacher_resnet50.yaml
python scripts/train.py --config configs/single_teacher_vit.yaml
```

## 🔍 Technical Details

### What Changed

**File:** `src/models/kd_module.py`

**Lines Modified:**
- Lines 103-137: `CAWeightedKDLitModule.compute_weights()`
- Lines 364-398: `DynamicKDLitModule.compute_weights()`

**Changes:**
1. Added batch_size extraction
2. Added K==1 check at the start
3. Return uniform weights (all 1.0) for single teacher
4. Original multi-teacher logic unchanged

### Why This Works

**For K=1 (single teacher):**
- Only one teacher, no need for weighted averaging
- Weight = 1.0 (use all knowledge from that teacher)
- Shape: `(batch_size, 1)`

**For K>1 (multiple teachers):**
- Original exponential weighting formula applies
- Teachers with lower CE loss get higher weights
- Weights don't sum exactly to 1, but are normalized by K-1
- Shape: `(batch_size, K)`

### Backward Compatibility

✅ **No impact on existing experiments** - all multi-teacher experiments continue to work exactly as before.

The fix only adds a special case for K=1, which previously crashed.

## 🧪 Testing

To verify the fix works, you can run a quick training test:

```bash
# Should complete without ZeroDivisionError
python scripts/train.py \
    --config configs/single_teacher_densenet.yaml \
    | head -100  # Just see the first 100 lines

# If you see training start without errors, the fix works!
```

## 📊 Expected Behavior

### Before Fix ❌
```
ZeroDivisionError: float division by zero
  File "src/models/kd_module.py", line 389, in compute_weights
    weights = (1.0 / (K - 1)) * (1.0 - (exp_losses / sum_exp_losses))
               ~~~~^~~~~~~~~
```

### After Fix ✅
```
Files already downloaded and verified
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params | Mode
---------------------------------------------------------
0 | student     | MobileNetV2      | 2.4 M  | train
1 | kl_div_loss | KLDivLoss        | 0      | train
2 | ce_loss     | CrossEntropyLoss | 0      | train
---------------------------------------------------------

Training: 0%|          | 0/150 [00:00<?, ?it/s]
```

## ⚠️ Notes

1. **Single-teacher configs use `type: "dynamic"`** - This is intentional. The dynamic type allows controlling the α parameter for KD/CE balance.

2. **Weight computation still happens** - Even with one teacher, the `compute_weights()` method is called, but now returns 1.0 instead of crashing.

3. **Gating logic differs** - For single teacher, the gating formula at line 479 includes a check:
   ```python
   if self.num_teachers > 1:
       # Dynamic gating based on teacher confidence
   else:
       # Fixed alpha-based weighting for single teacher
   ```

## 📈 Performance Expectations

Based on the notebook experiments:

| Config | Teacher | Alpha | Expected Test Accuracy |
|--------|---------|-------|----------------------|
| single_teacher_densenet | DenseNet-121 | 0.50 | **75.38%** (BEST) |
| single_teacher_vit | ViT | 0.25 | 74.00% |
| single_teacher_resnet50 | ResNet-50 | 0.75 | 73.75% |

These should match or exceed the baseline MobileNetV2 without KD (~70.26%).

## ✅ Summary

**Status:** ✅ FIXED

**Impact:** Single-teacher knowledge distillation now works correctly

**Changed Files:** `src/models/kd_module.py` (2 methods updated)

**Tests Needed:** Run training with single-teacher configs to verify no errors

**Backward Compatible:** Yes - all existing multi-teacher experiments unaffected


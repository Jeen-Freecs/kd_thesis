# Heterogeneous Complementary Distillation (HCD)

> **Paper**: "Heterogeneous Complementary Distillation" (AAAI 2026)
> **Authors**: Liuchi Xu, Hao Zheng, Lu Wang, Lisheng Xu, Jun Cheng
> **arXiv**: [2511.10942](https://arxiv.org/abs/2511.10942)
> **Code**: [github.com/yema-web/HCD](https://github.com/yema-web/HCD)

---

## 📌 Overview

**HCD** is a feature-based knowledge distillation framework designed for **heterogeneous architectures** (e.g., ViT → ResNet, Swin → MobileNet). Unlike traditional KD methods that struggle with mismatched spatial representations between different architecture types, HCD leverages **complementary features** from both teacher and student to create shared logits for knowledge transfer.

### Key Contributions

1. **Complementary Feature Mapper (CFM)**: Concatenates student intermediate features with teacher penultimate features, then maps them to shared logits
2. **Sub-logit Decoupled Distillation (SDD)**: Decomposes shared logits into $n$ sub-logits, fused with teacher logits for classification rectification
3. **Orthogonality Loss (OL)**: Ensures diversity among sub-logits to prevent redundant knowledge transfer

---

## 🔬 Problem: Heterogeneous KD Challenges

Traditional KD methods face issues when teacher and student have different architectures:

| Challenge | Description |
|-----------|-------------|
| **Spatial Mismatch** | CNNs have local receptive fields; ViTs have global attention |
| **Feature Dimension Gap** | Different intermediate feature dimensions |
| **Inductive Bias Difference** | CNNs: locality & translation equivariance; ViTs: global context |

**Previous approaches**:
- **OFA-KD**: Maps student features to logit space, aligns with teacher logits → *over-relies on teacher, ignores student strengths*
- **PAT**: Uses region-aware attention for alignment → *high computational cost, complex design*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HCD Framework                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Student                           Teacher                          │
│   ┌─────────┐                      ┌─────────┐                      │
│   │  Stage 1 │──┐                  │         │                      │
│   ├─────────┤  │                   │ Forward │                      │
│   │  Stage 2 │──┼── Projector ──┐  │  Pass   │                      │
│   ├─────────┤  │                │  │         │                      │
│   │  Stage 3 │──┤                │  └────┬────┘                      │
│   ├─────────┤  │                │       │                           │
│   │  Stage 4 │──┘                │       ▼                           │
│   └────┬────┘                    │  feat_teacher                     │
│        │                         │       │                           │
│        ▼                         │       │                           │
│   logits_student                 └───────┼───────────────────────────│
│        │                                 │                           │
│        │         ┌───────────────────────┘                           │
│        │         │                                                   │
│        │         ▼                                                   │
│        │    ┌─────────────────────────────────────┐                  │
│        │    │         Per-Stage CFM               │                  │
│        │    │  [feat_student_i ‖ feat_teacher]    │                  │
│        │    │            ↓                        │                  │
│        │    │   Linear → ReLU → Linear            │                  │
│        │    │            ↓                        │                  │
│        │    │    k × num_classes logits           │                  │
│        │    └────────────┬────────────────────────┘                  │
│        │                 │                                           │
│        │                 ▼                                           │
│        │    ┌─────────────────────────────────────┐                  │
│        │    │     SDD: Sub-logit Fusion           │                  │
│        │    │  z_fused = λ_s·z_sub + λ_t·z_t      │                  │
│        │    └────────────┬────────────────────────┘                  │
│        │                 │                                           │
│        │          ┌──────┴──────┐                                    │
│        │          │   DETACH    │  ← KL targets must be detached     │
│        │          └──────┬──────┘                                    │
│        ▼                 ▼                                           │
│   ┌─────────────────────────────────────────────┐                    │
│   │              Loss Computation                │                    │
│   │  L = L_gt + L_hcd + L_orthogonality          │                    │
│   │  (averaged across stages, not summed)         │                    │
│   └─────────────────────────────────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Feature Extraction & Projection

For each student stage $i \in \{1, 2, 3, 4\}$:

$$\mathbf{f}_s^i = \text{Projector}_i(\text{Stage}_i(\mathbf{x}))$$

Where the projector uses **SepConv** (depthwise separable convolutions) for downsampling:

```python
SepConv = DepthwiseConv → PointwiseConv → BN → ReLU → DepthwiseConv → PointwiseConv → BN → ReLU
```

### 2. Complementary Feature Mapper (CFM)

Concatenate student stage features with teacher's penultimate features:

$$\mathbf{f}_{cat}^i = [\mathbf{f}_s^i \| \mathbf{f}_t]$$

Map to shared logits via CFM (per stage):

$$\mathbf{z}_{shared}^i = \text{CFM}_i(\mathbf{f}_{cat}^i) = W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{f}_{cat}^i)$$

Where output dimension is $k \times C$ (k sub-logits, C classes).

### 3. Sub-logit Decoupled Distillation (SDD)

Reshape shared logits into k sub-logits:

$$\mathbf{z}_{shared}^i \in \mathbb{R}^{B \times kC} \rightarrow \mathbf{z}_{sub}^{i,j} \in \mathbb{R}^{B \times k \times C}$$

Fuse with teacher logits for rectification:

$$\mathbf{z}_{fused}^{i,j} = \lambda_s \cdot \mathbf{z}_{sub}^{i,j} + \lambda_t \cdot \mathbf{z}^t$$

Where $\lambda_s = \lambda_t = 1.0$ (default).

### 4. Orthogonality Loss (OL)

To ensure sub-logit diversity, **zero out** the ground-truth class and penalize high similarity:

$$\mathbf{z}_{masked} = \text{ZeroMask}(\mathbf{z}_{sub}, y)$$

> **Implementation note**: The ground-truth position is zeroed (set to 0), not replaced with a large negative value. Using `-1e6` causes overflow to `Inf` under float16 AMP (max float16 = 65504), which propagates `NaN` through `F.normalize`.

Normalize and compute pairwise similarity:

$$\mathbf{S} = \text{Normalize}(\mathbf{z}_{masked}) \cdot \text{Normalize}(\mathbf{z}_{masked})^T$$

Orthogonality loss with threshold $\theta = 0.5$:

$$\mathcal{L}_{orth} = \mathbb{E}\left[\text{ReLU}(\mathbf{S}_{off-diag} - \theta)^2\right]$$

---

## 📊 Loss Functions

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{gt} + \mathcal{L}_{hcd} + \mathcal{L}_{orth}$$

> **Implementation note**: All losses are **averaged** across stages (not summed). Summing 4 stages causes the HCD loss (~400) to overwhelm the GT loss (~20), preventing the student from learning the classification task.

### 1. Ground Truth Loss ($\mathcal{L}_{gt}$)

Cross-entropy on student logits + averaged CE on fused sub-logits:

$$\mathcal{L}_{gt} = \omega \cdot \left[ \text{CE}(\mathbf{z}^s, y) + \frac{1}{N_{\text{stages}}} \sum_{i=1}^{4} \frac{1}{k} \sum_{j=1}^{k} \text{CE}(\mathbf{z}_{fused}^{i,j}, y) \right]$$

### 2. HCD Loss ($\mathcal{L}_{hcd}$)

KL divergence from student to **detached** fused sub-logits:

$$\mathcal{L}_{hcd} = \frac{\lambda}{N_{\text{stages}}} \sum_{i=1}^{4} \frac{1}{k} \sum_{j=1}^{k} \text{KL}\left( \sigma(\mathbf{z}^s / T) \| \sigma(\text{sg}[\mathbf{z}_{fused}^{i,j}] / T) \right) \cdot T^2$$

Where $\text{sg}[\cdot]$ denotes **stop-gradient** (`.detach()`).

> **Why detach?** The fused sub-logits depend on student features via the CFM. Without detaching, the KL gradient flows backward through the sub_logits, pushing them *away* from the student distribution (because minimizing KL by moving the target is easier than moving the student). This conflicts with the CE loss that trains the CFM to produce correct predictions. Detaching ensures:
> - **CE** trains the CFM to produce good sub-logits ✓
> - **KL** trains the **student only** to match those sub-logits ✓

### 3. Orthogonality Loss ($\mathcal{L}_{orth}$)

$$\mathcal{L}_{orth} = \frac{\beta}{N_{\text{stages}}} \sum_{i=1}^{4} \text{OL}(\mathbf{z}_{sub}^i)$$

---

## ⚠️ Implementation Pitfalls & Fixes

Our implementation encountered and resolved several critical issues:

### 1. Float16 AMP Overflow in Masking

| Issue | Fix |
|-------|-----|
| `_hcd_remove_one_hot` used `-1e6` mask value | Changed to zeroing out: `logits * (1 - mask)` |
| Float16 max = 65504, so `1e6` → `Inf` | Zero-masking avoids all overflow issues |
| `Inf` in `F.normalize` → `NaN` propagation | Clean normalization on non-label dimensions |

### 2. `_init_weights` Destroyed Pretrained Weights

| Issue | Fix |
|-------|-----|
| `self.modules()` iterates over ALL submodules | Only init `stage_projectors` and `cfm` |
| Teacher's pretrained weights were re-randomized | Teacher weights preserved |
| Student's pre-init from timm was destroyed | Student weights preserved |

### 3. Dynamic Modules Not in Optimizer

| Issue | Fix |
|-------|-----|
| `_channel_proj_{i}` created with `setattr` in forward | Pre-registered as `nn.ModuleList` in `__init__` |
| Not in optimizer's param list | Added to `configure_optimizers` |
| Gradients computed but never applied | Now properly optimized |

### 4. Conflicting Gradients from Undeteached KL Targets

| Issue | Fix |
|-------|-----|
| Sub_logits used as KL targets without `.detach()` | Added `.detach()` on sub_logits in KL |
| KL gradient pushed sub_logits *away* from student | KL now only updates student parameters |
| CE and KL gave conflicting signals to CFM | CFM trained by CE only, student by KL only |

### 5. Loss Scale Explosion

| Issue | Fix |
|-------|-----|
| Losses summed over 4 stages: effective weight 24× CE | Averaged across stages: effective weight 6× CE |
| `loss_hcd` ≈ 400 vs `loss_gt` ≈ 20 | Balanced loss magnitudes |
| GT signal completely drowned out | Student can now learn from ground truth |

### 6. Temperature Too Low

| Issue | Fix |
|-------|-----|
| `T=1.0` gives sharp, peaked distributions | Enforce minimum `T=3.0` for KL computation |
| Large, noisy KL gradients | Softer distributions expose inter-class relationships |
| No "dark knowledge" transfer | Better knowledge transfer signal |

---

## ⚙️ Hyperparameters

### Default Values (CIFAR-100)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `hcd_loss_weight` | $\lambda$ | 6.0 | Weight for HCD KL loss (applied to per-stage average) |
| `gt_loss_weight` | $\omega$ | 1.0 | Weight for CE losses |
| `diversity` | $\beta$ | 1.0 | Weight for orthogonality loss |
| `k` | $k$ | 4 | Number of sub-logits |
| `temperature` | $T$ | 1.0 (min 3.0 enforced) | KL softmax temperature |
| `ortho_threshold` | $\theta$ | 0.5 | Orthogonality threshold |
| `lambda_student` | $\lambda_s$ | 1.0 | Sub-logit fusion weight |
| `lambda_teacher` | $\lambda_t$ | 1.0 | Teacher logit fusion weight |

### Training Settings (Official)

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning Rate | 0.05 |
| Weight Decay | 2e-3 |
| Scheduler | Cosine Annealing |
| Epochs | 300 |
| Batch Size | 128 |

---

## 💻 Implementation Details

### File: `src/models/kd_module.py`

#### Helper Functions

```python
def _hcd_remove_one_hot(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Zero out ground-truth class position before orthogonality computation.
    Uses zero-masking (NOT large negative values) to avoid float16 overflow.
    """
    B, k, C = logits.shape
    mask = torch.zeros_like(logits)
    mask.scatter_(2, labels.unsqueeze(1).unsqueeze(2).expand(B, k, 1), 1)
    masked_logits = logits * (1 - mask)  # Zero out label position
    return masked_logits


def _hcd_orthogonality_loss(vectors: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Squared penalty on pairwise similarity above threshold.
    """
    B, k, C = vectors.shape
    vectors = F.normalize(vectors, p=2, dim=-1)
    
    dot_product = torch.einsum('bik,bjk->bij', vectors, vectors)
    mask = torch.eye(k, device=vectors.device).bool()
    off_diagonal = dot_product[:, ~mask].view(B, k, k - 1)
    
    excess_sim = torch.relu(off_diagonal - threshold)
    loss = torch.mean(excess_sim.pow(2))
    return loss
```

#### HCDKDLitModule Class

```python
class HCDKDLitModule(pl.LightningModule):
    def __init__(
        self,
        teacher_models, student_model,
        temperature=1.0,
        hcd_loss_weight=6.0,
        gt_loss_weight=1.0,
        diversity=1.0,
        k=4,
        ortho_threshold=0.5,
        ...
    ):
        # Per-stage projectors (SepConv blocks)
        self.stage_projectors = nn.ModuleList([...])
        
        # Per-stage CFM
        self.cfm = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_s + feat_t, feat_t),
                nn.ReLU(),
                nn.Linear(feat_t, k * num_classes),
            )
            for _ in stages
        ])
        
        # Channel projectors registered properly (not dynamic setattr)
        self.channel_projs = nn.ModuleList([
            nn.Conv2d(student_final_dim, ch, 1) if ch != student_final_dim else nn.Identity()
            for ch in student_channels
        ])
        
        # _init_weights() only initializes stage_projectors and cfm
        # NOT the pretrained teacher or student
```

#### Loss Computation

```python
def compute_hcd_loss(self, student_logits, student_stage_features, 
                     teacher_features, teacher_logits, labels):
    hcd_losses, entropy_losses, orthogonality_losses = [], [], []
    T = max(self.temperature, 3.0)  # Enforce minimum temperature
    num_stages = len(student_stage_features)
    
    for stage_idx, (stage_feat, projector, cfm) in enumerate(...):
        # 1) Project student features
        feat_student_final = projector(stage_feat)
        
        # 2) Concatenate with teacher features
        feat_cat = torch.cat([feat_student_final, teacher_features], dim=1)
        
        # 3) CFM: produce k * num_classes logits
        logits_student_head = cfm(feat_cat).view(B, k, num_classes)
        
        # 4) Fuse with teacher logits (SDD)
        logits_student_head = λ_s * logits_student_head + λ_t * teacher_logits.unsqueeze(1)
        
        # 5) Orthogonality on masked sub-logits
        masked_logits = _hcd_remove_one_hot(logits_student_head, labels)
        orthogonality_losses.append(_hcd_orthogonality_loss(masked_logits))
        
        # 6) CE trains the CFM
        for i in range(k):
            entropy_losses.append(CE(logits_student_head[:, i], labels) / k)
        
        # 7) KL trains the STUDENT (sub_logits are DETACHED)
        sub_logits_detached = logits_student_head.detach()
        for i in range(k):
            hcd_losses.append(KL(student || detached_fused[:, i]) * T² / k)
    
    # AVERAGE across stages (not sum) to balance with GT loss
    loss_gt = gt_weight * (CE(student, labels) + sum(entropy) / num_stages)
    loss_hcd = hcd_weight * sum(hcd_losses) / num_stages
    loss_orth = diversity * sum(orthogonality) / num_stages
    
    return loss_gt + loss_hcd + loss_orth
```

---

## 📈 Results (CIFAR-100)

### Heterogeneous Distillation

| Teacher | Student | Baseline | KD | OFA | **HCD** |
|---------|---------|----------|-----|-----|---------|
| Swin-T | ResNet18 | 74.01 | 78.74 | 80.54 | **82.78** |
| ViT-S | ResNet18 | 74.01 | 77.82 | 79.31 | **81.33** |
| Swin-T | MobileNetV2 | 72.50 | 77.12 | 79.03 | **82.19** |

### Ablation: Number of Stages

| Stages Used | Top-1 Accuracy |
|-------------|----------------|
| None (baseline) | 74.01 |
| {1} | 81.82 (+7.81) |
| {1, 2} | 82.34 (+8.33) |
| {1, 2, 3} | 82.46 (+8.45) |
| {1, 2, 3, 4} | **82.78** (+8.77) |

### Ablation: Orthogonality Threshold

| θ | Swin-T→ResNet18 | ViT-S→ResNet18 |
|---|-----------------|----------------|
| 0.25 | 82.46 | 81.61 |
| **0.50** | **82.78** | 81.33 |
| 0.75 | 82.63 | 81.74 |

---

## 🔗 Config Example

```yaml
# configs/hcd_densenet.yaml
kd:
  type: "hcd"
  temperature: 1.0            # min 3.0 enforced internally for KL
  learning_rate: 0.05
  hcd_loss_weight: 6.0        # applied to per-stage AVERAGE
  gt_loss_weight: 1.0
  diversity: 1.0
  k: 4
  ortho_threshold: 0.5
  lambda_student: 1.0
  lambda_teacher: 1.0
  student_channels: [24, 32, 64, 1280]
  student_final_dim: 1280
  teacher_feature_dim: 384    # DenseNet-121 CIFAR (growth=12, blocks=(6,12,24,16))
```

---

## 🚀 Usage

```bash
# Train HCD with ViT teacher
python scripts/train.py --config configs/hcd_vit.yaml

# Train HCD with DenseNet teacher
python scripts/train.py --config configs/hcd_densenet.yaml

# Train HCD with ResNet-50 teacher
python scripts/train.py --config configs/hcd_resnet50.yaml
```

---

## 📚 Key Takeaways

1. **Complementary Features**: HCD leverages both teacher and student features, not just teacher logits
2. **Multi-Stage Processing**: Using all 4 stages provides +8.77% improvement over baseline
3. **Sub-logit Diversity**: Orthogonality loss prevents redundant knowledge transfer
4. **Simple Yet Effective**: CFM is just `Linear → ReLU → Linear`, no complex attention
5. **Teacher Logit Fusion**: Fusing teacher logits corrects classification, prevents drift
6. **Detach KL Targets**: Sub_logits must be detached in KL to prevent conflicting gradients between CE and KL
7. **Average, Don't Sum**: Per-stage losses must be averaged to prevent loss scale explosion
8. **AMP Safety**: Masking values must stay within float16 range (max 65504)

---

## 🔖 Tags

#knowledge-distillation #heterogeneous-kd #feature-distillation #cifar100 #vision-transformer #cnn #hcd

---

## 📎 References

- [HCD Paper (arXiv)](https://arxiv.org/abs/2511.10942)
- [Official GitHub](https://github.com/yema-web/HCD)
- [OFA-KD (NeurIPS 2023)](https://arxiv.org/abs/2310.05141)
- [PAT (arXiv 2501.08885)](https://arxiv.org/abs/2501.08885)

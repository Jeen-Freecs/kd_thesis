# PAT: Perspective-Aware Teaching for Heterogeneous Knowledge Distillation

> **Paper:** Lin et al. "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation" (arXiv:2501.08885)
> **GitHub:** https://github.com/jimmylin0979/PAT
> **PDF:** https://arxiv.org/pdf/2501.08885

---

## 🎯 Problem Statement

Traditional Knowledge Distillation assumes teacher and student have **similar architectures**. But heterogeneous setups (CNN ↔ ViT) face two critical problems:

### Problem 1: View Mismatch

| Architecture | Inductive Bias | Perspective |
|--------------|----------------|-------------|
| **MobileNet (Student)** | Locality (Convolution) | Sees "bricks" first, builds up slowly |
| **ViT (Teacher)** | Globality (Self-Attention) | Sees the "building" immediately |

The student **cannot mimic** teacher's global features because its receptive field is too small in early layers.

### Problem 2: Teacher Unawareness

The teacher is **frozen/static**. It provides complex features regardless of whether the student can understand them yet. Like a professor teaching quantum physics to a kindergartner.

---

## 💡 PAT Solution: Two Key Components

### 1️⃣ RAA: Region-Aware Attention
**"The Translator for the Student"**

**Goal:** Solve VIEW MISMATCH

**Mechanism:**
1. Extract features from **all 4 stages** of student CNN
2. **Patchify** features (like ViT tokens)
3. Apply **cross-stage self-attention** 
4. Student temporarily gains a "global view" for loss calculation
5. **Reblend** to spatial feature maps for comparison

```
Stage 1 Features ─┐
Stage 2 Features ─┼──► Patchify ──► Cross-Stage ──► Aligned
Stage 3 Features ─┤               Attention        Features
Stage 4 Features ─┘
```

> [!NOTE] 
> RAA doesn't change inference! It's only for training to align manifolds.

### 2️⃣ AFP: Adaptive Feedback Prompts
**"The Dynamic Tutor for the Teacher"**

**Goal:** Solve TEACHER UNAWARENESS

**Mechanism:**
1. **Error Profiling:** Compute `F_T - F_S` (where student failed)
2. **Prompt Injection:** Feed error into learnable prompt blocks
3. **Adaptation:** Teacher modifies features to help student
4. **Guardrail (L_Reg):** Prevent teacher from drifting too far

```
                    ┌──────────────┐
Teacher Features ──►│ Prompt Block │──► Adapted Features
                    └──────┬───────┘
                           ▲
                    Error Feedback
                    (T - S residue)
```

> [!WARNING] Risk of Collapse
> Without L_Reg, teacher might output "dumb" features to match student errors (blind leading the blind).

---

## 📐 Mathematical Formulation

### Total Loss Function

$$\mathcal{L}_{PAT} = \underbrace{\mathcal{L}_{CE}}_{\text{Task}} + \alpha \underbrace{\mathcal{L}_{KL}}_{\text{Logits}} + \beta \underbrace{\mathcal{L}_{FD}}_{\text{Features}} + \gamma \underbrace{\mathcal{L}_{Reg}}_{\text{Integrity}}$$

### Component Breakdown

**1. Cross-Entropy Loss (Task Performance)**
$$\mathcal{L}_{CE} = \text{CE}(z_S, y)$$

**2. KL Divergence Loss (Soft Logit Distillation)**
$$\mathcal{L}_{KL} = T^2 \cdot \text{KL}\left(\text{softmax}\left(\frac{z_T}{T}\right) \| \text{softmax}\left(\frac{z_S}{T}\right)\right)$$

**3. Feature Distillation Loss (RAA-Student vs AFP-Teacher)**
$$\mathcal{L}_{FD} = \text{MSE}\left(\text{norm}(F_S^{RAA}), \text{norm}(F_T^{AFP})\right)$$

**4. Regularization Loss (Teacher Integrity)**
$$\mathcal{L}_{Reg} = \text{MSE}\left(\text{norm}(F_T^{AFP}), \text{norm}(F_T^{frozen})\right)$$

---

## 🔄 Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Image x                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
    ┌───────────────┐                       ┌───────────────┐
    │    Student    │                       │    Teacher    │
    │  (trainable)  │                       │   (frozen)    │
    │  MobileNetV2  │                       │  ViT / ResNet │
    └───────┬───────┘                       └───────┬───────┘
            │                                       │
            ▼                                       ▼
┌─────────────────────┐                   ┌─────────────────────┐
│ Multi-Stage Features│                   │  Teacher Features   │
│ [Stage1...Stage4]   │                   │       F_T           │
└──────────┬──────────┘                   └──────────┬──────────┘
           │                                         │
           ▼                                         │
┌─────────────────────┐                              │
│        RAA          │                              │
│ Region-Aware Attn   │                              │
│ ─────────────────── │                              │
│ 1. Patchify stages  │                              │
│ 2. Cross-stage attn │                              │
│ 3. Get global view  │                              │
└──────────┬──────────┘                              │
           │                                         │
           ▼                                         │
┌─────────────────────┐                              │
│   Aligned Student   │                              │
│      F_S^RAA        │──────────┐                   │
└─────────────────────┘          │                   │
                                 │                   │
                                 ▼                   ▼
                         ┌─────────────────────────────────┐
                         │            AFP                   │
                         │   Adaptive Feedback Prompts      │
                         │ ─────────────────────────────── │
                         │ 1. Error = F_T - F_S             │
                         │ 2. Inject into prompts           │
                         │ 3. Adapt teacher output          │
                         └──────────────┬──────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────┐
                         │      Adapted Teacher F_T^AFP    │
                         └──────────────┬──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
            │   L_FD      │    │   L_Reg     │    │   L_KL      │
            │ Feature MSE │    │ Regularize  │    │ Logit KL    │
            └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │    L_CE     │◄──── Ground Truth y
                              └──────┬──────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │           TOTAL LOSS                 │
                    │  L = L_CE + α·L_KL + β·L_FD + γ·L_Reg │
                    └─────────────────────────────────────┘
                                     │
                                     ▼
                         Backprop to Student + RAA + AFP
```

---

## ⚙️ Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `temperature` | $T$ | 4.0 | Softmax temperature for KL |
| `alpha` | $\alpha$ | 1.0 | Weight for L_KL (logit distillation) |
| `beta` | $\beta$ | 1.0 | Weight for L_FD (feature distillation) |
| `gamma` | $\gamma$ | 0.1 | Weight for L_Reg (regularization) |
| `embed_dim` | - | 256 | RAA embedding dimension |
| `num_heads` | - | 8 | RAA attention heads |

### Teacher Feature Dimensions

| Teacher | `teacher_feature_dim` |
|---------|----------------------|
| ViT-Base | 768 |
| ResNet-50 | 2048 |
| ResNet-34/18 | 512 |
| DenseNet-121 (CIFAR) | 342 |

### Student Stage Channels (MobileNetV2)

```yaml
student_channels: [24, 32, 64, 1280]
```

---

## 📊 What Gets Logged to WandB

| Metric | Description |
|--------|-------------|
| `train/loss_ce` | Cross-entropy with ground truth |
| `train/loss_kl` | KL divergence (logit distillation) |
| `train/loss_fd` | Feature distillation (RAA vs AFP) |
| `train/loss_reg` | Regularization (AFP anchor) |
| `train/loss_total` | Combined total loss |
| `val/accuracy` | Validation accuracy |
| `val/auroc` | Validation AUROC |

---

## 🧪 Running PAT Experiments

### Available Configs

```bash
# PAT with DenseNet-121 teacher
python scripts/train.py --config configs/pat_densenet.yaml

# PAT with ResNet-50 teacher  
python scripts/train.py --config configs/pat_resnet50.yaml

# PAT with ViT teacher (MOST HETEROGENEOUS)
python scripts/train.py --config configs/pat_vit.yaml
```

### Run All Experiments

```bash
./run_all_experiments.sh
```

This runs:
1. Baseline (no KD)
2. Single-teacher Dynamic KD (×3 teachers)
3. **PAT (×3 teachers)** ← Heterogeneous KD

---

## 🔬 Why PAT Works

### The View Mismatch Problem

```
CNN (Student):     ViT (Teacher):
┌───┬───┬───┐      ┌─────────────┐
│ ▪ │ ▪ │ ▪ │      │             │
├───┼───┼───┤      │   Global    │
│ ▪ │ ▪ │ ▪ │      │    View     │
├───┼───┼───┤      │             │
│ ▪ │ ▪ │ ▪ │      └─────────────┘
└───┴───┴───┘
  Local only!        All at once!
```

**RAA Solution:** Give CNN a temporary "global view" via cross-stage attention.

### The Teacher Unawareness Problem

```
Without AFP:                  With AFP:
Teacher: "Here's calculus!"   Teacher: "Let me check your errors..."
Student: "I can't add yet!"   Student: "Thanks for adapting!"
         ❌ Mismatch                    ✓ Progressive
```

**AFP Solution:** Teacher adapts based on student's current errors.

---

## 📈 Expected Results

Based on the paper (CIFAR-100):

| Setup | Baseline | Standard KD | PAT | Improvement |
|-------|----------|-------------|-----|-------------|
| ViT → MobileNetV2 | 70.3% | 71.8% | **87.2%** | +15.4% |
| ResNet → MobileNetV2 | 70.3% | 73.5% | **78.9%** | +5.4% |
| ConvNeXt → Swin | 24.1% (FitNet) | 74.8% | **85.3%** | +10.5% |

PAT provides **up to 16.94% improvement** on CIFAR-100!

---

## 🔗 Comparison with Other Methods

| Method | Multi-Teacher | Feature KD | View Align | Teacher Adapt | Heterogeneous |
|--------|--------------|------------|------------|---------------|---------------|
| **Method 1 (CA-WKD)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Method 2 (Dynamic)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Method 3 (Confidence)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **OFA-KD** | ❌ | ✅ | ❌ | ❌ | ⚠️ Limited |
| **PAT** | ❌ | ✅ | ✅ (RAA) | ✅ (AFP) | ✅ Full |

**When to use PAT:**
- Single strong teacher (especially ViT)
- Heterogeneous architectures (CNN ↔ Transformer)
- Want maximum knowledge transfer across architecture gap

**When to use Methods 1-3:**
- Multiple teachers available
- Same architecture family
- Want ensemble diversity benefits

---

## 📝 Implementation Notes

### RAA Module

```python
class RegionAwareAttention(nn.Module):
    """
    1. Patchify multi-stage CNN features
    2. Project to shared dimension
    3. Apply cross-stage self-attention
    4. Pool to global representation
    """
    
    def forward(self, stage_features):
        # stage_features: [(B,C1,H1,W1), ..., (B,C4,H4,W4)]
        all_patches = []
        for i, feat in enumerate(stage_features):
            projected = self.stage_projectors[i](feat)
            patches = self.patchify(projected)
            patches += self.stage_embeddings[:, i:i+1, :]
            all_patches.append(patches)
        
        # Cross-stage attention
        all_patches = torch.cat(all_patches, dim=1)
        attended = self.cross_stage_attention(all_patches)
        
        return attended.mean(dim=1)  # Global pooling
```

### AFP Module

```python
class AdaptiveFeedbackPrompt(nn.Module):
    """
    1. Encode error signal (T - S)
    2. Modulate learnable prompts
    3. Project to teacher feature space
    4. Apply gated adaptation
    """
    
    def forward(self, teacher_feat, student_feat, frozen_feat):
        # Compute feedback
        feedback = teacher_feat - student_feat
        feedback_encoded = self.feedback_encoder(feedback)
        
        # Modulate prompts
        modulated = self.prompts + feedback_encoded
        adaptation = self.prompt_projector(modulated.mean(1))
        
        # Gated adaptation
        gate = self.gate(concat(teacher_feat, feedback_encoded))
        return teacher_feat + gate * adaptation
```

---

## 🎓 Key Takeaways

1. **RAA bridges the view gap** between CNN's local and ViT's global perspective
2. **AFP makes the teacher adaptive** to student's learning progress
3. **L_Reg prevents teacher collapse** by anchoring to frozen weights
4. **Best for heterogeneous setups** (CNN ↔ ViT)
5. **Up to 16.94% improvement** over standard KD on CIFAR-100

---

## 📚 References

```bibtex
@article{lin2025pat,
  title={Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation},
  author={Lin, Jhe-Hao and Yao, Yi and Hsu, Chan-Feng and Xie, Hong-Xia and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={arXiv preprint arXiv:2501.08885},
  year={2025}
}
```

---

#knowledge-distillation #PAT #heterogeneous-architectures #CNN-ViT #feature-distillation #prompt-tuning

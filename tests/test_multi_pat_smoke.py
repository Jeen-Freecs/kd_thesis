"""
Smoke tests for Multi-Teacher PAT Methods (A, B, D).

Uses lightweight dummy models to verify instantiation and forward pass
without downloading real teacher weights.
"""

import sys
import torch
import torch.nn as nn

# ── Dummy models ──────────────────────────────────────────────────────

class DummyMobileNetV2(nn.Module):
    """Mimics timm MobileNetV2 feature extraction interface."""

    def __init__(self, num_classes=100):
        super().__init__()
        # Stages produce feature maps with channels [24, 32, 96, 1280]
        self.conv_stem = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()

        # Need blocks at flat indices {2, 5, 12} to fire with channels 24, 32, 96
        # Build 13 groups × 1 block each so flat indices 0..12 exist
        channels = [16, 16, 24, 24, 24, 32, 32, 32, 32, 32, 32, 32, 96]
        self.blocks = nn.ModuleList()
        in_ch = 16
        for out_ch in channels:
            group = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                )
            ])
            self.blocks.append(group)
            in_ch = out_ch

        self.conv_head = nn.Conv2d(96, 1280, 1)
        self.bn2 = nn.BatchNorm2d(1280)
        self.act2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward_head(self, x):
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        for group in self.blocks:
            for block in group:
                x = block(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        return self.forward_head(x)


class DummyTeacher(nn.Module):
    """Generic teacher that returns (B, D, H, W) features + logits."""

    def __init__(self, feature_dim: int, num_classes: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x):
        return self.backbone(x)  # (B, D, H, W)

    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


# ── Helpers ───────────────────────────────────────────────────────────

def make_batch(batch_size=4, num_teachers=3, img_size=32):
    """Create a fake batch matching DualTransformDataset output."""
    batch = {
        'student_input': torch.randn(batch_size, 3, img_size, img_size),
        'label': torch.randint(0, 100, (batch_size,)),
    }
    for k in range(num_teachers):
        batch[f'teacher_input_{k}'] = torch.randn(batch_size, 3, img_size, img_size)
    return batch


def test_method(cls, extra_kwargs=None, method_name=""):
    """Instantiate a module, run training_step + validation_step + backward."""
    teacher_dims = [384, 2048, 768]
    teachers = [DummyTeacher(d) for d in teacher_dims]
    student = DummyMobileNetV2()

    kwargs = dict(
        teacher_models=teachers,
        student_model=student,
        temperature=4.0,
        learning_rate=1e-3,
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
        num_classes=100,
        student_channels=[24, 32, 96, 1280],
        teacher_feature_dims=teacher_dims,
        embed_dim=256,
        num_heads=8,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    module = cls(**kwargs)
    module.train()

    batch = make_batch(batch_size=4, num_teachers=3)

    # Training step
    loss = module.training_step(batch, batch_idx=0)
    assert loss is not None and torch.isfinite(loss), f"{method_name}: loss not finite: {loss}"
    print(f"  ✓ training_step  loss = {loss.item():.4f}")

    # Validation step
    module.eval()
    with torch.no_grad():
        acc = module.shared_eval_step(batch, 'val')
    print(f"  ✓ shared_eval_step  acc = {acc.item():.4f}")

    # Optimizer creation
    opts = module.configure_optimizers()
    assert len(opts) == 2 and len(opts[0]) == 1 and len(opts[1]) == 1
    print(f"  ✓ configure_optimizers")

    # Backward pass
    module.train()
    loss2 = module.training_step(batch, batch_idx=1)
    loss2.backward()
    print(f"  ✓ backward pass")

    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, '/workspace/kd_thesis')
    from src.models.kd_module import (
        TeacherFusionAttention,
        MultiTeacherPATIndLitModule,
        MultiTeacherPATCWLitModule,
        MultiTeacherPATAttnLitModule,
    )

    all_passed = True

    # --- TeacherFusionAttention standalone ---
    print("\n═══ TeacherFusionAttention ═══")
    tfa = TeacherFusionAttention(
        student_dim=256, teacher_dims=[384, 2048, 768],
        attn_dim=256, num_heads=4,
    )
    s_feat = torch.randn(4, 256)
    t_feats = [torch.randn(4, 384), torch.randn(4, 2048), torch.randn(4, 768)]
    fused, attn_w = tfa(s_feat, t_feats)
    assert fused.shape == (4, 256), f"fused shape: {fused.shape}"
    assert attn_w.shape == (4, 3), f"attn_w shape: {attn_w.shape}"
    assert torch.allclose(attn_w.sum(dim=1), torch.ones(4), atol=1e-5)
    print(f"  ✓ fused={fused.shape}, attn_weights={attn_w.shape}, sums to 1")

    # --- Method A: PAT-Ind ---
    print("\n═══ Method A: PAT-Ind ═══")
    try:
        test_method(MultiTeacherPATIndLitModule, method_name="PAT-Ind")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # --- Method B: PAT-CW ---
    print("\n═══ Method B: PAT-CW ═══")
    try:
        test_method(MultiTeacherPATCWLitModule, method_name="PAT-CW")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # --- Method D: PAT-Attn ---
    print("\n═══ Method D: PAT-Attn ═══")
    try:
        test_method(
            MultiTeacherPATAttnLitModule,
            extra_kwargs=dict(attn_dim=256, attn_heads=4, entropy_weight=0.01),
            method_name="PAT-Attn",
        )
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # --- Summary ---
    print("\n" + "═" * 50)
    if all_passed:
        print("ALL SMOKE TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == '__main__':
    main()

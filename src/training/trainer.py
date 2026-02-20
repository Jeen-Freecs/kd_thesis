"""Training utilities for knowledge distillation"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Dict, Any
import torch

from ..models.kd_module import (
    CAWeightedKDLitModule,
    DynamicKDLitModule,
    ConfidenceBasedKDLitModule,
    PATKDLitModule,
    HCDKDLitModule,
    OFAKDLitModule,
    BaselineStudentModule,
    MultiTeacherPATIndLitModule,
    MultiTeacherPATCWLitModule,
    MultiTeacherPATAttnLitModule,
)
from ..data.datamodule import CIFAR100DataModule
from ..utils.logger import get_logger

logger = get_logger()


def train_kd_model(
    kd_module: pl.LightningModule,
    data_module: CIFAR100DataModule,
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None
) -> pl.LightningModule:
    """
    Train knowledge distillation model.
    
    Args:
        kd_module: Knowledge distillation Lightning module
        data_module: Data module for CIFAR-100
        config: Training configuration dictionary
        checkpoint_path: Path to checkpoint to resume from (optional)
        
    Returns:
        Trained Lightning module
    """
    # Extract training config
    train_config = config.get('training', {})
    max_epochs = train_config.get('max_epochs', 100)
    patience = train_config.get('patience', 15)
    log_every_n_steps = train_config.get('log_every_n_steps', 50)
    
    # WandB logger
    wandb_config = config.get('wandb', {})
    experiment_name = wandb_config.get('name', 'KD-Experiment')
    # log_model=False: Don't auto-upload every checkpoint to WandB artifacts.
    # We save checkpoints locally and only the best/last are kept by ModelCheckpoint.
    # This prevents WandB storage from filling up with dozens of artifact versions.
    # Use scripts/cleanup_wandb_artifacts.py to clean up existing artifacts.
    wandb_logger = WandbLogger(
        project=wandb_config.get('project', 'Knowledge-Distillation-CIFAR100'),
        name=experiment_name,
        log_model=wandb_config.get('log_model', False),
        resume=wandb_config.get('resume', 'allow')
    )
    
    # Log hyperparameters to WandB for tracking
    model_config = config.get('model', {})
    kd_config = config.get('kd', {})
    data_config = config.get('data', {})
    
    wandb_logger.experiment.config.update({
        # Model configuration
        "student_model": model_config.get('student_name', 'unknown'),
        "teacher_models": model_config.get('teacher_names', []),
        "num_classes": model_config.get('num_classes', 100),
        "student_pretrained": model_config.get('student_pretrained', False),
        
        # KD configuration
        "kd_type": kd_config.get('type', 'baseline'),
        "temperature": kd_config.get('temperature', None),
        "alpha": kd_config.get('alpha', None),
        "gamma": kd_config.get('gamma', None),
        "threshold": kd_config.get('threshold', None),
        "learning_rate": kd_config.get('learning_rate', 0.001),
        "use_soft_loss": kd_config.get('use_soft_loss', True),
        "use_hard_loss": kd_config.get('use_hard_loss', True),
        
        # Training configuration
        "max_epochs": max_epochs,
        "patience": patience,
        "log_every_n_steps": log_every_n_steps,
        
        # Data configuration
        "batch_size": data_config.get('batch_size', 128),
        "val_size": data_config.get('val_size', 5000),
        "num_workers": data_config.get('num_workers', 4),
        "seed": data_config.get('seed', 42),
    })
    
    # Get WandB run ID for organizing checkpoints
    run_id = wandb_logger.experiment.id
    
    # Create experiment-specific checkpoint directory with run ID
    kd_type = config.get('kd', {}).get('type', 'unknown')
    checkpoint_dir = f'checkpoints/{kd_type}/{experiment_name}/{run_id}'
    
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"WandB Run ID: {run_id}")
    
    # Callbacks - Save best and latest checkpoints
    callbacks = [
        EarlyStopping(
            monitor='val/accuracy',
            patience=patience,
            mode='max',
            verbose=True
        ),
        # Save best checkpoint based on validation accuracy
        # Only keep the single best checkpoint and the last checkpoint
        # Note: Using simple filenames to avoid directory creation from metric names with slashes
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val/accuracy',
            mode='max',
            save_top_k=1,
            filename='best',
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True
        )
    ]
    
    # Trainer
    # Note: With WandB logger, Lightning won't create lightning_logs/ directory
    # All logging goes to WandB, checkpoints go to our custom directory
    # Gradient clipping: prevents training collapse from gradient explosion,
    # especially important for PAT with real intermediate features where
    # feature loss gradients flow back through the entire student backbone.
    gradient_clip_val = train_config.get('gradient_clip_val', 1.0)
    
    # Precision: default to fp16 mixed precision for speed, but allow fp32
    # via config. OFA with SGD lr=0.05 requires fp32 (official doesn't use AMP).
    precision_cfg = train_config.get('precision', None)
    if precision_cfg is not None:
        precision = precision_cfg  # Use explicit config value (e.g., 32 or '16-mixed')
    elif torch.cuda.is_available():
        precision = '16-mixed'
    else:
        precision = 32
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,  # WandB logger prevents lightning_logs/ creation
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    logger.info("Starting training...")
    if checkpoint_path:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.fit(kd_module, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(kd_module, datamodule=data_module)
    
    logger.info("Training completed!")
    
    return kd_module


def create_kd_module_from_config(
    config: Dict[str, Any],
    teacher_models,
    student_model
) -> pl.LightningModule:
    """
    Create KD module from configuration.
    
    Supports all three methods from the AML Final Project:
    - 'ca_weighted': Method 1 - CA-WKD (Confidence-Aware Weighted KD)
    - 'dynamic': Method 2 - α-Guided CA-WKD (with gating)
    - 'confidence': Method 3 - Adaptive α-Guided KD (most confident teacher)
    - 'baseline': Baseline student without KD
    - 'pat': PAT for heterogeneous distillation
    - 'hcd': HCD for heterogeneous complementary distillation
    
    Args:
        config: Configuration dictionary
        teacher_models: List of teacher models (can be None/empty for baseline)
        student_model: Student model
        
    Returns:
        Knowledge distillation Lightning module
    """
    kd_config = config.get('kd', {})
    kd_type = kd_config.get('type', 'dynamic')
    
    # Baseline: Student without KD
    if kd_type == 'baseline':
        kd_module = BaselineStudentModule(
            student_model=student_model,
            learning_rate=kd_config.get('learning_rate', 1e-2),
            num_classes=kd_config.get('num_classes', 100)
        )
        return kd_module
    
    # Common parameters for all KD methods
    common_params = {
        'teacher_models': teacher_models,
        'student_model': student_model,
        'temperature': kd_config.get('temperature', 4.0),
        'learning_rate': kd_config.get('learning_rate', 1e-2),
        'num_classes': kd_config.get('num_classes', 100)
    }
    
    # Method 1: CA-WKD (Confidence-Aware Weighted KD)
    if kd_type == 'ca_weighted':
        kd_module = CAWeightedKDLitModule(**common_params)
    
    # Method 2: α-Guided CA-WKD (Dynamic with gating)
    elif kd_type == 'dynamic':
        kd_module = DynamicKDLitModule(
            **common_params,
            gamma=kd_config.get('gamma', 10.0),
            threshold=kd_config.get('threshold', 0.5),
            alpha=kd_config.get('alpha', 0.5),
            use_soft_loss=kd_config.get('use_soft_loss', True),
            use_hard_loss=kd_config.get('use_hard_loss', True)
        )
    
    # Method 3: Adaptive α-Guided KD (Confidence-based)
    elif kd_type == 'confidence':
        kd_module = ConfidenceBasedKDLitModule(
            **common_params,
            use_soft_loss=kd_config.get('use_soft_loss', True),
            use_hard_loss=kd_config.get('use_hard_loss', True)
        )
    
    # PAT: Perspective-Aware Teaching (arXiv:2501.08885)
    # For heterogeneous teacher-student architectures (CNN ↔ ViT)
    elif kd_type == 'pat':
        kd_module = PATKDLitModule(
            **common_params,
            alpha=kd_config.get('alpha', 1.0),           # L_KL weight
            beta=kd_config.get('beta', 1.0),             # L_FD weight
            gamma=kd_config.get('gamma', 0.1),           # L_Reg weight
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            teacher_feature_dim=kd_config.get('teacher_feature_dim', 768),
            embed_dim=kd_config.get('embed_dim', 256),
            num_heads=kd_config.get('num_heads', 8),
        )
    
    # Multi-Teacher PAT-Ind (Method A): Independent channels
    elif kd_type == 'multi_pat_ind':
        kd_module = MultiTeacherPATIndLitModule(
            **common_params,
            alpha=kd_config.get('alpha', 1.0),
            beta=kd_config.get('beta', 1.0),
            gamma=kd_config.get('gamma', 0.1),
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            teacher_feature_dims=kd_config.get('teacher_feature_dims', [768]),
            embed_dim=kd_config.get('embed_dim', 256),
            num_heads=kd_config.get('num_heads', 8),
        )
    
    # Multi-Teacher PAT-CW (Method B): Confidence-Weighted
    elif kd_type == 'multi_pat_cw':
        kd_module = MultiTeacherPATCWLitModule(
            **common_params,
            alpha=kd_config.get('alpha', 1.0),
            beta=kd_config.get('beta', 1.0),
            gamma=kd_config.get('gamma', 0.1),
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            teacher_feature_dims=kd_config.get('teacher_feature_dims', [768]),
            embed_dim=kd_config.get('embed_dim', 256),
            num_heads=kd_config.get('num_heads', 8),
        )
    
    # Multi-Teacher PAT-Attn (Method D): Attention-Based Fusion
    elif kd_type == 'multi_pat_attn':
        kd_module = MultiTeacherPATAttnLitModule(
            **common_params,
            alpha=kd_config.get('alpha', 1.0),
            beta=kd_config.get('beta', 1.0),
            gamma=kd_config.get('gamma', 0.1),
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            teacher_feature_dims=kd_config.get('teacher_feature_dims', [768]),
            embed_dim=kd_config.get('embed_dim', 256),
            num_heads=kd_config.get('num_heads', 8),
            attn_dim=kd_config.get('attn_dim', 256),
            attn_heads=kd_config.get('attn_heads', 4),
            entropy_weight=kd_config.get('entropy_weight', 0.01),
        )
    
    # HCD: Heterogeneous Complementary Distillation (arXiv:2511.10942)
    # Official implementation: https://github.com/yema-web/HCD
    elif kd_type == 'hcd':
        kd_module = HCDKDLitModule(
            **common_params,
            hcd_loss_weight=kd_config.get('hcd_loss_weight', 6.0),
            kd_loss_weight=kd_config.get('kd_loss_weight', 1.0),
            gt_loss_weight=kd_config.get('gt_loss_weight', 1.0),
            diversity=kd_config.get('diversity', 1.0),
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            student_final_dim=kd_config.get('student_final_dim', 1280),
            teacher_feature_dim=kd_config.get('teacher_feature_dim', 768),
            embed_dim=kd_config.get('embed_dim', 256),
            k=kd_config.get('k', 4),
            ortho_threshold=kd_config.get('ortho_threshold', 0.5),
            lambda_student=kd_config.get('lambda_student', 1.0),
            lambda_teacher=kd_config.get('lambda_teacher', 1.0),
            label_smoothing=kd_config.get('label_smoothing', 0.1),
        )
    
    # OFA-KD: One-for-All Knowledge Distillation (NeurIPS 2023)
    # Official implementation: https://github.com/Hao840/OFAKD
    elif kd_type == 'ofa':
        train_config = config.get('training', {})
        kd_module = OFAKDLitModule(
            **common_params,
            ofa_eps=kd_config.get('ofa_eps', 1.0),
            ofa_loss_weight=kd_config.get('ofa_loss_weight', 1.0),
            gt_loss_weight=kd_config.get('gt_loss_weight', 1.0),
            kd_loss_weight=kd_config.get('kd_loss_weight', 1.0),
            ofa_temperature=kd_config.get('ofa_temperature', 1.0),
            label_smoothing=kd_config.get('label_smoothing', 0.1),
            student_channels=kd_config.get('student_channels', [24, 32, 96, 1280]),
            student_final_dim=kd_config.get('student_final_dim', 1280),
            teacher_feature_dim=kd_config.get('teacher_feature_dim', 768),
            warmup_epochs=kd_config.get('warmup_epochs', 3),
            max_epochs=train_config.get('max_epochs', 300),
            teacher_input_size=kd_config.get('teacher_input_size', None),
        )
    
    else:
        raise ValueError(
            f"Unknown KD type: {kd_type}. "
            f"Supported types: 'baseline', 'ca_weighted', 'dynamic', 'confidence', "
            f"'pat', 'multi_pat_ind', 'multi_pat_cw', 'multi_pat_attn', "
            f"'hcd', 'ofa'"
        )
    
    return kd_module


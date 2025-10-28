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
    BaselineStudentModule
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
    wandb_logger = WandbLogger(
        project=wandb_config.get('project', 'Knowledge-Distillation-CIFAR100'),
        name=experiment_name,
        log_model='all',  # Upload all checkpoints (best + latest = 2 files)
        resume=wandb_config.get('resume', 'allow')
    )
    
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
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val/accuracy',
            mode='max',
            save_top_k=1,
            filename='best-epoch={epoch:02d}-val_acc={val/accuracy:.4f}',
            save_last=False,
            verbose=True
        ),
        # Save latest checkpoint (last epoch)
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=0,
            save_last=True,
            filename='latest',
            verbose=True
        )
    ]
    
    # Trainer
    # Note: With WandB logger, Lightning won't create lightning_logs/ directory
    # All logging goes to WandB, checkpoints go to our custom directory
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,  # WandB logger prevents lightning_logs/ creation
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
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
    
    else:
        raise ValueError(
            f"Unknown KD type: {kd_type}. "
            f"Supported types: 'baseline', 'ca_weighted', 'dynamic', 'confidence'"
        )
    
    return kd_module


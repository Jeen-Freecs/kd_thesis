"""Training utilities for knowledge distillation"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Dict, Any
import torch

from ..models.kd_module import DynamicKDLitModule, ConfidenceBasedKDLitModule
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
    wandb_logger = WandbLogger(
        project=wandb_config.get('project', 'Knowledge-Distillation-CIFAR100'),
        name=wandb_config.get('name', 'KD-Experiment'),
        log_model=wandb_config.get('log_model', 'all'),
        resume=wandb_config.get('resume', 'allow')
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val/accuracy',
            patience=patience,
            mode='max',
            verbose=True
        ),
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            filename='kd-{epoch:02d}-{val_accuracy:.4f}'
        )
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
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
    
    Args:
        config: Configuration dictionary
        teacher_models: List of teacher models
        student_model: Student model
        
    Returns:
        Knowledge distillation Lightning module
    """
    kd_config = config.get('kd', {})
    kd_type = kd_config.get('type', 'dynamic')
    
    common_params = {
        'teacher_models': teacher_models,
        'student_model': student_model,
        'temperature': kd_config.get('temperature', 4.0),
        'learning_rate': kd_config.get('learning_rate', 1e-3),
        'use_soft_loss': kd_config.get('use_soft_loss', True),
        'use_hard_loss': kd_config.get('use_hard_loss', True),
        'num_classes': kd_config.get('num_classes', 100)
    }
    
    if kd_type == 'dynamic':
        kd_module = DynamicKDLitModule(
            **common_params,
            gamma=kd_config.get('gamma', 10.0),
            threshold=kd_config.get('threshold', 0.5),
            alpha=kd_config.get('alpha', 0.5)
        )
    elif kd_type == 'confidence':
        kd_module = ConfidenceBasedKDLitModule(**common_params)
    else:
        raise ValueError(f"Unknown KD type: {kd_type}")
    
    return kd_module


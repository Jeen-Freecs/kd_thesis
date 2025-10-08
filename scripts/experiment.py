#!/usr/bin/env python3
"""
Run experiments with different configurations.

Usage:
    python scripts/experiment.py --config configs/config.yaml --exp-name my_experiment
"""

import argparse
import torch
from pathlib import Path
import sys
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CIFAR100DataModule
from src.models import create_student_model, create_teacher_models
from src.training import train_kd_model, create_kd_module_from_config
from src.utils import load_config, setup_logger


def run_experiment(config, exp_name, checkpoint_path=None, device='cuda'):
    """
    Run a single experiment.
    
    Args:
        config: Configuration dictionary
        exp_name: Experiment name
        checkpoint_path: Path to checkpoint to resume from
        device: Device to use
    """
    logger = setup_logger(log_file=f'logs/experiment_{exp_name}.log')
    logger.info(f"Starting experiment: {exp_name}")
    
    # Update WandB config with experiment name
    config['wandb']['name'] = exp_name
    config['wandb']['project'] = config.get('wandb', {}).get(
        'project', 
        f'Knowledge-Distillation-{exp_name}'
    )
    
    # Generate run ID if not resuming
    if not checkpoint_path:
        run_id = wandb.util.generate_id()
        config['wandb']['id'] = run_id
        logger.info(f"Generated run ID: {run_id}")
    
    # Create data module
    data_config = config.get('data', {})
    logger.info("Creating data module...")
    data_module = CIFAR100DataModule(
        data_dir=data_config.get('data_dir', './data'),
        batch_size=data_config.get('batch_size', 128),
        num_workers=data_config.get('num_workers', 4),
        teacher_models=data_config.get('teacher_model_types', ['resnet']),
        student_model=data_config.get('student_model_type', 'mobilenet'),
        pre_trained=data_config.get('pre_trained', False),
        val_size=data_config.get('val_size', 5000),
        seed=data_config.get('seed', 42)
    )
    
    # Create student model
    model_config = config.get('model', {})
    logger.info("Creating student model...")
    student_model = create_student_model(
        model_name=model_config.get('student_name', 'mobilenetv2_100'),
        num_classes=model_config.get('num_classes', 100),
        pretrained=model_config.get('student_pretrained', False)
    )
    
    # Create teacher models
    teacher_names = model_config.get('teacher_names', ['resnet50_cifar100'])
    logger.info(f"Creating teacher models: {teacher_names}")
    teacher_models = create_teacher_models(
        teacher_names=teacher_names,
        num_classes=model_config.get('num_classes', 100),
        device=device
    )
    
    # Create KD module
    logger.info("Creating KD module...")
    kd_module = create_kd_module_from_config(
        config=config,
        teacher_models=teacher_models,
        student_model=student_model
    )
    
    # Train
    logger.info("Starting training...")
    trained_module = train_kd_model(
        kd_module=kd_module,
        data_module=data_module,
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    logger.info(f"Experiment {exp_name} completed!")
    
    # Finish wandb run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run Knowledge Distillation Experiments")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        required=True,
        help='Experiment name'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    run_experiment(
        config=config,
        exp_name=args.exp_name,
        checkpoint_path=args.checkpoint,
        device=args.device
    )


if __name__ == '__main__':
    main()


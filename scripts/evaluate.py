#!/usr/bin/env python3
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt --config configs/config.yaml
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CIFAR100DataModule
from src.models import create_student_model
from src.evaluation import evaluate_model, load_model_from_checkpoint
from src.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Distillation Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Which split to evaluate on'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(log_file='logs/evaluate.log')
    logger.info("Starting evaluation script...")
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create data module
    data_config = config.get('data', {})
    logger.info("Creating data module...")
    data_module = CIFAR100DataModule(
        data_dir=data_config.get('data_dir', './data'),
        batch_size=data_config.get('batch_size', 128),
        num_workers=data_config.get('num_workers', 4),
        student_model=data_config.get('student_model_type', 'mobilenet'),
        pre_trained=data_config.get('pre_trained', False),
        val_size=data_config.get('val_size', 5000),
        seed=data_config.get('seed', 42)
    )
    
    # Prepare data
    data_module.prepare_data()
    data_module.setup(stage='test' if args.split == 'test' else 'fit')
    
    # Get appropriate dataloader
    if args.split == 'test':
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()
    
    # Create student model
    model_config = config.get('model', {})
    logger.info("Creating student model...")
    student_model = create_student_model(
        model_name=model_config.get('student_name', 'mobilenetv2_100'),
        num_classes=model_config.get('num_classes', 100),
        pretrained=False
    )
    
    # Load from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract student model state dict
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('student.'):
            name = k.replace('student.', '')
            new_state_dict[name] = v
    
    student_model.load_state_dict(new_state_dict, strict=True)
    
    # Evaluate
    logger.info(f"Evaluating on {args.split} set...")
    results = evaluate_model(
        model=student_model,
        dataloader=dataloader,
        device=args.device
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  Loss: {results['loss']:.4f}")
    logger.info(f"  Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"  Correct: {results['correct']}/{results['total']}")
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Split: {args.split}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()


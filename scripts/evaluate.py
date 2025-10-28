#!/usr/bin/env python3
"""
Evaluation script for trained models.

Usage:
    # Evaluate from local checkpoint
    python scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt --config configs/method2_dynamic_kd.yaml
    
    # Evaluate from WandB artifact
    python scripts/evaluate.py --wandb-artifact username/project/artifact:version --config configs/method2_dynamic_kd.yaml
"""

import argparse
import torch
from pathlib import Path
import sys
import wandb
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CIFAR100DataModule
from src.models import create_student_model
from src.evaluation import evaluate_model, load_model_from_checkpoint
from src.utils import load_config, setup_logger


def download_wandb_checkpoint(artifact_path, logger):
    """
    Download checkpoint from WandB artifact.
    
    Args:
        artifact_path: WandB artifact path (e.g., 'username/project/model-best:v0')
        logger: Logger instance
    
    Returns:
        Path to downloaded checkpoint file
    """
    logger.info(f"Downloading checkpoint from WandB: {artifact_path}")
    
    try:
        # Initialize WandB API
        api = wandb.Api()
        
        # Get artifact
        artifact = api.artifact(artifact_path)
        
        # Download to temporary directory
        artifact_dir = artifact.download()
        
        # Find checkpoint file in downloaded directory
        checkpoint_files = list(Path(artifact_dir).glob('*.ckpt'))
        
        if not checkpoint_files:
            raise ValueError(f"No .ckpt files found in artifact: {artifact_path}")
        
        checkpoint_path = str(checkpoint_files[0])
        logger.info(f"Downloaded checkpoint to: {checkpoint_path}")
        
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint from WandB: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Distillation Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to local model checkpoint'
    )
    parser.add_argument(
        '--wandb-artifact',
        type=str,
        default=None,
        help='WandB artifact path (e.g., username/project/model-best:v0)'
    )
    parser.add_argument(
        '--wandb-run-id',
        type=str,
        default=None,
        help='WandB run ID to download best checkpoint from (e.g., l9plp8p2)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/method2_dynamic_kd.yaml',
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
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging for evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(log_file='logs/archive/evaluate.log')
    logger.info("Starting evaluation script...")
    
    # Validate arguments
    if not args.checkpoint and not args.wandb_artifact and not args.wandb_run_id:
        parser.error("Must provide either --checkpoint, --wandb-artifact, or --wandb-run-id")
    
    # Determine checkpoint path
    checkpoint_path = None
    
    if args.checkpoint:
        # Use local checkpoint
        checkpoint_path = args.checkpoint
        logger.info(f"Using local checkpoint: {checkpoint_path}")
    
    elif args.wandb_artifact:
        # Download from WandB artifact
        checkpoint_path = download_wandb_checkpoint(args.wandb_artifact, logger)
    
    elif args.wandb_run_id:
        # Download best checkpoint from specific run
        config = load_config(args.config)
        wandb_config = config.get('wandb', {})
        project = wandb_config.get('project', 'Knowledge-Distillation-CIFAR100')
        
        # Try to find the best model artifact for this run
        artifact_path = f"{project}/model-{args.wandb_run_id}:best"
        logger.info(f"Attempting to download from run: {args.wandb_run_id}")
        
        try:
            checkpoint_path = download_wandb_checkpoint(artifact_path, logger)
        except:
            # Fallback: try latest version
            logger.warning(f"Could not find 'best' artifact, trying 'latest'")
            artifact_path = f"{project}/model-{args.wandb_run_id}:latest"
            checkpoint_path = download_wandb_checkpoint(artifact_path, logger)
    
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
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
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
    
    # Log results to WandB (unless disabled)
    if not args.no_wandb and checkpoint_path:
        logger.info("Logging evaluation results to WandB...")
        
        try:
            # Initialize WandB for logging evaluation results
            wandb_config = config.get('wandb', {})
            
            # Extract run ID from checkpoint path or arguments
            run_id_for_name = args.wandb_run_id if args.wandb_run_id else checkpoint_path.split('/')[-2] if '/' in checkpoint_path else 'unknown'
            
            # Get model config for logging metadata
            model_config = config.get('model', {})
            kd_config = config.get('kd', {})
            
            eval_run = wandb.init(
                project=wandb_config.get('project', 'Knowledge-Distillation-CIFAR100'),
                name=f"eval-{args.split}-{run_id_for_name}",
                job_type="evaluation",
                tags=[args.split, "evaluation", run_id_for_name, kd_config.get('type', 'baseline')],
                config={
                    "config_file": args.config,
                    "checkpoint": checkpoint_path,
                    "split": args.split,
                    "device": args.device,
                    "run_id": run_id_for_name,
                    # Add model metadata for filtering/comparison
                    "student_model": model_config.get('student_name', 'unknown'),
                    "teacher_models": model_config.get('teacher_names', []),
                    "kd_type": kd_config.get('type', 'baseline'),
                    "temperature": kd_config.get('temperature', None),
                }
            )
            
            # Log evaluation metrics
            wandb.log({
                f"{args.split}/loss": results['loss'],
                f"{args.split}/accuracy": results['accuracy'],
                f"{args.split}/accuracy_percent": results['accuracy'] * 100,
                f"{args.split}/correct": results['correct'],
                f"{args.split}/total": results['total']
            })
            
            # Create summary with comprehensive metadata
            wandb.summary.update({
                # Metrics
                f"{args.split}_loss": results['loss'],
                f"{args.split}_accuracy": results['accuracy'],
                f"{args.split}_accuracy_percent": results['accuracy'] * 100,
                # Metadata for easy filtering
                "checkpoint_path": checkpoint_path,
                "model_name": model_config.get('student_name', 'unknown'),
                "teachers": ', '.join(model_config.get('teacher_names', [])),
                "kd_method": kd_config.get('type', 'baseline'),
                "num_teachers": len(model_config.get('teacher_names', [])),
            })
            
            wandb.finish()
            logger.info("✓ Results logged to WandB")
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")
            logger.info("Continuing without WandB logging...")
    
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


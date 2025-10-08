"""Model evaluation utilities"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_class: type,
    student_model: nn.Module,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load a student model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_class: The Lightning module class
        student_model: Student model architecture
        device: Device to load model on
        
    Returns:
        Loaded student model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract student model state dict
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('student.'):
            name = k.replace('student.', '')
            new_state_dict[name] = v
    
    student_model.load_state_dict(new_state_dict, strict=True)
    student_model.to(device)
    student_model.eval()
    
    return student_model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle both dict and tuple batch formats
            if isinstance(batch, dict):
                data = batch['student_input'].to(device)
                target = batch['label'].to(device)
            else:
                data, target = batch
                data = data.to(device)
                target = target.to(device)
            
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    return results


class EvaluationLightningModule(pl.LightningModule):
    """
    Lightning module for evaluating pretrained models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize evaluation module.
        
        Args:
            model: Pretrained model to evaluate
        """
        super().__init__()
        self.model = model
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def configure_optimizers(self):
        """No optimizer needed for evaluation."""
        return None
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, dict):
            images = batch['student_input']
            labels = batch['label']
        else:
            images, labels = batch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        logits = self.model(images)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        if isinstance(batch, dict):
            images = batch['student_input']
            labels = batch['label']
        else:
            images, labels = batch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        logits = self.model(images)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc


"""Teacher model creation and loading"""

from typing import List
import torch
import torch.nn as nn
import timm


def create_teacher_models(
    teacher_names: List[str],
    num_classes: int = 100,
    device: str = 'cuda'
) -> List[nn.Module]:
    """
    Create and load pretrained teacher models.
    
    Args:
        teacher_names: List of teacher model names
        num_classes: Number of output classes
        device: Device to load models on
        
    Returns:
        List[nn.Module]: List of loaded teacher models
    """
    teacher_models = []
    
    for name in teacher_names:
        # Handle Vision Transformer variants
        if name == 'vit' or 'vit_base_patch16_224' in name.lower():
            teacher = _load_vit_teacher(num_classes, device)
        else:
            try:
                teacher = timm.create_model(name, pretrained=True)
            except Exception as e:
                raise ValueError(f"Invalid teacher model name: {name}. Error: {e}")
        
        teacher = teacher.to(device)
        teacher.eval()
        
        # Freeze teacher parameters
        for param in teacher.parameters():
            param.requires_grad = False
            
        teacher_models.append(teacher)
    
    return teacher_models


def _load_vit_teacher(num_classes: int, device: str) -> nn.Module:
    """
    Load Vision Transformer teacher model.
    
    Args:
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        nn.Module: Loaded ViT teacher model
    """
    teacher = timm.create_model(
        "timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=False
    )
    teacher.head = nn.Linear(teacher.head.in_features, num_classes)
    
    # Load pretrained weights
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
        map_location=device,
        file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
    )
    teacher.load_state_dict(state_dict)
    
    return teacher


class TeacherEnsemble(nn.Module):
    """
    Ensemble wrapper for multiple teacher models.
    """
    
    def __init__(self, teachers: List[nn.Module]):
        """
        Initialize teacher ensemble.
        
        Args:
            teachers: List of teacher models
        """
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.num_teachers = len(teachers)
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through all teachers.
        
        Args:
            inputs: List of inputs for each teacher
            
        Returns:
            List of logits from each teacher
        """
        outputs = []
        for i, teacher in enumerate(self.teachers):
            with torch.no_grad():
                outputs.append(teacher(inputs[i]))
        return outputs
    
    def freeze(self):
        """Freeze all teacher parameters."""
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False


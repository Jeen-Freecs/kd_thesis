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
    Create and load pretrained teacher models from HuggingFace.
    
    Args:
        teacher_names: List of teacher model names (e.g., 'resnet50_cifar100', 'vit')
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
        # Handle CIFAR100-specific models from HuggingFace
        elif 'cifar100' in name.lower():
            teacher = _load_cifar100_teacher_from_hf(name, device)
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


def _load_cifar100_teacher_from_hf(name: str, device: str) -> nn.Module:
    """
    Load CIFAR100 teacher model from HuggingFace.
    CIFAR100 models have modified architecture for 32x32 images.
    
    Args:
        name: Model name (e.g., 'resnet50_cifar100', 'densenet121_cifar100')
        device: Device to load model on
        
    Returns:
        nn.Module: Loaded CIFAR100 teacher model
    """
    # Map model names to HuggingFace URLs
    # These are pretrained on CIFAR100 from https://huggingface.co/edadaltocg
    hf_urls = {
        'resnet50_cifar100': 'https://huggingface.co/edadaltocg/resnet50_cifar100/resolve/main/pytorch_model.bin',
        'resnet18_cifar100': 'https://huggingface.co/edadaltocg/resnet18_cifar100/resolve/main/pytorch_model.bin',
        'resnet34_cifar100': 'https://huggingface.co/edadaltocg/resnet34_cifar100/resolve/main/pytorch_model.bin',
        'densenet121_cifar100': 'https://huggingface.co/edadaltocg/densenet121_cifar100/resolve/main/pytorch_model.bin',
    }
    
    if name not in hf_urls:
        raise ValueError(f"Unknown CIFAR100 model: {name}. Available: {list(hf_urls.keys())}")
    
    # Extract base model name (remove _cifar100 suffix)
    base_name = name.replace('_cifar100', '')
    
    # DenseNet for CIFAR100 has completely different architecture
    if 'densenet' in name.lower():
        # Create DenseNet with CIFAR100-specific architecture
        # Based on https://github.com/kuangliu/pytorch-cifar
        from torchvision.models import DenseNet
        
        # DenseNet-121 for CIFAR100: growth_rate=12, same blocks as DenseNet-121
        model = DenseNet(
            growth_rate=12,
            block_config=(6, 12, 24, 16),  # Standard DenseNet-121 config
            num_init_features=24,          # CIFAR100: 24 instead of 64
            bn_size=4,
            drop_rate=0,
            num_classes=100
        )
        
        # Modify first conv for CIFAR100 (3x3 instead of 7x7)
        model.features.conv0 = nn.Conv2d(
            3, 24, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Add norm0 layer (expected by checkpoint)
        model.features.norm0 = nn.BatchNorm2d(24)
        # Remove pooling (CIFAR images are small)
        model.features.pool0 = nn.Identity()
    else:
        # ResNet models - create base and modify for CIFAR100
        model = timm.create_model(base_name, pretrained=False, num_classes=100)
        
        # Modify first conv layer for CIFAR100 (32x32 images)
        # CIFAR models use 3x3 conv instead of 7x7
        if hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
            out_channels = model.conv1.out_channels
            model.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool for CIFAR (images are too small)
        if hasattr(model, 'maxpool'):
            model.maxpool = nn.Identity()
    
    # Download and load CIFAR100-specific weights from HuggingFace
    state_dict = torch.hub.load_state_dict_from_url(
        hf_urls[name],
        map_location=device,
        file_name=f"{name}.pth",
    )
    
    # Load weights (use strict=False for debugging if needed)
    model.load_state_dict(state_dict, strict=False)
    
    return model


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


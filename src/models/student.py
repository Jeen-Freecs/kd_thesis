"""Student model creation and configuration"""

import torch.nn as nn
import timm


def create_student_model(
    model_name: str = 'mobilenetv2_100',
    num_classes: int = 100,
    pretrained: bool = False
) -> nn.Module:
    """
    Create a student model for knowledge distillation.
    
    Args:
        model_name: Name of the model architecture (from timm)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        nn.Module: Student model
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    
    # Reset classifier for the correct number of classes
    model.reset_classifier(num_classes=num_classes)
    
    return model


class StudentModelWrapper(nn.Module):
    """
    Wrapper for student models with additional utilities.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 100
    ):
        """
        Initialize student model wrapper.
        
        Args:
            base_model: Base student architecture
            num_classes: Number of output classes
        """
        super().__init__()
        self.model = base_model
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


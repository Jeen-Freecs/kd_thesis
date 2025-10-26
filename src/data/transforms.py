"""Dataset transformations for dual teacher-student setup"""

from typing import List, Optional
import torch
from torchvision import transforms


class DualTransformDataset(torch.utils.data.Dataset):
    """
    Wrapper for dataset that applies different transforms to teacher and student models.
    
    This is essential for knowledge distillation where teachers and students may require
    different preprocessing and augmentation strategies.
    """
    
    def __init__(
        self,
        dataset,
        teacher_transforms: Optional[List[transforms.Compose]] = None,
        student_transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dual transform dataset.
        
        Args:
            dataset: Base PyTorch dataset
            teacher_transforms: List of transform pipelines for each teacher model
            student_transform: Transform pipeline for student model
        """
        self.dataset = dataset
        self.teacher_transforms = teacher_transforms if teacher_transforms is not None else []
        self.student_transform = student_transform

    def __getitem__(self, idx):
        """
        Get item with all transformations applied.
        
        Returns:
            dict: Dictionary containing transformed inputs for teachers, student, and label
        """
        img, label = self.dataset[idx]
        
        # Convert to PIL if needed
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        output = {}
        
        # Apply each teacher's transform
        for i, teacher_transform in enumerate(self.teacher_transforms):
            output[f'teacher_input_{i}'] = teacher_transform(img)

        # Apply student transform
        if self.student_transform is not None:
            output['student_input'] = self.student_transform(img)

        # Add label
        output['label'] = label

        return output

    def __len__(self):
        return len(self.dataset)


def get_transform(
    model_type: str, 
    pre_trained: bool = False, 
    is_train: bool = True
) -> transforms.Compose:
    """
    Get appropriate transforms based on model type.
    
    Args:
        model_type: Type of model ('mobilenet', 'resnet', 'densenet', 'vit')
        pre_trained: Whether to use ImageNet normalization
        is_train: Whether this is for training (includes augmentation)
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    from PIL import Image
    
    if pre_trained:
        normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        normalizer = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])

    if model_type == 'mobilenet':
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    elif model_type == 'resnet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2673, 0.2564, 0.2762]
            )
        ])
    elif model_type == 'densenet':
        return transforms.Compose([
            transforms.Resize((36, 36), interpolation=Image.BILINEAR),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    elif model_type == 'vit':
        return transforms.Compose([
            transforms.Resize((248, 248), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    else:
        raise ValueError(f"Invalid model type: {model_type}")


"""PyTorch Lightning DataModule for CIFAR-100"""

from typing import Optional, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .transforms import DualTransformDataset, get_transform


class CIFAR100DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CIFAR-100 dataset.
    
    Handles data downloading, preparation, and provides dataloaders for
    training, validation, and testing with support for multiple teacher models.
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        teacher_models: Optional[List[str]] = None,
        student_model: str = 'mobilenet',
        pre_trained: bool = False,
        val_size: int = 5000,
        seed: int = 42
    ):
        """
        Initialize CIFAR-100 DataModule.
        
        Args:
            data_dir: Directory to store/load dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            teacher_models: List of teacher model types for transforms
            student_model: Student model type for transforms
            pre_trained: Whether to use ImageNet normalization
            val_size: Number of samples for validation set
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.teacher_models = teacher_models if teacher_models is not None else []
        self.student_model = student_model
        self.val_size = val_size
        self.seed = seed
        self.pre_trained = pre_trained

        # Student transforms
        self.student_transform = get_transform(
            self.student_model, self.pre_trained, is_train=True
        )
        self.student_transform_val = get_transform(
            self.student_model, self.pre_trained, is_train=False
        )

        # Teacher transforms
        self.teacher_transforms = []
        self.teacher_transforms_val = []
        if self.teacher_models:
            for model_type in self.teacher_models:
                teacher_transform = get_transform(
                    model_type, self.pre_trained, is_train=False
                )
                teacher_transform_val = get_transform(
                    model_type, self.pre_trained, is_train=False
                )
                self.teacher_transforms.append(teacher_transform)
                self.teacher_transforms_val.append(teacher_transform_val)

    def prepare_data(self):
        """Download CIFAR-100 dataset if not already present."""
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, or testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage in ('fit', 'validate', None):
            # Load full training dataset
            base_train_dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                transform=None
            )

            # Stratified split for train/val
            targets = base_train_dataset.targets
            strat_split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.val_size,
                random_state=self.seed
            )

            for train_idx, val_idx in strat_split.split(
                np.arange(len(targets)), targets
            ):
                self.train_dataset = DualTransformDataset(
                    Subset(base_train_dataset, train_idx),
                    teacher_transforms=self.teacher_transforms,
                    student_transform=self.student_transform
                )
                self.val_dataset = DualTransformDataset(
                    Subset(base_train_dataset, val_idx),
                    teacher_transforms=self.teacher_transforms_val,
                    student_transform=self.student_transform_val
                )

        if stage in ('test', None):
            self.test_dataset = DualTransformDataset(
                datasets.CIFAR100(
                    root=self.data_dir,
                    train=False,
                    transform=None
                ),
                teacher_transforms=self.teacher_transforms_val,
                student_transform=self.student_transform_val
            )

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


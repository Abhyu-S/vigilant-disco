"""
CIFAR-100 DataModule for PyTorch Lightning
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
import pytorch_lightning as pl
from typing import Optional


class CIFAR100DataModule(pl.LightningDataModule):
    """
    DataModule for CIFAR-100 dataset
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        image_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        
        # Transform for CIFAR-100 (resize to ImageNet size)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def prepare_data(self):
        """Download dataset if not already present"""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Set up train and validation datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR100(
                self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            self.val_dataset = CIFAR100(
                self.data_dir,
                train=False,
                transform=self.val_transform,
            )
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.val_dataset,  # Using validation set as test for now
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


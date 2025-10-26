"""
PyTorch Lightning module for Quantized ViT-L
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Optional
from torchmetrics.classification import Accuracy, F1Score
from src.models.quantized_vit import QuantizedVisionTransformer
from src.utils.metrics import compute_accuracy, compute_f1, compute_confusion_matrix
import wandb
import numpy as np


class LitQuantizedViT(pl.LightningModule):
    """
    Lightning module for Quantized Vision Transformer
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        num_classes: int = 100,
        nbits_w: int = 4,
        nbits_a: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        warmup_epochs: int = 10,
        max_epochs: int = 300,
        batch_size: int = 128,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = QuantizedVisionTransformer(
            model_name=model_name,
            num_classes=num_classes,
            nbits_w=nbits_w,
            nbits_a=nbits_a,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        
    def forward(self, pixel_values, labels=None):
        """Forward pass"""
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        # Logits
        logits = outputs.logits
        
        # Compute metrics
        acc = self.train_acc(logits, labels)
        f1 = self.train_f1(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        # Logits
        logits = outputs.logits
        
        # Compute metrics
        acc = self.val_acc(logits, labels)
        f1 = self.val_f1(logits, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'labels': labels, 'logits': logits}
    
    def on_validation_epoch_end(self):
        """Compute confusion matrix at end of validation epoch"""
        # Get all predictions and labels from validation
        # This is a simplified version - in practice you'd accumulate across all batches
        pass
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Choose optimizer
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Warmup: linear increase
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


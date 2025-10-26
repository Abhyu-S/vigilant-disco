"""
Training script for Quantized ViT-L on CIFAR-100
Uses PyTorch Lightning, Hydra, and Wandb
"""
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models import LitQuantizedViT
from src.data import CIFAR100DataModule
import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function
    """
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        name=cfg.logger.name,
        project=cfg.logger.project,
        log_model=cfg.logger.log_model,
    )
    
    # Initialize data module
    data_module = CIFAR100DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )
    
    # Initialize model
    model = LitQuantizedViT(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        nbits_w=cfg.model.nbits_w,
        nbits_a=cfg.model.nbits_a,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        optimizer=cfg.training.optimizer,
    )
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{wandb_logger.experiment.name}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        filename="epoch_{epoch:02d}_val_acc_{val_acc:.4f}",
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=50,
        enable_progress_bar=cfg.callbacks.enable_progress_bar,
        enable_model_summary=cfg.callbacks.enable_model_summary,
        profiler=None,
    )
    
    # Train
    trainer.fit(model, datamodule=data_module)
    
    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()


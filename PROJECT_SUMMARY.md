# Project Summary: Quantized ViT-L on CIFAR-100

## Overview
This project implements a quantized Vision Transformer (ViT-L) for CIFAR-100 classification using PyTorch Lightning, Hydra configuration, and Weights & Biases logging.

## Classifier Head Configuration

### Answer to Your Question: 100 vs 1000 Nodes

**Current Configuration: The classifier head has 100 nodes (for CIFAR-100)**

This is configured in:
- `configs/config.yaml`: `num_classes: 100`
- `src/models/quantized_vit.py`: The classifier is initialized with 100 output nodes
- The HuggingFace ViT-L model (1000 classes) is adapted by replacing the final classifier layer

### If You Want to Use ImageNet (1000 classes):
Change the config:
```yaml
model:
  num_classes: 1000  # Instead of 100
```

## File Structure

```
qvit_lightning/
├── configs/
│   ├── config.yaml              # Main Hydra configuration
│   └── train/
│       ├── default.yaml         # Training-specific configs
│       └── __init__.py
├── src/
│   ├── models/
│   │   ├── quantized_vit.py     # Quantized ViT architecture
│   │   ├── lit_vit.py           # Lightning module wrapper
│   │   └── __init__.py
│   ├── data/
│   │   ├── cifar100_datamodule.py  # CIFAR-100 data loading
│   │   └── __init__.py
│   ├── utils/
│   │   ├── metrics.py           # Accuracy, F1, Confusion Matrix
│   │   └── __init__.py
│   ├── Quant.py                 # Quantization modules (Q-ViT)
│   ├── _quan_base.py            # Base quantized layers
│   └── __init__.py
├── train.py                     # Main training script
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── README.md                    # Documentation
├── example_usage.py            # Usage examples
└── .gitignore                   # Git ignore rules
```

## Key Components

### 1. Model Architecture (`src/models/quantized_vit.py`)
- Imports `vit-large-patch16-224` from HuggingFace
- Replaces the classifier head from 1000 → 100 classes for CIFAR-100
- Applies 4-bit quantization to:
  - Convolutional layers (patch embedding)
  - Attention layers (Q, K, V, and output projections)
  - MLP layers (intermediate and output projections)
  - Final classifier

### 2. Lightning Module (`src/models/lit_vit.py`)
- Wraps the quantized model in PyTorch Lightning
- Implements training/validation steps
- Tracks metrics: Loss, Accuracy, F1 Score
- Configured with learning rate scheduling and warmup

### 3. Data Module (`src/data/cifar100_datamodule.py`)
- Loads CIFAR-100 dataset
- Resizes images to 224x224 (ViT input size)
- Applies data augmentation for training
- Returns batches in ImageNet format (mean/std normalization)

### 4. Configuration (`configs/config.yaml`)
- Model: ViT-Large with 100 classes, 4-bit quantization
- Training: LR=3e-4, Batch=128, 300 epochs
- Hardware: GPU with mixed precision (16-bit)
- Logging: WandB integration

### 5. Training Script (`train.py`)
- Uses Hydra for configuration management
- Integrates WandB for experiment tracking
- Sets up model checkpoints
- Logs learning rate over time

## Quantization Details

### Bits:
- **Weights (nbits_w)**: 4 bits
- **Activations (nbits_a)**: 4 bits

### Methods:
- **IRM (Intra-Rank Modulo)**: Quantization scheme from Q-ViT paper
- **DGD (Dynamic Gradient Descent)**: Training approach with gradient scaling

### Layers Quantized:
1. Patch embedding (Conv2d)
2. Attention layers (Linear)
3. MLP layers (Linear)
4. Classifier (Linear)

## Usage

### Basic Training:
```bash
python train.py
```

### Custom Configuration:
```bash
python train.py training.batch_size=256 model.nbits_w=8
```

### With WandB:
```bash
wandb login
python train.py logger.project=my-experiment
```

### Multi-GPU:
```bash
python train.py trainer.devices=-1 trainer.accelerator=gpu
```

## Hardware Requirements
- GPU: A100 (40GB VRAM)
- CPUs: 30 available
- RAM: 32GB+ recommended

## Metrics Logged
- **Training**: loss, accuracy, F1 score
- **Validation**: loss, accuracy, F1 score
- **Confusion Matrix**: logged to WandB
- **Learning Rate**: tracked per epoch

## Next Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Login to WandB: `wandb login`
3. Start training: `python train.py`
4. Monitor on WandB dashboard

## Important Notes
- The model downloads ViT-L weights from HuggingFace on first run
- CIFAR-100 will be downloaded automatically if not present in `./data/`
- Checkpoints are saved in `./checkpoints/`
- Logs go to WandB and local `lightning_logs/` directory


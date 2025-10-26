# Q-ViT Lightning: Quantized Vision Transformer on CIFAR-100

This repository implements a quantized Vision Transformer (ViT-L) for CIFAR-100 classification using PyTorch Lightning, Hydra, and Weights & Biases.

## Features

- **Model**: ViT-L (Large) from HuggingFace Transformers
- **Quantization**: IRM (Intra-Rank Modulo) and DGD (Dynamic Gradient Descent) quantization with 4-bit weights and activations
- **Dataset**: CIFAR-100 (100 classes)
- **Framework**: PyTorch Lightning
- **Configuration**: Hydra
- **Logging**: Weights & Biases (WandB)
- **Metrics**: Accuracy, F1 Score, and Confusion Matrix

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd qvit_lightning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train the quantized ViT-L model on CIFAR-100:

```bash
python train.py
```

### With Custom Configuration

Override configuration parameters:

```bash
python train.py training.batch_size=256 model.nbits_w=8 training.max_epochs=200
```

### With WandB

Make sure to login to WandB:
```bash
wandb login
```

Then run training with your WandB project:
```bash
python train.py logger.project=my-project training.max_epochs=300
```

### Multi-GPU Training

For multi-GPU training on available GPUs:

```bash
python train.py trainer.devices=-1 trainer.accelerator=gpu
```

## Configuration

Modify `configs/config.yaml` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data loading parameters
- WandB logging settings

## Project Structure

```
qvit_lightning/
├── configs/
│   ├── config.yaml              # Main configuration
│   └── train/
│       └── default.yaml         # Training-specific configs
├── src/
│   ├── models/
│   │   ├── quantized_vit.py     # Quantized ViT model
│   │   └── lit_vit.py            # Lightning module
│   ├── data/
│   │   └── cifar100_datamodule.py  # Data module
│   ├── utils/
│   │   └── metrics.py            # Metric utilities
│   └── Quant.py                  # Quantization modules
├── train.py                      # Training script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Model Architecture

- **Base Model**: Google ViT-Large-Patch16-224 from HuggingFace
- **Quantization**: 
  - 4-bit weights (nbits_w=4)
  - 4-bit activations (nbits_a=4)
  - IRM and DGD quantization methods
- **Classifier Head**: 100 output nodes for CIFAR-100

## Metrics

The model tracks the following metrics:
- **Training**: Loss, Accuracy, F1 Score
- **Validation**: Loss, Accuracy, F1 Score
- **Confusion Matrix**: Logged to WandB at validation

## Hardware Requirements

- GPU: A100 (40GB VRAM)
- CPUs: 30 CPUs
- RAM: Recommended 32GB+

## Citation

Based on the Q-ViT paper:
```
@inproceedings{li2022qvit,
  title={Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer},
  author={Li, Yanjing and Gong, Ruihao and Tan, Xitian and Yang, Yan and Hu, Peng and Zhang, Qi and Yu, Wei and Wang, Fengwei and Sun, Si},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License

This project is licensed under the MIT License.


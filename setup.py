"""
Setup script for qvit_lightning package
"""
from setuptools import setup, find_packages

setup(
    name="qvit_lightning",
    version="0.1.0",
    description="Quantized Vision Transformer using PyTorch Lightning",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pytorch-lightning>=2.0.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "torchmetrics>=1.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "timm>=0.9.0",
    ],
    python_requires=">=3.8",
)


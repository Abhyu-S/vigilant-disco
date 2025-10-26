"""
Example usage of the quantized ViT-L model
"""
import torch
from src.models import LitQuantizedViT
from PIL import Image
import requests

def example_classification():
    """Example of using the quantized model for inference"""
    
    # Initialize model (you would load from checkpoint after training)
    model = LitQuantizedViT(
        model_name="google/vit-large-patch16-224",
        num_classes=100,
        nbits_w=4,
        nbits_a=4,
    )
    
    model.eval()
    
    # Example: Download and process an image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Note: For actual inference, you'll need proper preprocessing
    # This is just an example
    
    print("Model initialized successfully!")
    print(f"Number of classes: {model.hparams.num_classes}")
    print(f"Quantization bits (weights): {model.hparams.nbits_w}")
    print(f"Quantization bits (activations): {model.hparams.nbits_a}")


if __name__ == "__main__":
    example_classification()


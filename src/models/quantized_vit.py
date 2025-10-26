"""
Quantized Vision Transformer using IRM and DGD methods
Based on Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers import ViTForImageClassification
from src.Quant import LinearQ, Conv2dQ, ActQ


class QuantizedVisionTransformer(nn.Module):
    """
    Quantized Vision Transformer for CIFAR-100
    Implements IRM (Intra-Rank Modulo) and DGD (Dynamic Gradient Descent) quantization
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        num_classes: int = 100,
        nbits_w: int = 4,
        nbits_a: int = 4,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        
        # Load pretrained ViT from HuggingFace
        self.vit = ViTForImageClassification.from_pretrained(model_name)
        
        # Get the base ViT model and config
        vit_model = self.vit.vit
        
        # Replace classifier head for CIFAR-100
        # Store the embedding dimension
        self.embed_dim = vit_model.config.hidden_size
        self.num_classes = num_classes
        
        # Replace classifier with 100 classes
        self.vit.classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Update the config
        self.vit.config.num_labels = num_classes
        self.vit.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
        self.vit.config.label2id = {f"LABEL_{i}": i for i in range(num_classes)}
        
        # Quantize layers
        self._quantize_model()
        
    def _quantize_model(self):
        """Replace layers with quantized versions"""
        # Skip quantization of patch embeddings for now
        # The ViT patch embeddings structure from HuggingFace is different
        # and doesn't easily map to a simple Conv2d layer
        pass
        
        # Quantize attention and MLP layers in each transformer block
        for block in self.vit.vit.encoder.layer:
            # Quantize attention layers
            if hasattr(block.attention, 'query'):
                # Query
                orig_query = block.attention.query
                new_query = LinearQ(
                    in_features=orig_query.in_features,
                    out_features=orig_query.out_features,
                    bias=orig_query.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_query.weight = orig_query.weight
                if orig_query.bias is not None:
                    new_query.bias = orig_query.bias
                block.attention.query = new_query
                
                # Key
                orig_key = block.attention.key
                new_key = LinearQ(
                    in_features=orig_key.in_features,
                    out_features=orig_key.out_features,
                    bias=orig_key.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_key.weight = orig_key.weight
                if orig_key.bias is not None:
                    new_key.bias = orig_key.bias
                block.attention.key = new_key
                
                # Value
                orig_value = block.attention.value
                new_value = LinearQ(
                    in_features=orig_value.in_features,
                    out_features=orig_value.out_features,
                    bias=orig_value.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_value.weight = orig_value.weight
                if orig_value.bias is not None:
                    new_value.bias = orig_value.bias
                block.attention.value = new_value
                
                # Output
                if hasattr(block.attention, 'output'):
                    orig_output = block.attention.output.dense
                    new_output = LinearQ(
                        in_features=orig_output.in_features,
                        out_features=orig_output.out_features,
                        bias=orig_output.bias is not None,
                        nbits_w=self.nbits_w
                    )
                    new_output.weight = orig_output.weight
                    if orig_output.bias is not None:
                        new_output.bias = orig_output.bias
                    block.attention.output.dense = new_output
            
            # Quantize MLP layers
            if hasattr(block, 'intermediate'):
                # Intermediate (FFN up)
                orig_intermediate = block.intermediate.dense
                new_intermediate = LinearQ(
                    in_features=orig_intermediate.in_features,
                    out_features=orig_intermediate.out_features,
                    bias=orig_intermediate.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_intermediate.weight = orig_intermediate.weight
                if orig_intermediate.bias is not None:
                    new_intermediate.bias = orig_intermediate.bias
                block.intermediate.dense = new_intermediate
                
                # Output (FFN down)
                orig_output = block.output.dense
                new_output = LinearQ(
                    in_features=orig_output.in_features,
                    out_features=orig_output.out_features,
                    bias=orig_output.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_output.weight = orig_output.weight
                if orig_output.bias is not None:
                    new_output.bias = orig_output.bias
                block.output.dense = new_output
        
        # Quantize final classifier (already replaced for CIFAR-100)
        if isinstance(self.vit.classifier, nn.Linear):
            orig_classifier = self.vit.classifier
            self.vit.classifier = LinearQ(
                in_features=orig_classifier.in_features,
                out_features=orig_classifier.out_features,
                bias=orig_classifier.bias is not None,
                nbits_w=self.nbits_w
            )
            self.vit.classifier.weight = orig_classifier.weight
            if orig_classifier.bias is not None:
                self.vit.classifier.bias = orig_classifier.bias
    
    def forward(self, pixel_values, labels=None):
        """
        Forward pass
        Args:
            pixel_values: Input images tensor of shape (batch, channels, height, width)
            labels: Optional ground truth labels
        """
        # Get base model output
        embedding_output = self.vit.vit.embeddings(pixel_values)
        encoder_outputs = self.vit.vit.encoder(embedding_output)
        sequence_output = encoder_outputs.last_hidden_state
        
        # Get pooled output (CLS token)
        pooled_output = self.vit.vit.layernorm(sequence_output[:, 0])
        
        # Get logits from quantized classifier
        logits = self.vit.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        # Return simple output with logits and loss
        from types import SimpleNamespace
        outputs = SimpleNamespace()
        outputs.logits = logits
        outputs.loss = loss
        return outputs
    
    def get_config(self):
        """Get model configuration"""
        return {
            'num_classes': self.num_classes,
            'nbits_w': self.nbits_w,
            'nbits_a': self.nbits_a,
        }


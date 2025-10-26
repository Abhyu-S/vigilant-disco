# Answer: Classifier Head Nodes (100 vs 1000)

## Direct Answer

**The classifier head in this codebase is configured for 100 nodes (for CIFAR-100).**

## Details

### Where It's Configured

1. **`configs/config.yaml`** (Line 8):
   ```yaml
   model:
     num_classes: 100  # ← 100 nodes for CIFAR-100
   ```

2. **`src/models/quantized_vit.py`** (Lines 67-69):
   ```python
   # Replace classifier head for CIFAR-100
   self.vit.classifier = nn.Linear(vit_model.config.hidden_size, num_classes)
   ```
   Where `num_classes` = 100

3. **`src/models/quantized_vit.py`** (Lines 217-227):
   ```python
   # Quantize final classifier
   orig_classifier = self.vit.classifier
   self.vit.classifier = LinearQ(
       in_features=orig_classifier.in_features,
       out_features=orig_classifier.out_features,  # = 100
       bias=orig_classifier.bias is not None,
       nbits_w=self.nbits_w
   )
   ```

### Why 100?

- **CIFAR-100 dataset** has 100 classes
- The original ViT-L from HuggingFace has **1000 nodes** (for ImageNet)
- We **replace** the classifier layer to adapt it to CIFAR-100

### If You Want 1000 Classes (ImageNet):

Simply change the config file:
```yaml
# configs/config.yaml
model:
  num_classes: 1000  # ← Change this to 1000
```

Then the classifier will output 1000 logits for ImageNet classes.

## Architecture Details

```
ViT-L Architecture:
├── Patch Embedding (4-bit quantized)
├── 24x Transformer Blocks
│   ├── Attention (Q,K,V,O) - 4-bit quantized
│   └── MLP (Intermediate + Output) - 4-bit quantized
└── Classifier Head
    └── Linear(1024 → 100) ← You are here! 100 nodes for CIFAR-100
```

Where:
- Input: 1024-dim embedding from ViT-L
- Output: 100 logits (for CIFAR-100 classes)
- Weights: 4-bit quantized using IRM method
- Activations: 4-bit quantized using DGD method

## Model Summary

To see the exact classifier configuration:
```bash
python -c "from src.models import LitQuantizedViT; model = LitQuantizedViT(); print(model.model.vit.classifier)"
```

This will show:
```
LinearQ(1024 -> 100)  # 100 output nodes
```

## Conclusion

✅ **Classifier has 100 nodes** - correctly configured for CIFAR-100  
📊 If you need ImageNet (1000 classes), change `num_classes` in config  
🔧 The ViT-L backbone remains the same, only the classifier head is changed


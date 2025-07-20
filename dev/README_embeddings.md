# EnCodec Embeddings for Emotion Detection

This document explains how to extract and use embeddings from the EnCodec model for emotion detection tasks.

## Overview

EnCodec is a high-quality neural audio codec that can be used to extract rich audio representations (embeddings) that are useful for downstream tasks like emotion detection. The embeddings capture both local and global audio features in a compressed, meaningful representation.

## Types of Embeddings

### 1. Encoder Embeddings (Recommended)
These are the raw embeddings from the encoder before quantization. They contain the most information and are best for downstream tasks.

```python
def get_encoder_embeddings(model, audio):
    """Extract embeddings from EnCodec encoder."""
    with torch.no_grad():
        # Normalize if needed
        if model.normalize:
            mono = audio.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            audio = audio / scale
        
        # Get embeddings from encoder
        embeddings = model.encoder(audio)
        return embeddings
```

### 2. Quantized Embeddings
These are embeddings after quantization (from the decoder). They are more compressed but may lose some information.

```python
def get_quantized_embeddings(model, audio):
    """Extract embeddings after quantization."""
    with torch.no_grad():
        encoded_frames = model.encode(audio)
        quantized_embeddings = []
        for frame in encoded_frames:
            codes, scale = frame
            codes = codes.transpose(0, 1)
            emb = model.quantizer.decode(codes)
            quantized_embeddings.append(emb)
        
        if len(quantized_embeddings) > 1:
            embeddings = torch.cat(quantized_embeddings, dim=-1)
        else:
            embeddings = quantized_embeddings[0]
        
        return embeddings
```

## Usage Examples

### Basic Embeddings Extraction

```python
from encodec import EncodecModel
import torch

# Load pre-trained EnCodec model
model = EncodecModel.encodec_model_24khz(pretrained=True)
model.eval()

# Set bandwidth (optional)
model.set_target_bandwidth(6.0)  # 6 kbps

# Your audio tensor [B, C, T]
audio = torch.randn(1, 1, 24000)  # 1 second of audio

# Extract embeddings
embeddings = model.encoder(audio)
print(f"Embeddings shape: {embeddings.shape}")  # [B, D, T']
```

### Using the Integrated Model

```python
from model import EmotionDetectionModel

# Create emotion detection model with EnCodec embeddings
model = EmotionDetectionModel(num_classes=8)

# Get embeddings for analysis
embeddings = model.get_embeddings(audio)

# Or use for classification
logits = model(audio)
```

### Global Embeddings (for Classification)

For emotion classification, you typically want a fixed-size representation. You can aggregate temporal embeddings:

```python
def get_global_embeddings(embeddings, method='mean'):
    """Convert temporal embeddings to global embeddings."""
    if method == 'mean':
        return embeddings.mean(dim=-1)
    elif method == 'max':
        return embeddings.max(dim=-1)[0]
    elif method == 'attention':
        # Simple attention mechanism
        attention_weights = torch.softmax(embeddings.mean(dim=1, keepdim=True), dim=-1)
        return (embeddings * attention_weights).sum(dim=-1)

# Usage
global_emb = get_global_embeddings(embeddings, 'mean')
print(f"Global embeddings shape: {global_emb.shape}")  # [B, D]
```

## Model Architecture

The `EmotionDetectionModel` uses EnCodec embeddings as follows:

1. **Feature Extraction**: EnCodec encoder extracts rich audio features
2. **Global Pooling**: Temporal embeddings are aggregated to fixed-size vectors
3. **Classification**: Multi-layer perceptron classifies emotions

```python
class EmotionDetectionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        # Load pre-trained EnCodec model
        self.encodec = EncodecModel.encodec_model_24khz(pretrained=True)
        self.encodec.eval()
        
        # Freeze EnCodec parameters
        for param in self.encodec.parameters():
            param.requires_grad = False
            
        # Emotion classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.encodec.encoder.dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
```

## Testing and Validation

### Run Tests
```bash
cd dev
python test_embeddings.py
```

### Extract Embeddings for Analysis
```bash
python extract_embeddings.py
```

This will:
- Analyze embeddings from your dataset
- Save embeddings for visualization (t-SNE, UMAP, etc.)
- Print statistics about the embeddings

## Embeddings Analysis

The saved embeddings can be used for:

1. **Visualization**: t-SNE, UMAP, PCA
2. **Clustering**: K-means, DBSCAN
3. **Similarity Analysis**: Cosine similarity, Euclidean distance
4. **Transfer Learning**: Use embeddings for other audio tasks

### Example Analysis Script

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load saved embeddings
embeddings = np.load("embeddings_analysis/embeddings.npy")
labels = np.load("embeddings_analysis/labels.npy")

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of EnCodec embeddings')
plt.show()
```

## Performance Considerations

### Memory Usage
- EnCodec embeddings are typically 128-dimensional
- For large datasets, consider batch processing
- Use `torch.no_grad()` for inference

### Speed
- EnCodec is optimized for real-time processing
- GPU acceleration significantly improves speed
- Consider caching embeddings for repeated analysis

### Quality vs. Speed Trade-offs
- Higher bandwidth → better quality, slower processing
- Lower bandwidth → faster processing, lower quality
- Recommended: 6-12 kbps for emotion detection

## Troubleshooting

### Common Issues

1. **Import Error**: Install encodec with `pip install encodec`
2. **CUDA Memory**: Use CPU or reduce batch size
3. **Audio Format**: Ensure audio is [B, C, T] format
4. **Sample Rate**: EnCodec expects 24kHz or 48kHz

### Debug Tips

```python
# Check model properties
print(f"Sample rate: {model.sample_rate}")
print(f"Channels: {model.channels}")
print(f"Embedding dimension: {model.encoder.dimension}")

# Check audio format
print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
```

## Advanced Usage

### Fine-tuning EnCodec
You can unfreeze and fine-tune the EnCodec model:

```python
# Unfreeze encoder layers
for param in model.encodec.encoder.parameters():
    param.requires_grad = True

# Use smaller learning rate for pre-trained parts
optimizer = torch.optim.Adam([
    {'params': model.encodec.encoder.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### Multi-scale Embeddings
Extract embeddings at different temporal scales:

```python
def get_multi_scale_embeddings(model, audio):
    """Extract embeddings at different temporal resolutions."""
    embeddings = model.encoder(audio)
    
    # Different pooling strategies
    global_mean = embeddings.mean(dim=-1)
    global_max = embeddings.max(dim=-1)[0]
    global_std = embeddings.std(dim=-1)
    
    # Concatenate different representations
    multi_scale = torch.cat([global_mean, global_max, global_std], dim=1)
    return multi_scale
```

## References

- [EnCodec Paper](https://arxiv.org/abs/2210.13438)
- [EnCodec GitHub](https://github.com/facebookresearch/encodec)
- [Audio Embeddings Guide](https://huggingface.co/docs/transformers/model_doc/encodec) 
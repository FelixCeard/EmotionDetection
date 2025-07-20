"""
Utility script to extract embeddings from EnCodec models.
This demonstrates different ways to get embeddings from the EnCodec model.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from encodec import EncodecModel
from dataset import EmotionDataset
from dotenv import load_dotenv
import os


def extract_encoder_embeddings(model: EncodecModel, audio: torch.Tensor) -> torch.Tensor:
    """
    Extract embeddings from the EnCodec encoder (before quantization).
    
    Args:
        model: EnCodec model
        audio: Audio tensor of shape [B, C, T]
    
    Returns:
        Embeddings tensor of shape [B, D, T'] where D is the embedding dimension
    """
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


def extract_quantized_embeddings(model: EncodecModel, audio: torch.Tensor) -> torch.Tensor:
    """
    Extract embeddings after quantization (from decoder).
    
    Args:
        model: EnCodec model
        audio: Audio tensor of shape [B, C, T]
    
    Returns:
        Quantized embeddings tensor of shape [B, D, T']
    """
    with torch.no_grad():
        # Encode to get codes
        encoded_frames = model.encode(audio)
        
        # Decode codes back to embeddings
        quantized_embeddings = []
        for frame in encoded_frames:
            codes, scale = frame
            codes = codes.transpose(0, 1)
            emb = model.quantizer.decode(codes)
            quantized_embeddings.append(emb)
        
        # Concatenate frames if multiple
        if len(quantized_embeddings) > 1:
            embeddings = torch.cat(quantized_embeddings, dim=-1)
        else:
            embeddings = quantized_embeddings[0]
    
    return embeddings


def get_global_embeddings(embeddings: torch.Tensor, method: str = 'mean') -> torch.Tensor:
    """
    Convert temporal embeddings to global embeddings.
    
    Args:
        embeddings: Temporal embeddings of shape [B, D, T]
        method: Aggregation method ('mean', 'max', 'attention')
    
    Returns:
        Global embeddings of shape [B, D]
    """
    if method == 'mean':
        return embeddings.mean(dim=-1)
    elif method == 'max':
        return embeddings.max(dim=-1)[0]
    elif method == 'attention':
        # Simple attention mechanism
        attention_weights = torch.softmax(embeddings.mean(dim=1, keepdim=True), dim=-1)
        return (embeddings * attention_weights).sum(dim=-1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def analyze_embeddings():
    """Analyze embeddings from the emotion dataset."""
    
    # Load environment
    load_dotenv()
    path_datasets = Path(os.getenv("PATH_DATASETS"))
    
    # Load dataset
    dataset = EmotionDataset(root_dir=path_datasets, resample_rate=16_000)
    
    # Load EnCodec model
    model = EncodecModel.encodec_model_24khz(pretrained=True)
    model.eval()
    
    # Set bandwidth (optional)
    model.set_target_bandwidth(6.0)  # 6 kbps
    
    print(f"EnCodec model loaded:")
    print(f"- Sample rate: {model.sample_rate}")
    print(f"- Channels: {model.channels}")
    print(f"- Embedding dimension: {model.encoder.dimension}")
    print(f"- Frame rate: {model.frame_rate}")
    
    # Extract embeddings from a few samples
    emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for i in range(min(5, len(dataset))):
        audio, label, dataset_source = dataset[i]
        
        # Ensure audio is in correct format [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        print(f"\nSample {i+1}:")
        print(f"- Audio shape: {audio.shape}")
        print(f"- Emotion: {emotion_labels[label-1] if label <= len(emotion_labels) else 'unknown'}")
        print(f"- Dataset source: {dataset_source}")
        
        # Extract different types of embeddings
        encoder_emb = extract_encoder_embeddings(model, audio)
        quantized_emb = extract_quantized_embeddings(model, audio)
        
        print(f"- Encoder embeddings shape: {encoder_emb.shape}")
        print(f"- Quantized embeddings shape: {quantized_emb.shape}")
        
        # Get global embeddings
        global_encoder_emb = get_global_embeddings(encoder_emb, 'mean')
        global_quantized_emb = get_global_embeddings(quantized_emb, 'mean')
        
        print(f"- Global encoder embeddings shape: {global_encoder_emb.shape}")
        print(f"- Global quantized embeddings shape: {global_quantized_emb.shape}")
        
        # Calculate some statistics
        print(f"- Encoder embedding stats: mean={encoder_emb.mean():.4f}, std={encoder_emb.std():.4f}")
        print(f"- Quantized embedding stats: mean={quantized_emb.mean():.4f}, std={quantized_emb.std():.4f}")


def save_embeddings_for_analysis():
    """Save embeddings for further analysis (e.g., t-SNE visualization)."""
    
    # Load environment
    load_dotenv()
    path_datasets = Path(os.getenv("PATH_DATASETS"))
    
    # Load dataset
    dataset = EmotionDataset(root_dir=path_datasets, resample_rate=16_000)
    
    # Load EnCodec model
    model = EncodecModel.encodec_model_24khz(pretrained=True)
    model.eval()
    model.set_target_bandwidth(6.0)
    
    # Prepare storage
    all_embeddings = []
    all_labels = []
    all_sources = []
    
    print("Extracting embeddings from dataset...")
    
    # Process a subset for analysis
    num_samples = min(1000, len(dataset))
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Processing sample {i}/{num_samples}")
        
        audio, label, dataset_source = dataset[i]
        
        # Ensure audio is in correct format
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        # Extract embeddings
        encoder_emb = extract_encoder_embeddings(model, audio)
        global_emb = get_global_embeddings(encoder_emb, 'mean')
        
        all_embeddings.append(global_emb.squeeze().cpu().numpy())
        all_labels.append(label.item())
        all_sources.append(dataset_source)
    
    # Convert to numpy arrays
    embeddings_array = np.array(all_embeddings)
    labels_array = np.array(all_labels)
    sources_array = np.array(all_sources)
    
    # Save embeddings
    save_path = Path("embeddings_analysis")
    save_path.mkdir(exist_ok=True)
    
    np.save(save_path / "embeddings.npy", embeddings_array)
    np.save(save_path / "labels.npy", labels_array)
    np.save(save_path / "sources.npy", sources_array)
    
    print(f"\nSaved embeddings analysis to {save_path}")
    print(f"- Embeddings shape: {embeddings_array.shape}")
    print(f"- Labels shape: {labels_array.shape}")
    print(f"- Sources shape: {sources_array.shape}")
    
    # Print some statistics
    unique_labels, label_counts = np.unique(labels_array, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"- Label {label}: {count} samples")


if __name__ == "__main__":
    print("EnCodec Embeddings Extraction Demo")
    print("=" * 40)
    
    # Analyze embeddings
    analyze_embeddings()
    
    print("\n" + "=" * 40)
    
    # Save embeddings for analysis
    save_embeddings_for_analysis() 
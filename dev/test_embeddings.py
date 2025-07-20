"""
Simple test script to verify embeddings extraction works.
"""

import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

# Test with a simple audio signal first
def test_simple_audio():
    """Test embeddings extraction with a simple synthetic audio signal."""
    
    print("Testing with synthetic audio...")
    
    # Create a simple audio signal (1 second of 440 Hz sine wave)
    sample_rate = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Add batch and channel dimensions: [1, 1, T]
    audio = audio.unsqueeze(0).unsqueeze(0)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    try:
        from encodec import EncodecModel
        
        # Load EnCodec model
        model = EncodecModel.encodec_model_24khz(pretrained=True)
        model.eval()
        
        print(f"EnCodec model loaded successfully")
        print(f"- Sample rate: {model.sample_rate}")
        print(f"- Embedding dimension: {model.encoder.dimension}")
        
        # Test encoder embeddings
        with torch.no_grad():
            # Normalize if needed
            if model.normalize:
                mono = audio.mean(dim=1, keepdim=True)
                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                scale = 1e-8 + volume
                audio = audio / scale
            
            # Get embeddings
            embeddings = model.encoder(audio)
            
        print(f"Encoder embeddings shape: {embeddings.shape}")
        print(f"Embeddings stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        
        # Test global pooling
        global_emb = embeddings.mean(dim=-1)
        print(f"Global embeddings shape: {global_emb.shape}")
        
        print("✅ Synthetic audio test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install encodec: pip install encodec")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_with_dataset():
    """Test embeddings extraction with actual dataset."""
    
    print("\nTesting with actual dataset...")
    
    try:
        from dataset import EmotionDataset
        
        # Load environment
        load_dotenv()
        path_datasets = Path(os.getenv("PATH_DATASETS"))
        
        if not path_datasets.exists():
            print(f"❌ Dataset path not found: {path_datasets}")
            print("Please set PATH_DATASETS in your .env file")
            return False
        
        # Load dataset
        dataset = EmotionDataset(root_dir=path_datasets, resample_rate=16_000)
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Test with first sample
        audio, label, dataset_source = dataset[0]
        
        # Ensure audio is in correct format
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        print(f"Sample audio shape: {audio.shape}")
        print(f"Label: {label}, Dataset source: {dataset_source}")
        
        # Test embeddings extraction
        from encodec import EncodecModel
        
        model = EncodecModel.encodec_model_24khz(pretrained=True)
        model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model.encoder(audio)
            global_emb = embeddings.mean(dim=-1)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Global embeddings shape: {global_emb.shape}")
        
        print("✅ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_model_integration():
    """Test the integrated model with embeddings."""
    
    print("\nTesting model integration...")
    
    try:
        from model import EmotionDetectionModel
        
        # Create model
        model = EmotionDetectionModel(num_classes=8)
        print(f"Model created successfully")
        print(f"Embedding dimension: {model.embedding_dim}")
        
        # Test with synthetic audio
        audio = torch.randn(1, 1, 24000)  # 1 second of random audio
        
        # Get embeddings
        embeddings = model.get_embeddings(audio)
        print(f"Model embeddings shape: {embeddings.shape}")
        
        # Test forward pass
        logits = model(audio)
        print(f"Model output shape: {logits.shape}")
        
        print("✅ Model integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("EnCodec Embeddings Test")
    print("=" * 30)
    
    # Run tests
    test1 = test_simple_audio()
    test2 = test_with_dataset()
    test3 = test_model_integration()
    
    print("\n" + "=" * 30)
    print("Test Results:")
    print(f"Simple audio test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Dataset test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Model integration test: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! Embeddings extraction is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.") 
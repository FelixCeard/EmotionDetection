import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
import os
import re
import soundfile as sf
import librosa
import random


class EmotionDataset(Dataset):

    def __init__(self, root_dir, resample_rate=16_000, n_mfcc=20, n_fft=64, use_augmentation=False):

        self.root_dir = root_dir
        self.resample_rate = resample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft

        self.audio_files = []
        self.labels = []
        # the labels are:
        # 1: anger
        # 2: disgust
        # 3: fear
        # 4: happy
        # 5: neutral
        # 6: sad
        # 7: surprise
        
        self.dataset_source = []
        # the dataset source is:
        # 1: crema_d
        # 2: esd
        # 3: jl_corpus
        # 4: ravdess - Actors
        # 5: ravdess - Speech

        self.indices = None

        self.setup_datasets()

        if use_augmentation:
            self.transform = create_train_data_augmentation()
        else:
            self.transform = None

    def setup_datasets(self):
        # read the different datasets and save the audio files and labels

        # ? CREMA-D
        print('Loading CREMA-D...')
        crema_d_path = self.root_dir / "CREMA-D" / "AudioWAV"
        crema_d_files = crema_d_path.glob("*.wav")
        crema_d_conversion_dict = {
            "ANG": 1,
            "DIS": 2,
            "FEA": 3,
            "HAP": 4,
            "NEU": 5,
            "SAD": 6,
            # "SUR": 7
        }


        for file in crema_d_files:
            label = file.stem.split("_")[-2]
            label = crema_d_conversion_dict[label]
            
            self.audio_files.append(file)
            self.labels.append(label)
            self.dataset_source.append(1)

        
        # ? ESD
        print('Loading ESD...')
        esd_path = self.root_dir / "ESD"
        esd_conversion_dict = {
            "Angry": 1,
            # "Discust": 2,
            # "Fear": 3,
            "Happy": 4,
            "Neutral": 5,
            "Sad": 6,
            "Surprise": 7
        }

        
        for speaker in esd_path.iterdir():
            if not speaker.is_dir():
                continue
            for emotion in speaker.iterdir():
                if not emotion.is_dir():
                    continue
                for file in emotion.glob("*.wav"):
                    label = esd_conversion_dict[emotion.name]

                    self.audio_files.append(file)
                    self.labels.append(label)
                    self.dataset_source.append(2)


        # ? JL-Corpus
        print('Loading JL-Corpus...')
        jl_corpus_path = self.root_dir / "JL-Corpus" / 'Raw JL corpus (unchecked and unannotated)' / 'JL(wav+txt)'
        files = jl_corpus_path.glob("*.wav")
        jl_corpus_conversion_dict = {
            "angry": 1,
            # "discust": 2,
            # "fear": 3,
            "happy": 4,
            "neutral": 5,
            "sad": 6,
            "surprise": 7,
            "anxious": 3, # we use it as fear here
            "apologetic": None,
            "assertive": None, 
            "concerned": 3, # we use it as fear here
            "encouraging": None, 
            "excited": 4,
        }

        for file in files:
            label = file.stem.split("_")[1]
            label = jl_corpus_conversion_dict[label]
            if label is None:
                continue # skip the file

            self.audio_files.append(file)
            self.labels.append(label)
            self.dataset_source.append(3)


        # ? RAVDESS Actors
        print('Loading RAVDESS Actors...')
        ravdess_actors_path = self.root_dir / "RAVDESS" / "actors"
        ravdess_conversion_dict = {
            "01": 5,
            "02": 5, # is neutral too
            "03": 4,
            "04": 6,
            "05": 1,
            "06": 3,
            "07": 2,
            "08": 7
        }

        for actor in ravdess_actors_path.iterdir():
            if not actor.is_dir():
                continue
            for file in actor.glob("*.wav"):
                
                label = file.stem.split("-")[2]
                label = ravdess_conversion_dict[label]

                self.audio_files.append(file)
                self.labels.append(label)
                self.dataset_source.append(4)


        # ? RAVDESS Speech
        print('Loading RAVDESS Speech...')
        ravdess_speech_path = self.root_dir / "RAVDESS" / "speech"
        ravdess_conversion_dict = {
            "01": 5,
            "02": 5, # is neutral too
            "03": 4,
            "04": 6,
            "05": 1,
            "06": 3,
            "07": 2,
            "08": 7
        }

        for actor in ravdess_speech_path.iterdir():
            if not actor.is_dir():
                continue
            for file in actor.glob("*.wav"):
                
                label = file.stem.split("-")[2]
                label = ravdess_conversion_dict[label]

                self.audio_files.append(file)
                self.labels.append(label)
                self.dataset_source.append(4)


        # ? SAVE-E
        print('Loading SAVE-E...')
        savee_path = self.root_dir / "SAVE-E" / "ALL"
        savee_conversion_dict = {
            "a": 1,
            "d": 2,
            "f": 3,
            "h": 4,
            "n": 5,
            "sa": 6,
            "su": 7
        }

        for file in savee_path.glob("*.wav"):
            label = re.findall(r'([a-z]+)\d+', file.stem)[0]
            label = savee_conversion_dict[label]

            self.audio_files.append(file)
            self.labels.append(label)
            self.dataset_source.append(5)

        # ? TESS
        print('Loading TESS...')
        tess_path = self.root_dir / "TESS" / "TESS Toronto emotional speech set data"
        tess_conversion_dict = {
            "angry": 1,
            "disgust": 2,
            "fear": 3,
            "happy": 4,
            "neutral": 5,
            "sad": 6,
            "pleasant_surprise": 7,
            "surprise": 7,
            "surprised": 7
        }

        for emotion in tess_path.iterdir():
            if not emotion.is_dir():
                continue
            if emotion.name == "TESS Toronto emotional speech set data":
                continue

            label = emotion.name.split("_")[-1]
            label = tess_conversion_dict[label.lower()]

            for file in emotion.glob("*.wav"):
                self.audio_files.append(file)
                self.labels.append(label)
                self.dataset_source.append(6)

        print('Loaded all datasets')

    def __len__(self):
        if self.indices is None:
            return len(self.audio_files)
        else:
            return len(self.indices)
    
    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        dataset_source = self.dataset_source[idx]

        # Use soundfile to read the audio file as a workaround for torchaudio backend issues
        X, sr = librosa.load(str(audio_file), sr=self.resample_rate, mono=True)

        if self.transform is not None:
            X = self.transform(X)

        # remove the silences
        X, _ = librosa.effects.trim(X, top_db=20)
       
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.n_fft//2)

        # to tensor
        mfccs = torch.tensor(mfccs, dtype=torch.float32)

        # mel_spec = librosa.feature.melspectrogram(y=X, sr=sr, n_fft=self.n_fft, hop_length=self.n_fft//2)
        # mel_spec = torch.tensor(mel_spec, dtype=torch.float32)

        # chroma = librosa.feature.chroma_stft(y=X, sr=sr, n_fft=self.n_fft, hop_length=self.n_fft//2)
        # chroma = torch.tensor(chroma, dtype=torch.float32)

        # X = torch.cat([mfccs], dim=0)
        X = mfccs

        return X, label
    
    def set_indices(self, indices):
        self.indices = indices

def pad_collate_fn(batch, max_mfcc_length=3000):
    """
    Custom collate function to pad MFCCs to the same length within a batch.
    
    Args:
        batch: List of (mfccs, label) tuples
        max_mfcc_length: Maximum length to pad/truncate MFCCs to
        
    Returns:
        padded_mfccs: Tensor of shape (batch_size, n_mfcc, max_mfcc_length)
        labels: Tensor of shape (batch_size,)
    """
    mfccs_list, labels = zip(*batch)
    
    # Find the maximum length in this batch
    max_length_in_batch = max(mfcc.shape[1] for mfcc in mfccs_list)
    
    # Use the smaller of max_length_in_batch or max_mfcc_length
    target_length = min(max_length_in_batch, max_mfcc_length)
    
    padded_mfccs = []
    for mfcc in mfccs_list:
        current_length = mfcc.shape[1]
        
        if current_length > target_length:
            # Truncate if too long
            padded_mfcc = mfcc[:, :target_length]
        elif current_length < target_length:
            # Pad with zeros if too short
            padding_length = target_length - current_length
            padded_mfcc = np.pad(mfcc, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
        else:
            padded_mfcc = mfcc
            
        padded_mfccs.append(padded_mfcc)
    
    # Convert to tensors
    padded_mfccs = torch.tensor(np.array(padded_mfccs), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_mfccs, labels

def create_dataloader(dataset: EmotionDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, max_mfcc_length: int = 3000):
    """
    Create a DataLoader with custom padding for variable-length MFCCs.
    
    Args:
        dataset: EmotionDataset instance
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        max_mfcc_length: Maximum length to pad/truncate MFCCs to
        
    Returns:
        DataLoader with custom collate function
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=lambda batch: pad_collate_fn(batch, max_mfcc_length)
    )

def create_dataloaders(batch_size: int, num_workers: int = 0, max_mfcc_length: int = 3000, **kwargs):
    dataset_train = EmotionDataset(root_dir=Path(os.getenv("PATH_DATASETS")), resample_rate=16_000, use_augmentation=True, **kwargs)
    dataset_test = EmotionDataset(root_dir=Path(os.getenv("PATH_DATASETS")), resample_rate=16_000, use_augmentation=False, **kwargs)
    
    indices = list(range(len(dataset_train)))
    random.shuffle(indices)
    train_indices = indices[:int(len(indices) * 0.8)]
    test_indices = indices[int(len(indices) * 0.8):]

    dataset_train.set_indices(train_indices)
    dataset_test.set_indices(test_indices)

    return create_dataloader(dataset_train, batch_size, True, num_workers, max_mfcc_length), create_dataloader(dataset_test, batch_size, False, num_workers, max_mfcc_length)

def create_train_data_augmentation(augment_prob=0.3, sr=16000):
    """
    Create a simple audio augmentation transform for raw audio.
    
    Args:
        augment_prob: Probability of applying augmentation (default: 0.7)
        sr: Sample rate (default: 16000)
    
    Returns:
        A callable transform function that takes audio and returns augmented audio.
    """
    def transform(audio):
        """
        Apply augmentation to raw audio.
        
        Args:
            audio: Raw audio numpy array
            
        Returns:
            Augmented audio numpy array
        """
        if random.random() > augment_prob:
            return audio
        
        # Time stretching (speed up/slow down)
        if random.random() < 0.3:
            rate = random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting
        if random.random() < 0.3:
            steps = random.randint(-4, 4)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        
        # Add noise
        if random.random() < 0.2:
            noise_level = random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise
        
        # Volume scaling
        if random.random() < 0.3:
            scale = random.uniform(0.7, 1.3)
            audio = audio * scale
        
        return audio
    
    return transform




if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    dataset = EmotionDataset(root_dir=Path(os.getenv("PATH_DATASETS")), resample_rate=16_000)

    print(len(dataset))
    print(dataset[0])
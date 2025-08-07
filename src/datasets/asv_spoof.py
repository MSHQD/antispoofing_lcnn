import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Optional


class ASVSpoofDataset(Dataset):
    """Dataset for ASVspoof2019 dataset."""
    
    def __init__(
        self,
        root_dir: str,
        partition: str = "train",
        transform: Optional[Any] = None,
        sample_rate: int = 16000,
        max_length: Optional[int] = None
    ):
        """
        Initialize ASVSpoofDataset.
        
        Args:
            root_dir: Root directory of the dataset
            partition: Dataset partition ('train', 'dev', 'eval')
            transform: Optional transform to apply to audio
            sample_rate: Target sample rate
            max_length: Maximum audio length in samples
        """
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Define label mapping
        self.label_map = {
            'bonafide': 0,
            'spoof': 1
        }
        
        # Load file list and labels
        self.files, self.labels = self._load_file_list()
        
    def _load_file_list(self):
        """Load list of audio files and their labels."""
        files = []
        labels = []
        
        # Define paths based on partition
        if self.partition == "train":
            audio_dir = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.train.trl")
            protocol_file = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.train.trl.txt")
        elif self.partition == "dev":
            audio_dir = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.dev.trl")
            protocol_file = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
        elif self.partition == "eval":
            audio_dir = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.eval.trl")
            protocol_file = os.path.join(self.root_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
        else:
            raise ValueError(f"Unknown partition: {self.partition}")
        
        # Check if protocol file exists
        if not os.path.exists(protocol_file):
            print(f"Warning: Protocol file {protocol_file} not found. Using dummy data.")
            # Create dummy data for testing
            dummy_files = [f"dummy_{i}.flac" for i in range(10)]
            dummy_labels = [0 if i % 2 == 0 else 1 for i in range(10)]
            return dummy_files, dummy_labels
        
        # Read protocol file
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    file_name = parts[1]
                    label = parts[3]
                    
                    # Map label to integer
                    label_int = self.label_map.get(label, 0)
                    
                    # Check if audio file exists
                    audio_path = os.path.join(audio_dir, f"{file_name}.flac")
                    if os.path.exists(audio_path):
                        files.append(audio_path)
                        labels.append(label_int)
        
        return files, labels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """Get audio file and label."""
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to numpy array
            audio = waveform.squeeze().numpy()
            
            # Pad or truncate to max_length
            if self.max_length is not None:
                if len(audio) > self.max_length:
                    audio = audio[:self.max_length]
                else:
                    # Pad with zeros
                    padding = self.max_length - len(audio)
                    audio = np.pad(audio, (0, padding), 'constant')
            
            # Convert to torch tensor first
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Apply transform if provided
            if self.transform is not None:
                audio_tensor = self.transform(audio_tensor)
                # Add channel dimension if needed (for spectrograms)
                if audio_tensor.dim() == 3:  # [batch, freq, time]
                    audio_tensor = audio_tensor.unsqueeze(1)  # [batch, 1, freq, time]
                elif audio_tensor.dim() == 2:  # [freq, time] - single sample
                    audio_tensor = audio_tensor.unsqueeze(0)  # [1, freq, time]
                # Remove extra dimensions if we have too many
                if audio_tensor.dim() == 5:  # [batch, extra, channels, freq, time]
                    audio_tensor = audio_tensor.squeeze(1)  # [batch, channels, freq, time]
            
            return {
                'audio': audio_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'file_name': os.path.basename(audio_path)
            }
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data on error
            dummy_audio = torch.zeros(self.max_length or 16000, dtype=torch.float32)
            if self.transform is not None:
                dummy_audio = self.transform(dummy_audio)
                # Add channel dimension if needed
                if dummy_audio.dim() == 3:  # [batch, freq, time]
                    dummy_audio = dummy_audio.unsqueeze(1)  # [batch, 1, freq, time]
                elif dummy_audio.dim() == 2:  # [freq, time] - single sample
                    dummy_audio = dummy_audio.unsqueeze(0)  # [1, freq, time]
                # Remove extra dimensions if we have too many
                if dummy_audio.dim() == 5:  # [batch, extra, channels, freq, time]
                    dummy_audio = dummy_audio.squeeze(1)  # [batch, channels, freq, time]
            return {
                'audio': dummy_audio,
                'label': torch.tensor(label, dtype=torch.long),
                'file_name': os.path.basename(audio_path)
            }
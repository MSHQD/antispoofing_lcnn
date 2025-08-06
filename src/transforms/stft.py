import torch
import torchaudio
from typing import Optional, Tuple


class STFTTransform:
    """Short-time Fourier transform for audio preprocessing."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[str] = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        return_complex: bool = False,
        power: float = 2.0  # Power of the magnitude spectrogram
    ):
        """
        Args:
            n_fft (int): Size of FFT
            hop_length (int, optional): Number of samples between successive frames
            win_length (int, optional): Window size
            window (str, optional): Window function type
            center (bool): Whether to pad on both sides
            pad_mode (str): Padding method
            normalized (bool): Whether to normalize STFT
            onesided (bool): Return half of frequencies
            return_complex (bool): Return complex tensor instead of real and imaginary parts
            power (float): Power of the magnitude spectrogram (1.0 for magnitude, 2.0 for power)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.return_complex = return_complex
        self.power = power
        
        # Create window tensor
        self.window_tensor = getattr(torch, f"{window}_window")(
            self.win_length, dtype=torch.float32
        )
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to audio waveform and return magnitude spectrogram.
        
        Args:
            waveform (torch.Tensor): Audio of shape [channels, time]
            
        Returns:
            torch.Tensor: Magnitude spectrogram of shape [channels, freq, time]
        """
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_tensor,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True  # Always return complex for simpler processing
        )
        
        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)
        
        # Apply power
        if self.power != 1.0:
            magnitude = magnitude.pow(self.power)
            
        return magnitude 

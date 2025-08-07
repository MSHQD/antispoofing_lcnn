import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MaxFeatureMap2D(nn.Module):
    """Max feature map activation.
    
    MaxFeatureMap2D takes a tensor as input and max-out on the channel/feature dimension.
    The channel/feature dimension must be even number.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            assert x.size(1) % 2 == 0, f"Channel dimension must be even, got {x.size(1)}"

            out = torch.max(x.view(x.size(0), -1, 2, x.size(2), x.size(3)), dim=2)[0]
        elif x.dim() == 2:
            assert x.size(1) % 2 == 0, f"Feature dimension must be even, got {x.size(1)}"

            out = torch.max(x.view(x.size(0), -1, 2), dim=2)[0]
        else:
            raise ValueError(f"Input must be 2D or 4D tensor, got {x.dim()}D")
            
        return out


class ASoftmaxLoss(nn.Module):
    """A-Softmax loss for better feature learning."""
    
    def __init__(self, margin=4, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        
    def forward(self, x, labels):
        x_norm = F.normalize(x, p=2, dim=1)
        cos_theta = x_norm
        cos_theta_m = torch.cos(self.margin * torch.acos(torch.clamp(cos_theta, -1, 1)))
        output = cos_theta_m * self.scale
        
        return F.cross_entropy(output, labels)


class LCNN(nn.Module):
    """Light CNN for anti-spoofing detection based on Speech Technology Center paper."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.5):
        """
        Args:
            in_channels (int): Number of input channels (1 for mono audio)
            num_classes (int): Number of output classes (2 for bonafide/spoof)
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
          
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64),
          
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),

            nn.MaxPool2d(2, 2),
        )

        # Use actual input size for feature calculation
        dummy_input = torch.zeros(1, in_channels, 257, 101)  # Updated size
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 160),
            MaxFeatureMap2D(),
            nn.BatchNorm1d(80),
            nn.Dropout(dropout),
            nn.Linear(80, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input spectrogram of shape [batch, channels, freq, time]
            
        Returns:
            tuple:
                torch.Tensor: Logits of shape [batch, num_classes]
                torch.Tensor: Features before classifier of shape [batch, features]
        """
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        logits = self.classifier(features_flat)
        return logits, features_flat


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for imbalanced classes."""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, logits, labels):
        if self.weights is not None:
            weights = self.weights.to(logits.device)
            return F.cross_entropy(logits, labels, weight=weights)
        else:
            return F.cross_entropy(logits, labels) 

import numpy as np
import torch
from typing import Tuple
from sklearn.metrics import roc_curve

from .base_metric import BaseMetric


class EERMetric(BaseMetric):
    """Equal Error Rate (EER) metric for anti-spoofing evaluation."""
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        """Reset metric state."""
        self.predictions = []
        self.targets = []
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metric state with batch predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions (logits) of shape [batch, 2]
            target (torch.Tensor): Ground truth labels of shape [batch]
        """
        # Convert logits to probabilities and take spoof probability
        probs = torch.softmax(preds, dim=1)
        spoof_probs = probs[:, 1]  # Probability of being spoof
        
        # Convert to numpy and store
        self.predictions.extend(spoof_probs.cpu().numpy())
        self.targets.extend(target.cpu().numpy())
        
    def compute(self) -> Tuple[float, float, float]:
        """
        Compute EER and corresponding threshold.
        
        Returns:
            tuple:
                float: Equal Error Rate (EER)
                float: False Acceptance Rate (FAR) at EER threshold
                float: False Rejection Rate (FRR) at EER threshold
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(targets, predictions)
        
        # False Rejection Rate = 1 - True Positive Rate
        frr = 1 - tpr
        
        # Find threshold where FAR (=FPR) = FRR
        abs_diff = np.abs(fpr - frr)
        min_index = np.argmin(abs_diff)
        
        # Get EER and threshold
        eer = np.mean([fpr[min_index], frr[min_index]])
        far = fpr[min_index]
        frr = frr[min_index]
        
        return eer, far, frr 

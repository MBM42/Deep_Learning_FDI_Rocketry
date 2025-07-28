"""
focal_loss.py

Focal loss dynamically addresses challening classes that are easily misclassified and underpredicted.
It offers an advantage over cross-entropy loss, where models are overwhelmed by the dominant class, negleting the minority class.

This is achieved by introducing a scaling factor - (1 - p_t)**gamma (where p_t is the predicted probability for the true class).
If p_t is low (the model is uncertain), the scaling factor will be large, making the loss higher for that sample.

Parameters:
- gamma (γ): A higher gamma reduces the loss contribution from well-classified examples more aggressively.
- alpha (α): A balancing factor used to assign different weights to classes.

Tuning gamma and alpha for Tabular Data:
- gamma (γ): Values of 1–2 often work well, focusing the loss without overwhelming model stability.
             A gamma of 0 shall be equivalent to the standard Cross-Entropy Loss
- alpha (α): For Highly Skewed Datasets  where the minority class is extremely rare, setting alpha around 0.25 
             for the majority class and 0.75 for the minority class can be effective. In fact, alpha can sometimes
             be used to balance gamma. Giving the minority class a lower alpha than that of the majority class.

Author: Miguel Marques
Date: 06-04-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class FocalLoss(nn.Module):
    """
    Parameters:
        gamma (float, optional): Focusing parameter. Default is 2.0.
        weight (torch.Tensor, optional): Class weighting (shape: [num_classes]). Default is None
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Default is 'mean'
    """

    def __init__(
        self, 
        gamma: Optional[float] = 2.0, 
        weight: Optional[Tensor] = None, 
        reduction: Optional[str] = 'mean',
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets)-> torch.Tensor:
        """
        Params:
            inputs (torch.float32): model outputs - logits - shape: [batch_size, num classes]
            targets (torch.long): ground truth class index for each sample - shape: [barch_size]
        """

        # Compute log probabilities (log_softmax: applies a normalization to the logits followed by the natural log)
        logp = F.log_softmax(inputs, dim=1)
        
        # Compute standard cross entropy loss (without reduction)
        # ce_loss will be a tensor of shape [batch_size], containing the loss for each sample
        ce_loss = F.nll_loss(logp, targets, reduction='none', weight=self.alpha)
        
        # Get the probabilities of the true classes
        # - ce_loss is computed as -log(p_t) (where p_t is the probability of the true class)
        # - p_t is recovered by taking the exponential of the negative loss - shape: [batch_size]
        pt = torch.exp(-ce_loss)
        
        # Compute the focal loss - shape: [batch_size]
        loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply the specified reduction:
        # - If 'mean' - average the loss over the batch - Shape: [single scalar value]
        # - If 'sum' - sum the loss over the batch - Shape: [single scalar value]
        # - Otherwise - return the per-sample losses - Shape: [batch_size]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @property
    def weight(self):
        """
        Alias for self.alpha to match logging expectations.
        Whenever a caller searches for a "weight" attribute of FocalLoss, the attribute alpha is returned.
        """
        return self.alpha

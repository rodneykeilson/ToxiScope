"""
ToxiScope: Loss Functions for Multilabel Classification

Implements various loss functions designed to handle class imbalance:
- Focal Loss: Down-weights easy examples
- Asymmetric Loss: Different treatment for positive/negative
- Label Smoothing BCE: Reduces overconfidence
- Weighted BCE: Class-weighted binary cross-entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """
    Focal Loss for multilabel classification.
    
    Focal Loss addresses class imbalance by down-weighting well-classified
    examples and focusing on hard, misclassified examples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            alpha: Weighting factor for positive examples (0-1)
            gamma: Focusing parameter (γ >= 0)
            reduction: 'none', 'mean', or 'sum'
            pos_weight: Per-class positive weights
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_labels) - raw scores
            targets: (batch, num_labels) - binary labels
        
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Binary cross-entropy component
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal weight: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combined loss
        loss = alpha_t * focal_weight * bce
        
        # Apply class weights if provided
        if self.pos_weight is not None:
            # Expand pos_weight to match batch
            weight = self.pos_weight.unsqueeze(0).expand_as(loss)
            loss = loss * (targets * weight + (1 - targets))
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multilabel classification.
    
    Applies different focusing parameters for positive and negative samples,
    which is particularly effective for highly imbalanced datasets.
    
    Reference:
        Ben-Baruch et al. "Asymmetric Loss for Multi-Label Classification" (2020)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples  
            clip: Probability clipping for negative samples
            reduction: 'none', 'mean', or 'sum'
            eps: Small value for numerical stability
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_labels)
            targets: (batch, num_labels)
        
        Returns:
            Asymmetric loss value
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Separate positive and negative samples
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric clipping for negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Basic cross-entropy
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        
        # Asymmetric focusing
        if self.gamma_pos > 0:
            pt_pos = probs_pos * targets + (1 - targets)
            focal_pos = (1 - pt_pos) ** self.gamma_pos
            loss_pos = focal_pos * loss_pos
        
        if self.gamma_neg > 0:
            pt_neg = probs_neg * (1 - targets) + targets
            focal_neg = (1 - pt_neg) ** self.gamma_neg
            loss_neg = focal_neg * loss_neg
        
        loss = -(loss_pos + loss_neg)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    
    Softens the targets to reduce overconfidence and improve calibration.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            smoothing: Label smoothing factor (0-0.5)
            reduction: 'none', 'mean', or 'sum'
            pos_weight: Per-class positive weights
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_labels)
            targets: (batch, num_labels)
        
        Returns:
            Label-smoothed BCE loss
        """
        targets = targets.float()
        
        # Apply label smoothing
        # targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
        # This shifts 0 -> smoothing/2, and 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for class imbalance.
    
    Computes class weights based on inverse frequency.
    """
    
    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            class_counts: Number of positive samples per class
            num_samples: Total number of samples
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.reduction = reduction
        
        if class_counts is not None and num_samples is not None:
            # Compute inverse frequency weights
            # weight = num_negative / num_positive
            num_negative = num_samples - class_counts
            self.pos_weight = num_negative / (class_counts + 1e-8)
        else:
            self.pos_weight = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_labels)
            targets: (batch, num_labels)
        
        Returns:
            Weighted BCE loss
        """
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)
        
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=pos_weight,
            reduction=self.reduction,
        )
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function mixing multiple loss types.
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        bce_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        
        if label_smoothing > 0:
            self.bce_loss = LabelSmoothingBCE(
                smoothing=label_smoothing,
                pos_weight=pos_weight,
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        
        return self.focal_weight * focal + self.bce_weight * bce


def get_loss_function(
    loss_type: str,
    num_labels: int = 7,
    class_counts: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: One of 'bce', 'focal', 'asymmetric', 'weighted', 'combined'
        num_labels: Number of labels
        class_counts: Per-class positive counts for weighting
        num_samples: Total samples for weight computation
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        label_smoothing: Label smoothing factor
    
    Returns:
        Configured loss function
    """
    # Compute pos_weight if class counts provided
    pos_weight = None
    if class_counts is not None and num_samples is not None:
        num_negative = num_samples - class_counts
        pos_weight = num_negative / (class_counts + 1e-8)
    
    if loss_type == "bce":
        if label_smoothing > 0:
            return LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
    
    elif loss_type == "asymmetric":
        return AsymmetricLoss(**kwargs)
    
    elif loss_type == "weighted":
        return WeightedBCELoss(class_counts=class_counts, num_samples=num_samples)
    
    elif loss_type == "combined":
        return CombinedLoss(
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
            pos_weight=pos_weight,
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    num_labels = 7
    
    logits = torch.randn(batch_size, num_labels)
    targets = torch.randint(0, 2, (batch_size, num_labels)).float()
    
    print("Testing loss functions...")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print()
    
    # Test each loss
    losses = {
        "BCE": nn.BCEWithLogitsLoss(),
        "Focal": FocalLoss(),
        "Asymmetric": AsymmetricLoss(),
        "LabelSmoothing": LabelSmoothingBCE(smoothing=0.1),
    }
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(logits, targets)
        print(f"{name}: {loss_value.item():.4f}")

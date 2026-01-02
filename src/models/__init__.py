"""
ToxiScope Models Module

Contains all model architectures for toxic comment classification.
Primary model: DeBERTa-v3 (state-of-the-art on Jigsaw toxicity benchmarks)
"""

from .deberta import (
    ToxiScopeModel,
    ToxiScopeConfig,
    AttentionPooling,
    MultiSampleDropout,
    create_toxiscope_model,
    ToxiScopeForSequenceClassification,
)

from .bilstm_cnn import (
    BiLSTM_CNN,
    TextCNN,
    create_bilstm_cnn,
)

from .focal_loss import (
    FocalLoss,
    AsymmetricLoss,
    LabelSmoothingBCE,
    WeightedBCELoss,
    CombinedLoss,
    get_loss_function,
)

__all__ = [
    # DeBERTa-v3 / RoBERTa
    "ToxiScopeModel",
    "ToxiScopeConfig", 
    "AttentionPooling",
    "MultiSampleDropout",
    "create_toxiscope_model",
    "ToxiScopeForSequenceClassification",
    # BiLSTM-CNN
    "BiLSTM_CNN",
    "TextCNN",
    "create_bilstm_cnn",
    # Loss functions
    "FocalLoss",
    "AsymmetricLoss",
    "LabelSmoothingBCE",
    "WeightedBCELoss",
    "CombinedLoss",
    "get_loss_function",
]

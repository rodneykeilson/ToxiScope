"""
ToxiScope Source Package

Deep learning pipeline for toxic comment detection in gaming communities.
Powered by DeBERTa-v3, state-of-the-art on Jigsaw toxicity benchmarks.
"""

__version__ = "2.0.0"
__author__ = "ToxiScope Team"

from .models import (
    ToxiScopeModel,
    ToxiScopeConfig,
    create_toxiscope_model,
    FocalLoss,
    get_loss_function,
)

from .preprocessing import (
    EnglishTextNormalizer,
    normalize_text,
    TextAugmenter,
)

from .inference import ToxiScopePredictor

__all__ = [
    "ToxiScopeModel",
    "ToxiScopeConfig", 
    "create_toxiscope_model",
    "FocalLoss",
    "get_loss_function",
    "EnglishTextNormalizer",
    "normalize_text",
    "TextAugmenter",
    "ToxiScopePredictor",
]

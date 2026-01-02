"""
ToxiScope Preprocessing Module

English NLP preprocessing pipeline for toxic comment detection.
Optimized for gaming community content from Reddit.
"""

from .normalizer import (
    EnglishTextNormalizer,
    normalize_text,
    batch_normalize,
    create_normalizer_for_training,
    create_normalizer_for_inference,
    ABBREVIATIONS,
)

from .augmentation import (
    TextAugmenter,
    MixUpAugmenter,
)

__all__ = [
    "EnglishTextNormalizer",
    "normalize_text",
    "batch_normalize",
    "create_normalizer_for_training",
    "create_normalizer_for_inference",
    "ABBREVIATIONS",
    "TextAugmenter",
    "MixUpAugmenter",
]

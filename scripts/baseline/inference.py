#!/usr/bin/env python3
"""Standalone helper for running the saved baseline toxicity model.

Usage:
    python -m scripts.baseline.inference "This is toxic text" "This is clean text"
    python -m scripts.baseline.inference --model-dir outputs/models/baseline "some text"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts import PROJECT_ROOT, OUTPUTS_DIR

# Default model directory
MODEL_DIR = OUTPUTS_DIR / "models" / "baseline"


def _normalize_text(text: str | None) -> str:
    """Normalize text for inference."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\r", " ").replace("\n", " ")
    return " ".join(cleaned.split()).strip()


def _load_artifacts(model_dir: Path = MODEL_DIR) -> Tuple:
    """Load model artifacts from disk."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Baseline artifacts not found at {model_dir}. Run the training notebook first." 
        )
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    classifier = joblib.load(model_dir / "ovr_lr.joblib")
    labels = (model_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    return vectorizer, classifier, labels, thresholds


def _apply_thresholds(probs: np.ndarray, thresholds: dict, labels: Sequence[str]) -> np.ndarray:
    """Apply per-label thresholds to probabilities."""
    output = np.zeros_like(probs, dtype=int)
    for idx, label in enumerate(labels):
        thr = thresholds.get(label, 0.5)
        output[:, idx] = (probs[:, idx] >= thr).astype(int)
    return output


class BaselineToxicityPredictor:
    """Wraps the saved TF–IDF + logistic regression model for inference."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.vectorizer, self.classifier, self.labels, self.thresholds = _load_artifacts(model_dir)

    def predict(self, texts: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Predict toxicity labels for a list of texts."""
        normalized = [_normalize_text(t) for t in texts]
        features = self.vectorizer.transform(normalized)
        probs = self.classifier.predict_proba(features)
        preds = _apply_thresholds(probs, self.thresholds, self.labels)
        return self.labels, probs, preds


def _format_output(text: str, labels: Sequence[str], probs: np.ndarray, preds: np.ndarray) -> str:
    """Format prediction output for display."""
    active = [lab for lab, flag in zip(labels, preds) if flag]
    prob_pairs = ", ".join(f"{lab}:{prob:.3f}" for lab, prob in zip(labels, probs))
    active_display = ", ".join(active) if active else "none"
    return (
        f"TEXT: {text}\n"
        f"Active labels: {active_display}\n"
        f"Probabilities: {prob_pairs}\n"
    )


def run_cli(args: argparse.Namespace) -> None:
    """Run inference from command line arguments."""
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    
    predictor = BaselineToxicityPredictor(model_dir)
    labels, probs, preds = predictor.predict(args.text)
    for text, prob_row, pred_row in zip(args.text, probs, preds):
        print(_format_output(text, labels, prob_row, pred_row))
        print("-" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the saved baseline toxicity model on input text."
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="One or more strings to score. Enclose multi-word inputs in quotes.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        help="Path to the exported baseline artifacts (default: outputs/models/baseline)",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_cli(cli_args)

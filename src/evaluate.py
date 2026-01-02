#!/usr/bin/env python3
"""
ToxiScope Evaluation Script

Evaluation suite for trained models.

Features:
- Per-label and aggregate metrics
- Confusion matrices
- Precision-Recall curves
- ROC curves
- Inference speed benchmarks
- Error analysis
- Journal-ready figures
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    classification_report,
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import normalize_text


# =============================================================================
# Dataset (reuse from train.py)
# =============================================================================

class ToxicityDataset(Dataset):
    """PyTorch Dataset for evaluation."""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = normalize_text(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_model_and_tokenizer(model_dir: str, device: str = "cuda"):
    """Load trained model and tokenizer."""
    model_path = Path(model_dir)
    
    # Check for best_model subdirectory
    if (model_path / "best_model").exists():
        model_path = model_path / "best_model"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Load model using standard HuggingFace API
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    # Get labels from config
    if hasattr(model.config, 'id2label'):
        label_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    else:
        label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "racism"]
    
    # Load thresholds
    thresholds_path = model_path.parent / "thresholds.json"
    if not thresholds_path.exists():
        thresholds_path = model_path / "thresholds.json"
    
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {label: 0.5 for label in label_names}
    
    return model, tokenizer, label_names, thresholds


def predict_batch(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a dataloader."""
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()
            
            all_logits.append(logits)
            all_labels.append(labels)
    
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    return logits, probs, labels


def apply_thresholds(
    probs: np.ndarray,
    thresholds: Dict[str, float],
    label_names: List[str],
) -> np.ndarray:
    """Apply per-label thresholds to probabilities."""
    preds = np.zeros_like(probs, dtype=int)
    
    for i, label in enumerate(label_names):
        t = thresholds.get(label, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)
    
    return preds


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict:
    """Compute metrics."""
    metrics = {}
    
    # Aggregate metrics
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["samples_f1"] = f1_score(y_true, y_pred, average="samples", zero_division=0)
    
    metrics["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    metrics["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Subset accuracy (exact match)
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-label metrics
    per_label = {}
    for i, label in enumerate(label_names):
        label_metrics = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "support": int(y_true[:, i].sum()),
        }
        
        # ROC-AUC (requires both classes)
        if len(np.unique(y_true[:, i])) > 1:
            label_metrics["roc_auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
            label_metrics["ap"] = average_precision_score(y_true[:, i], y_prob[:, i])
        else:
            label_metrics["roc_auc"] = np.nan
            label_metrics["ap"] = np.nan
        
        per_label[label] = label_metrics
    
    metrics["per_label"] = per_label
    
    # Confusion matrices
    metrics["confusion_matrices"] = multilabel_confusion_matrix(y_true, y_pred)
    
    return metrics


def benchmark_inference_speed(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    num_runs: int = 3,
    batch_size: int = 1,
) -> Dict:
    """Benchmark inference speed."""
    model.eval()
    
    # Warm up
    sample_text = texts[0]
    encoding = tokenizer(
        sample_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
    
    # Benchmark single inference
    single_times = []
    for text in texts[:min(100, len(texts))]:
        encoding = tokenizer(
            text, truncation=True, max_length=256, padding="max_length", return_tensors="pt"
        )
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        single_times.append((time.perf_counter() - start) * 1000)  # ms
    
    # Benchmark batch inference
    batch_texts = texts[:min(batch_size * 10, len(texts))]
    batch_times = []
    
    for i in range(0, len(batch_texts), batch_size):
        batch = batch_texts[i:i+batch_size]
        encoding = tokenizer(
            batch, truncation=True, max_length=256, padding="max_length", return_tensors="pt"
        )
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - start) * 1000 / len(batch))
    
    return {
        "single_mean_ms": np.mean(single_times),
        "single_std_ms": np.std(single_times),
        "single_p50_ms": np.percentile(single_times, 50),
        "single_p95_ms": np.percentile(single_times, 95),
        "single_p99_ms": np.percentile(single_times, 99),
        "batch_mean_ms": np.mean(batch_times),
        "batch_std_ms": np.std(batch_times),
        "device": device,
        "batch_size": batch_size,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrices(
    confusion_matrices: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
):
    """Plot confusion matrices for all labels."""
    n_labels = len(label_names)
    n_cols = 4
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i, (cm, label) in enumerate(zip(confusion_matrices, label_names)):
        ax = axes[i]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
        )
        ax.set_title(label)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    
    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    fig.suptitle("Confusion Matrices per Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
):
    """Plot PR curves for all labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", linewidth=2)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
):
    """Plot ROC curves for all labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_metrics_comparison(
    models_metrics: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """Plot metrics comparison across models."""
    model_names = list(models_metrics.keys())
    metrics_to_plot = ["macro_f1", "micro_f1", "macro_precision", "macro_recall"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model_name in enumerate(model_names):
        values = [models_metrics[model_name].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8
            )
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def generate_metrics_table(
    metrics: Dict,
    label_names: List[str],
) -> pd.DataFrame:
    """Generate a metrics table for journal."""
    rows = []
    
    for label in label_names:
        label_metrics = metrics["per_label"][label]
        rows.append({
            "Label": label,
            "F1": f"{label_metrics['f1']:.3f}",
            "Precision": f"{label_metrics['precision']:.3f}",
            "Recall": f"{label_metrics['recall']:.3f}",
            "ROC-AUC": f"{label_metrics['roc_auc']:.3f}" if not np.isnan(label_metrics['roc_auc']) else "-",
            "AP": f"{label_metrics['ap']:.3f}" if not np.isnan(label_metrics['ap']) else "-",
            "Support": label_metrics['support'],
        })
    
    # Add aggregate row
    rows.append({
        "Label": "**Macro**",
        "F1": f"**{metrics['macro_f1']:.3f}**",
        "Precision": f"**{metrics['macro_precision']:.3f}**",
        "Recall": f"**{metrics['macro_recall']:.3f}**",
        "ROC-AUC": "-",
        "AP": "-",
        "Support": "-",
    })
    
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ToxiScope Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--test-data", type=str, default="data/processed/test.csv")
    parser.add_argument("--text-col", type=str, default="body_clean")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model, tokenizer, label_names, thresholds = load_model_and_tokenizer(args.model, device)
    
    print(f"Labels: {label_names}")
    print(f"Thresholds: {thresholds}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data, dtype=str, keep_default_na=False)
    
    texts = test_df[args.text_col].tolist()
    labels = test_df[label_names].values.astype(int)
    
    print(f"Test samples: {len(texts)}")
    
    # Create dataset and dataloader
    dataset = ToxicityDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Run inference
    print("\nRunning inference...")
    logits, probs, y_true = predict_batch(model, dataloader, device)
    
    # Apply thresholds
    y_pred = apply_thresholds(probs, thresholds, label_names)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(y_true, probs, y_pred, label_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Macro F1':<20} {metrics['macro_f1']:>10.4f}")
    print(f"{'Micro F1':<20} {metrics['micro_f1']:>10.4f}")
    print(f"{'Weighted F1':<20} {metrics['weighted_f1']:>10.4f}")
    print(f"{'Subset Accuracy':<20} {metrics['subset_accuracy']:>10.4f}")
    
    print(f"\n{'Label':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print("-" * 48)
    for label in label_names:
        m = metrics["per_label"][label]
        auc_str = f"{m['roc_auc']:.3f}" if not np.isnan(m['roc_auc']) else "N/A"
        print(f"{label:<15} {m['f1']:>8.3f} {m['precision']:>8.3f} {m['recall']:>8.3f} {auc_str:>8}")
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_table = generate_metrics_table(metrics, label_names)
    metrics_table.to_csv(os.path.join(args.output_dir, "evaluation_metrics.csv"), index=False)
    print(f"\nSaved metrics to {args.output_dir}/evaluation_metrics.csv")
    
    # Run benchmark
    if args.benchmark:
        print("\nRunning inference benchmark...")
        benchmark = benchmark_inference_speed(model, tokenizer, texts, device)
        
        print(f"\n{'Inference Speed Benchmark':^40}")
        print("-" * 40)
        print(f"Single inference (mean): {benchmark['single_mean_ms']:.2f} ms")
        print(f"Single inference (p95):  {benchmark['single_p95_ms']:.2f} ms")
        print(f"Single inference (p99):  {benchmark['single_p99_ms']:.2f} ms")
        print(f"Batch inference (mean):  {benchmark['batch_mean_ms']:.2f} ms per sample")
        
        with open(os.path.join(args.output_dir, "benchmark.json"), 'w') as f:
            json.dump(benchmark, f, indent=2)
    
    # Generate figures
    if args.figures:
        print("\nGenerating figures...")
        
        plot_confusion_matrices(
            metrics["confusion_matrices"], label_names,
            os.path.join(args.output_dir, "confusion_matrices.png")
        )
        
        plot_precision_recall_curves(
            y_true, probs, label_names,
            os.path.join(args.output_dir, "pr_curves.png")
        )
        
        plot_roc_curves(
            y_true, probs, label_names,
            os.path.join(args.output_dir, "roc_curves.png")
        )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

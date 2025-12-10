#!/usr/bin/env python3
"""
ToxiScope Training Script

Comprehensive training pipeline for DeBERTa-v3/RoBERTa-based toxicity detection.

Features:
- HuggingFace Transformers integration
- 5-fold cross-validation
- WandB experiment tracking
- Focal Loss for class imbalance
- Mixed precision training
- Gradient accumulation
- Model checkpointing
- Threshold calibration

Supported Models:
- microsoft/deberta-v3-base (recommended)
- microsoft/deberta-v3-large
- roberta-base / roberta-large

Usage:
    python train.py --config configs/deberta_base.yaml
    python train.py --config configs/deberta_base.yaml --wandb
    python train.py --config configs/deberta_large.yaml --kfold 5
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

# HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# Metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

# Stratification
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    STRATIFY_AVAILABLE = True
except ImportError:
    STRATIFY_AVAILABLE = False
    print("Warning: iterative-stratification not installed. K-fold CV disabled.")
    print("Install with: pip install iterative-stratification")

# Experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Experiment tracking disabled.")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Dataset
# =============================================================================

class ToxicityDataset(Dataset):
    """PyTorch Dataset for multilabel toxicity classification."""
    
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
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        # Tokenize
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
# Metrics
# =============================================================================

def compute_metrics(eval_pred, label_names: List[str], threshold: float = 0.5):
    """Compute comprehensive metrics for multilabel classification."""
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    
    # Per-label metrics
    for i, label in enumerate(label_names):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        y_prob = probs[:, i]
        
        metrics[f"{label}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            try:
                metrics[f"{label}_roc_auc"] = roc_auc_score(y_true, y_prob)
            except:
                pass
    
    return metrics


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """Calibrate per-label thresholds to maximize F1 score."""
    thresholds = {}
    
    for i, label in enumerate(label_names):
        best_threshold = 0.5
        best_f1 = 0.0
        
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        thresholds[label] = round(best_threshold, 2)
        print(f"  {label}: threshold={best_threshold:.2f}, F1={best_f1:.3f}")
    
    return thresholds


# =============================================================================
# Custom Trainer with Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ToxiScopeTrainer(Trainer):
    """Custom Trainer with Focal Loss support."""
    
    def __init__(self, *args, focal_gamma: float = 2.0, focal_alpha: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Training Functions
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str, text_col: str, label_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    """Load data from CSV file."""
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    
    # Get texts
    texts = df[text_col].tolist()
    
    # Get labels
    labels = np.zeros((len(df), len(label_cols)), dtype=np.float32)
    for i, col in enumerate(label_cols):
        if col in df.columns:
            labels[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).values
    
    return texts, labels


def train(
    config: Dict,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    output_dir: str,
    use_wandb: bool = False,
) -> Dict:
    """Execute training run."""
    
    # Extract config
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    
    model_name = model_config.get("name", "microsoft/deberta-v3-base")
    num_labels = model_config.get("num_labels", 7)
    max_length = config.get("data", {}).get("max_length", 256)
    
    label_cols = config.get("data", {}).get("label_columns", [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "racism"
    ])
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training ToxiScope with {model_name}")
    print(f"{'='*60}")
    print(f"Labels: {label_cols}")
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Load tokenizer with robust fallback handling
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Create datasets
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Load model
    print(f"Loading model from {model_name}...")
    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(label_cols)},
        label2id={label: i for i, label in enumerate(label_cols)},
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get("num_train_epochs", 5),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 32),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 2),
        learning_rate=float(train_config.get("learning_rate", 2e-5)),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        fp16=train_config.get("fp16", True) and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=train_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="wandb" if use_wandb and WANDB_AVAILABLE else "none",
        seed=seed,
        dataloader_num_workers=train_config.get("dataloader_num_workers", 0),
    )
    
    # Metrics function
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, label_cols)
    
    # Use Focal Loss
    use_focal = train_config.get("loss_function", "focal") == "focal"
    focal_gamma = train_config.get("focal_gamma", 2.0)
    focal_alpha = train_config.get("focal_alpha", 0.25)
    
    # Create trainer
    if use_focal:
        print(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        trainer = ToxiScopeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_wrapper,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=train_config.get("early_stopping_patience", 3)
            )],
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
    else:
        print("Using BCE Loss")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_wrapper,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=train_config.get("early_stopping_patience", 3)
            )],
        )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save best model
    best_dir = Path(output_dir) / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    
    # Calibrate thresholds
    print("\nCalibrating thresholds on validation set...")
    val_pred = trainer.predict(val_dataset)
    val_probs = torch.sigmoid(torch.tensor(val_pred.predictions)).numpy()
    
    thresholds = calibrate_thresholds(val_labels, val_probs, label_cols)
    
    # Save thresholds
    with open(best_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    
    # Save labels
    with open(best_dir / "labels.txt", "w") as f:
        f.write("\n".join(label_cols))
    
    # Final evaluation
    print("\nFinal evaluation with calibrated thresholds...")
    final_preds = np.zeros_like(val_probs)
    for i, label in enumerate(label_cols):
        final_preds[:, i] = (val_probs[:, i] >= thresholds[label]).astype(int)
    
    final_metrics = {
        "macro_f1": f1_score(val_labels, final_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(val_labels, final_preds, average="micro", zero_division=0),
    }
    
    print(f"\nFinal Metrics:")
    print(f"  Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1: {final_metrics['micro_f1']:.4f}")
    
    return {
        "metrics": final_metrics,
        "thresholds": thresholds,
        "model_dir": str(best_dir),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ToxiScope Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--kfold", type=int, default=0, help="K-fold CV (0 = disabled)")
    parser.add_argument("--output", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Paths
    data_config = config.get("data", {})
    train_path = data_config.get("train_path", "data/processed/train.csv")
    val_path = data_config.get("val_path", "data/processed/val.csv")
    text_col = data_config.get("text_column", "body_clean")
    label_cols = data_config.get("label_columns", [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "racism"
    ])
    
    output_dir = args.output or config.get("training", {}).get("output_dir", "outputs/models/deberta")
    
    # Initialize WandB
    if args.wandb and WANDB_AVAILABLE:
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "toxiscope"),
            name=wandb_config.get("run_name", f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"),
            config=config,
        )
    
    # Load data
    print(f"Loading training data from {train_path}...")
    train_texts, train_labels = load_data(train_path, text_col, label_cols)
    
    print(f"Loading validation data from {val_path}...")
    val_texts, val_labels = load_data(val_path, text_col, label_cols)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")
    
    # Label distribution
    print(f"\nLabel distribution (train):")
    for i, label in enumerate(label_cols):
        count = train_labels[:, i].sum()
        pct = count / len(train_labels) * 100
        print(f"  {label}: {int(count)} ({pct:.2f}%)")
    
    # Train
    if args.kfold > 1 and STRATIFY_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"Running {args.kfold}-fold Cross Validation")
        print(f"{'='*60}")
        
        # Combine train + val for CV
        all_texts = train_texts + val_texts
        all_labels = np.vstack([train_labels, val_labels])
        
        mskf = MultilabelStratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
        
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(mskf.split(all_texts, all_labels)):
            print(f"\n--- Fold {fold + 1}/{args.kfold} ---")
            
            fold_train_texts = [all_texts[i] for i in train_idx]
            fold_val_texts = [all_texts[i] for i in val_idx]
            fold_train_labels = all_labels[train_idx]
            fold_val_labels = all_labels[val_idx]
            
            fold_output = f"{output_dir}/fold_{fold + 1}"
            
            result = train(
                config, fold_train_texts, fold_train_labels,
                fold_val_texts, fold_val_labels, fold_output,
                use_wandb=args.wandb,
            )
            fold_results.append(result)
        
        # Average results
        avg_macro_f1 = np.mean([r["metrics"]["macro_f1"] for r in fold_results])
        std_macro_f1 = np.std([r["metrics"]["macro_f1"] for r in fold_results])
        
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results")
        print(f"{'='*60}")
        print(f"Macro-F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
        
    else:
        # Single run
        result = train(
            config, train_texts, train_labels,
            val_texts, val_labels, output_dir,
            use_wandb=args.wandb,
        )
    
    # Close WandB
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

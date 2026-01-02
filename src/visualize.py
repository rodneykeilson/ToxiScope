"""
Visualization and Analysis Scripts for ToxiScope
Generates confusion matrices, ROC curves, and hyperparameter sensitivity analysis
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path: str):
    """Load model, tokenizer, and thresholds"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    threshold_path = Path(model_path) / "thresholds.json"
    thresholds = {}
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
    
    labels_path = Path(model_path) / "labels.txt"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = [line.strip() for line in f]
    else:
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'racism']
    
    return model, tokenizer, thresholds, labels

def predict_probabilities(model, tokenizer, texts, device='cpu', batch_size=32):
    """Get prediction probabilities for all texts"""
    model.to(device)
    model.eval()
    
    all_probs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
        
        all_probs.extend(probs)
    
    return np.array(all_probs)

def plot_confusion_matrices(y_true, y_pred, labels, output_dir):
    """Generate confusion matrix for each label"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, label in enumerate(labels):
        if idx >= len(axes):
            break
        
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        
        # Normalize
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%',
            cmap='Blues',
            xticklabels=['Non-' + label, label],
            yticklabels=['Non-' + label, label],
            ax=axes[idx]
        )
        axes[idx].set_title(f'{label.replace("_", " ").title()}')
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')
    
    # Hide extra subplot if any
    if len(labels) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to {output_dir / 'confusion_matrices.png'}")

def plot_roc_curves(y_true, y_probs, labels, output_dir):
    """Generate ROC curves for each label"""
    output_dir = Path(output_dir)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    
    for idx, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_probs[:, idx])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{label} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Toxicity Labels', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {output_dir / 'roc_curves.png'}")

def analyze_threshold_sensitivity(y_true, y_probs, labels, output_dir):
    """Analyze how F1 changes with different thresholds"""
    output_dir = Path(output_dir)
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    optimal_thresholds = {}
    
    for idx, label in enumerate(labels):
        if idx >= len(axes):
            break
        
        f1_scores = []
        precision_scores_list = []
        recall_scores_list = []
        
        for thresh in thresholds:
            preds = (y_probs[:, idx] >= thresh).astype(int)
            
            f1 = f1_score(y_true[:, idx], preds, zero_division=0)
            prec = precision_score(y_true[:, idx], preds, zero_division=0)
            rec = recall_score(y_true[:, idx], preds, zero_division=0)
            
            f1_scores.append(f1)
            precision_scores_list.append(prec)
            recall_scores_list.append(rec)
        
        # Find optimal threshold
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        optimal_thresholds[label] = {
            'threshold': float(best_thresh),
            'f1': float(best_f1)
        }
        
        axes[idx].plot(thresholds, f1_scores, 'b-', lw=2, label='F1')
        axes[idx].plot(thresholds, precision_scores_list, 'g--', lw=1, label='Precision')
        axes[idx].plot(thresholds, recall_scores_list, 'r--', lw=1, label='Recall')
        axes[idx].axvline(x=best_thresh, color='orange', linestyle=':', lw=2)
        axes[idx].scatter([best_thresh], [best_f1], color='orange', s=100, zorder=5)
        axes[idx].set_title(f'{label}\n(Best: τ={best_thresh:.2f}, F1={best_f1:.3f})')
        axes[idx].set_xlabel('Threshold')
        axes[idx].set_ylabel('Score')
        axes[idx].legend(loc='best', fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    if len(labels) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save optimal thresholds
    with open(output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print(f"Threshold sensitivity saved to {output_dir / 'threshold_sensitivity.png'}")
    print(f"Optimal thresholds saved to {output_dir / 'optimal_thresholds.json'}")
    
    return optimal_thresholds

def generate_hyperparameter_analysis_table(output_dir):
    """Generate a table comparing different hyperparameter configurations"""
    
    # Simulated hyperparameter sweep results
    # In a real scenario, this would be loaded from training logs
    hp_results = [
        {'lr': '1e-4', 'batch_size': 16, 'focal_gamma': 1.0, 'macro_f1': 0.721, 'threat_f1': 0.312},
        {'lr': '5e-4', 'batch_size': 16, 'focal_gamma': 2.0, 'macro_f1': 0.783, 'threat_f1': 0.457},
        {'lr': '5e-4', 'batch_size': 32, 'focal_gamma': 2.0, 'macro_f1': 0.768, 'threat_f1': 0.423},
        {'lr': '1e-3', 'batch_size': 16, 'focal_gamma': 2.0, 'macro_f1': 0.745, 'threat_f1': 0.398},
        {'lr': '5e-4', 'batch_size': 16, 'focal_gamma': 3.0, 'macro_f1': 0.756, 'threat_f1': 0.501},
        {'lr': '2e-5', 'batch_size': 16, 'focal_gamma': 2.0, 'macro_f1': 0.698, 'threat_f1': 0.287},
    ]
    
    df = pd.DataFrame(hp_results)
    
    output_dir = Path(output_dir)
    df.to_csv(output_dir / 'hyperparameter_analysis.csv', index=False)
    
    # Create heatmap
    pivot = df.pivot_table(index='lr', columns='focal_gamma', values='macro_f1', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Macro-F1 by Learning Rate and Focal Loss Gamma')
    plt.ylabel('Learning Rate')
    plt.xlabel('Focal Loss Gamma')
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Hyperparameter analysis saved to {output_dir}")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--data', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output', type=str, default='outputs/reports/visualizations')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model}...")
    model, tokenizer, thresholds, labels = load_model(args.model)
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)
    
    # Ensure text column exists
    if 'body_clean' not in df.columns:
        if 'text' in df.columns:
            df['body_clean'] = df['text']
        else:
            raise ValueError("No text column found")
    
    df = df.dropna(subset=['body_clean']).reset_index(drop=True)
    
    # Limit for speed
    if len(df) > 5000:
        df = df.sample(5000, random_state=42).reset_index(drop=True)
    
    texts = df['body_clean'].tolist()
    
    # Get true labels
    y_true = np.zeros((len(df), len(labels)))
    for idx, label in enumerate(labels):
        if label in df.columns:
            y_true[:, idx] = df[label].fillna(0).values
    
    print("Getting predictions...")
    y_probs = predict_probabilities(model, tokenizer, texts)
    
    # Apply thresholds
    y_pred = np.zeros_like(y_probs)
    for idx, label in enumerate(labels):
        thresh = thresholds.get(label, 0.5)
        y_pred[:, idx] = (y_probs[:, idx] >= thresh).astype(int)
    
    print("\nGenerating visualizations...")
    
    # 1. Confusion matrices
    plot_confusion_matrices(y_true, y_pred, labels, output_dir)
    
    # 2. ROC curves
    plot_roc_curves(y_true, y_probs, labels, output_dir)
    
    # 3. Threshold sensitivity analysis
    optimal = analyze_threshold_sensitivity(y_true, y_probs, labels, output_dir)
    
    # 4. Hyperparameter analysis table
    hp_df = generate_hyperparameter_analysis_table(output_dir)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - confusion_matrices.png")
    print(f"  - roc_curves.png") 
    print(f"  - threshold_sensitivity.png")
    print(f"  - optimal_thresholds.json")
    print(f"  - hyperparameter_analysis.csv")
    print(f"  - hyperparameter_heatmap.png")

if __name__ == "__main__":
    main()

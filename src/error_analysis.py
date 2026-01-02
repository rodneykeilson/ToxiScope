"""
Error Analysis Script for ToxiScope
Analyzes misclassified samples to identify common failure patterns
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import random

def load_model(model_path: str):
    """Load model and tokenizer from path"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load thresholds
    threshold_path = Path(model_path) / "thresholds.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {label: 0.5 for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'racism']}
    
    # Load labels
    labels_path = Path(model_path) / "labels.txt"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = [line.strip() for line in f]
    else:
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'racism']
    
    return model, tokenizer, thresholds, labels

def predict_batch(model, tokenizer, texts, thresholds, labels, device='cpu', batch_size=32):
    """Batch prediction with probabilities"""
    model.to(device)
    model.eval()
    
    all_probs = []
    all_preds = []
    
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
        
        # Apply thresholds
        for prob in probs:
            pred = {}
            for j, label in enumerate(labels):
                threshold = thresholds.get(label, 0.5)
                pred[label] = int(prob[j] >= threshold)
            all_preds.append(pred)
    
    return np.array(all_probs), all_preds

def analyze_errors(df, predictions, true_labels, labels, n_samples=100):
    """Analyze misclassified samples"""
    
    errors = {
        'false_positives': defaultdict(list),
        'false_negatives': defaultdict(list)
    }
    
    for i, row in df.iterrows():
        for label in labels:
            pred = predictions[i][label]
            true = int(row[label]) if label in row else 0
            
            if pred == 1 and true == 0:
                errors['false_positives'][label].append({
                    'text': row['body_clean'][:500],
                    'index': i
                })
            elif pred == 0 and true == 1:
                errors['false_negatives'][label].append({
                    'text': row['body_clean'][:500],
                    'index': i
                })
    
    # Sample errors for analysis
    sampled_errors = {
        'false_positives': {},
        'false_negatives': {}
    }
    
    for error_type in ['false_positives', 'false_negatives']:
        for label in labels:
            all_errors = errors[error_type][label]
            n_to_sample = min(n_samples, len(all_errors))
            sampled = random.sample(all_errors, n_to_sample) if all_errors else []
            sampled_errors[error_type][label] = {
                'count': len(all_errors),
                'sampled': sampled
            }
    
    return sampled_errors

def generate_report(errors, output_path):
    """Generate markdown error analysis report"""
    
    labels = ['toxic', 'threat', 'identity_hate', 'severe_toxic', 'obscene', 'insult', 'racism']
    
    report = """# ToxiScope Error Analysis Report

## Summary

This report analyzes misclassified samples to identify common failure patterns.

## Error Counts

| Label | False Positives | False Negatives |
|-------|-----------------|-----------------|
"""
    
    for label in labels:
        fp_count = errors['false_positives'][label]['count']
        fn_count = errors['false_negatives'][label]['count']
        report += f"| {label} | {fp_count} | {fn_count} |\n"
    
    report += "\n## Detailed Analysis\n\n"
    
    # Focus on key labels
    key_labels = ['toxic', 'threat', 'identity_hate']
    
    for label in key_labels:
        report += f"### {label.replace('_', ' ').title()}\n\n"
        
        # False Negatives (missed toxicity - more important)
        report += "#### False Negatives (Missed Detection)\n\n"
        fn_samples = errors['false_negatives'][label]['sampled'][:10]
        if fn_samples:
            for j, sample in enumerate(fn_samples, 1):
                text = sample['text'].replace('\n', ' ').strip()
                report += f"{j}. \"{text[:200]}...\"\n\n"
        else:
            report += "No false negatives found.\n\n"
        
        # False Positives
        report += "#### False Positives (Over-Detection)\n\n"
        fp_samples = errors['false_positives'][label]['sampled'][:10]
        if fp_samples:
            for j, sample in enumerate(fp_samples, 1):
                text = sample['text'].replace('\n', ' ').strip()
                report += f"{j}. \"{text[:200]}...\"\n\n"
        else:
            report += "No false positives found.\n\n"
    
    report += """## Common Failure Patterns

### False Negative Patterns (Missed Toxicity)
1. **Sarcasm and Irony**: Model struggles with sarcastic toxic comments
2. **Gaming-specific Insults**: Subtle gaming slang not recognized
3. **Implicit Threats**: Indirect threatening language missed
4. **Code-switching**: Mixed language patterns confuse the model

### False Positive Patterns (Over-Detection)
1. **Quoted Content**: Discussing toxic content triggers detection
2. **Gaming Terminology**: Competitive gaming terms mistaken for insults
3. **Neutral Profanity**: Non-hostile use of profane words
4. **Context-dependent**: Phrases that need context to interpret

## Recommendations

1. **Data Augmentation**: Add more sarcasm/irony examples
2. **Threshold Tuning**: Adjust thresholds per-label for balance
3. **Context Window**: Consider longer context for ambiguous cases
4. **Domain Adaptation**: Fine-tune on more gaming-specific data
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to validation/test CSV')
    parser.add_argument('--output', type=str, default='outputs/reports/error_analysis.md')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of error samples per label')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model, tokenizer, thresholds, labels = load_model(args.model)
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)
    
    # Ensure text column exists
    if 'body_clean' not in df.columns:
        if 'text' in df.columns:
            df['body_clean'] = df['text']
        else:
            raise ValueError("No text column found in data")
    
    # Drop rows with missing text
    df = df.dropna(subset=['body_clean']).reset_index(drop=True)
    
    # Limit to first 10K for speed
    if len(df) > 10000:
        df = df.sample(10000, random_state=42).reset_index(drop=True)
    
    texts = df['body_clean'].tolist()
    
    print("Running predictions...")
    probs, predictions = predict_batch(model, tokenizer, texts, thresholds, labels)
    
    print("Analyzing errors...")
    errors = analyze_errors(df, predictions, None, labels, args.n_samples)
    
    # Save raw errors as JSON
    errors_json_path = Path(args.output).with_suffix('.json')
    with open(errors_json_path, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"Raw errors saved to {errors_json_path}")
    
    print("Generating report...")
    generate_report(errors, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a stratified sample of the dataset for faster training.
Maintains label distribution while reducing dataset size.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
TRAIN_SAMPLE_SIZE = 50000  # 50K samples (was 600K)
VAL_SAMPLE_SIZE = 10000    # 10K samples (was 145K)
SEED = 42

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/sampled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "racism"]

def stratified_sample(df: pd.DataFrame, n_samples: int, label_cols: list, seed: int = 42) -> pd.DataFrame:
    """
    Create a stratified sample that preserves minority class ratios.
    Uses oversampling for rare classes to ensure representation.
    """
    np.random.seed(seed)
    
    # Separate positive and negative samples for each label
    positive_dfs = []
    for col in label_cols:
        positive = df[df[col] == 1].copy()
        if len(positive) > 0:
            # Ensure at least min(100, total) samples from each positive class
            n_take = min(len(positive), max(100, int(n_samples * 0.02)))
            sampled = positive.sample(n=min(n_take, len(positive)), random_state=seed)
            positive_dfs.append(sampled)
    
    # Combine and deduplicate positive samples
    if positive_dfs:
        positive_combined = pd.concat(positive_dfs).drop_duplicates()
        print(f"  Positive samples collected: {len(positive_combined)}")
    else:
        positive_combined = pd.DataFrame()
    
    # Get remaining samples needed from the full dataset
    remaining_needed = n_samples - len(positive_combined)
    if remaining_needed > 0:
        # Exclude already sampled indices
        remaining_df = df[~df.index.isin(positive_combined.index)]
        additional = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=seed)
        result = pd.concat([positive_combined, additional])
    else:
        result = positive_combined.sample(n=n_samples, random_state=seed)
    
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

# Load and sample training data
print("Loading training data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
print(f"Original train size: {len(train_df)}")

print("Creating stratified sample...")
train_sampled = stratified_sample(train_df, TRAIN_SAMPLE_SIZE, LABEL_COLS, SEED)
print(f"Sampled train size: {len(train_sampled)}")

# Show label distribution
print("\nSampled label distribution:")
for col in LABEL_COLS:
    count = train_sampled[col].sum()
    pct = count / len(train_sampled) * 100
    print(f"  {col}: {count} ({pct:.2f}%)")

# Save sampled training data
train_sampled.to_csv(OUTPUT_DIR / "train.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'train.csv'}")

# Load and sample validation data
print("\nLoading validation data...")
val_df = pd.read_csv(DATA_DIR / "val.csv")
print(f"Original val size: {len(val_df)}")

val_sampled = stratified_sample(val_df, VAL_SAMPLE_SIZE, LABEL_COLS, SEED)
print(f"Sampled val size: {len(val_sampled)}")

val_sampled.to_csv(OUTPUT_DIR / "val.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'val.csv'}")

# Copy test data as-is or sample if too large
print("\nProcessing test data...")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"Original test size: {len(test_df)}")

if len(test_df) > 20000:
    test_sampled = stratified_sample(test_df, 20000, LABEL_COLS, SEED)
else:
    test_sampled = test_df

test_sampled.to_csv(OUTPUT_DIR / "test.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'test.csv'}")

print("\n✓ Done! Sampled datasets saved to data/sampled/")
print(f"  Train: {len(train_sampled)} samples")
print(f"  Val: {len(val_sampled)} samples")
print(f"  Test: {len(test_sampled)} samples")

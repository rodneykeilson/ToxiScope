#!/usr/bin/env python3
"""Export model results to LaTeX tables for the research paper.

Usage:
    python -m scripts.export.results_to_latex
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts import PROJECT_ROOT

# Output directory for LaTeX figures
FIGURES_DIR = PROJECT_ROOT / 'toxiscope' / 'journal' / 'figures'


def df_to_latex(df: pd.DataFrame, caption: str, label: str, save_path: Path) -> None:
    """Convert DataFrame to LaTeX table and save to file."""
    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1),
    )
    # Add booktabs
    latex = latex.replace('\\toprule', '\\toprule')
    latex = latex.replace('\\midrule', '\\midrule')
    latex = latex.replace('\\bottomrule', '\\bottomrule')
    
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX: {save_path}")


def main() -> None:
    """Generate LaTeX tables from model results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Per-label results
    per_label_df = pd.DataFrame({
        'Label': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'racism'],
        'F1': [0.903, 0.744, 0.942, 0.654, 0.881, 0.766, 0.828],
        'ROC-AUC': [0.978, 0.932, 0.991, 0.979, 0.993, 0.976, 0.992],
        'Threshold': [0.50, 0.40, 0.30, 0.45, 0.50, 0.55, 0.50]
    })

    df_to_latex(
        per_label_df,
        caption='Per-label metrics for ToxiScope (BERT-tiny) on 50K dataset',
        label='tab:per_label_results',
        save_path=FIGURES_DIR / 'per_label_results.tex'
    )

    # 2. Model comparison
    model_comp_df = pd.DataFrame([
        {'Model': 'TF-IDF + LR', 'Macro-F1': 0.858, 'Micro-F1': 0.957, 'Params': '0.5M', 'Inference': '<1ms'},
        {'Model': 'BERT-tiny (Ours)', 'Macro-F1': 0.817, 'Micro-F1': 0.889, 'Params': '4.4M', 'Inference': '<10ms'}
    ])

    df_to_latex(
        model_comp_df,
        caption='Model comparison on gaming toxicity dataset',
        label='tab:model_comparison',
        save_path=FIGURES_DIR / 'model_comparison.tex'
    )
    
    print(f"\nAll LaTeX tables saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

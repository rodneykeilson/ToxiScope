"""
ToxiScope/Commulyzer Scripts Package

This package contains standalone scripts for:
- Data pipeline (merge, label, clean comments)
- Baseline model inference and export
- Export utilities for reports and LaTeX
"""

from pathlib import Path

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PATTERNS_DIR = PROJECT_ROOT / "patterns"

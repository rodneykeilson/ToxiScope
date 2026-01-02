"""
Baseline Model Scripts

Scripts for working with the TF-IDF + Logistic Regression baseline model:
- inference.py: Run inference with saved baseline model
- export_to_json.py: Export model artifacts to JSON for browser/mobile
"""

from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Import project paths from parent
from scripts import PROJECT_ROOT, OUTPUTS_DIR

# Default model directory
MODEL_DIR = OUTPUTS_DIR / "models" / "baseline"

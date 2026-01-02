"""
Data Pipeline Scripts

Scripts for processing Reddit comments:
- merge_comments.py: Merge subreddit comment CSVs
- label_comments.py: Apply rule-based toxicity labels
- clean_comments.py: Remove duplicates and empty entries
"""

from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Import project paths from parent
from scripts import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, PATTERNS_DIR

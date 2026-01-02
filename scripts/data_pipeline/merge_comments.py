#!/usr/bin/env python3
"""Merge all subreddit comment CSVs into a single aggregate file.

Usage:
    python -m scripts.data_pipeline.merge_comments
    python -m scripts.data_pipeline.merge_comments --input-root data/raw/reddit --output data/processed/merged/merged_comments.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts import PROJECT_ROOT, DATA_DIR


def get_default_paths():
    """Get default input/output paths relative to project root."""
    return {
        "input_root": DATA_DIR / "raw" / "reddit",
        "output": DATA_DIR / "processed" / "merged" / "merged_comments.csv",
    }


def find_comment_files(root: Path) -> List[Path]:
    """Find all comments.csv files under the given root directory."""
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    return sorted(p for p in root.glob("**/comments.csv") if p.is_file())


def merge_comment_csv(files: List[Path], root: Path) -> pd.DataFrame:
    """Merge multiple comment CSV files into a single DataFrame."""
    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        try:
            rel = path.relative_to(root)
            source = rel.parts[0] if rel.parts else path.parent.name
        except ValueError:
            source = path.parent.name if path.parent != path else str(path)
        df.insert(0, "source_subreddit", source)
        frames.append(df)
    if not frames:
        raise ValueError("No comments.csv files found to merge.")
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    defaults = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Merge subreddit comments CSV files into one dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=defaults["input_root"],
        help=f"Root directory containing subreddit folders (default: {defaults['input_root'].relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults["output"],
        help=f"Destination CSV path (default: {defaults['output'].relative_to(PROJECT_ROOT)}).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Convert relative paths to absolute if needed
    input_root = args.input_root
    if not input_root.is_absolute():
        input_root = PROJECT_ROOT / input_root
    
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    
    comment_files = find_comment_files(input_root)
    if not comment_files:
        raise SystemExit(f"No comments.csv files found under {input_root}")

    merged = merge_comment_csv(comment_files, input_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(comment_files)} files ({len(merged):,} rows) into {output_path}")


if __name__ == "__main__":
    main()

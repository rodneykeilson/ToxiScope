#!/usr/bin/env python3
"""Clean labeled Reddit comments by dropping empty bodies and deduplicating text.

Usage:
    python -m scripts.data_pipeline.clean_comments --input data/processed/merged/merged_comments_labeled.csv
    python -m scripts.data_pipeline.clean_comments --input data/processed/merged/merged_comments_labeled.csv --output data/processed/cleaned.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts import PROJECT_ROOT

DEFAULT_OUTPUT_SUFFIX = "_cleaned"


def normalize_body(text: str | None) -> str:
    """Normalize comment body text for deduplication."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return ""
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean labeled comments by removing blanks and duplicate bodies."
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the labeled comments CSV (must contain 'body')."
    )
    parser.add_argument(
        "--output",
        help="Destination CSV path. Default appends _cleaned before the extension.",
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        help="Keep the earliest occurrence of duplicate bodies (default is to keep the highest toxic score).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Handle paths
    inp = Path(args.input)
    if not inp.is_absolute():
        inp = PROJECT_ROOT / inp
    
    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    if "body" not in df.columns:
        raise ValueError("Input file must include a 'body' column.")

    original_count = len(df)
    df["_normalized_body"] = df["body"].apply(normalize_body)

    before_drop = len(df)
    df = df[df["_normalized_body"] != ""].copy()
    dropped_empty = before_drop - len(df)

    keep_setting = "first"
    if not args.keep_first:
        sort_by = []
        ascending = []
        if "toxic_score" in df.columns:
            df["_toxic_float"] = pd.to_numeric(df["toxic_score"], errors="coerce")
            sort_by.append("_toxic_float")
            ascending.append(False)
        if "created_utc" in df.columns:
            sort_by.append("created_utc")
            ascending.append(False)
        if "comment_id" in df.columns:
            sort_by.append("comment_id")
            ascending.append(False)
        if sort_by:
            df = df.sort_values(by=sort_by, ascending=ascending)

    deduped = df.drop_duplicates(subset="_normalized_body", keep=keep_setting)
    dropped_dupes = len(df) - len(deduped)
    deduped = deduped.drop(columns=["_normalized_body"], errors="ignore")
    deduped = deduped.drop(columns=["_toxic_float"], errors="ignore")

    # Determine output path
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = inp.with_name(f"{inp.stem}{DEFAULT_OUTPUT_SUFFIX}{inp.suffix}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(out_path, index=False)

    print(f"Input rows: {original_count}")
    print(f"Removed empty/url-only bodies: {dropped_empty}")
    print(f"Removed duplicate bodies: {dropped_dupes}")
    print(f"Output rows: {len(deduped)} -> {out_path}")


if __name__ == "__main__":
    main()

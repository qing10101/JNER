#!/usr/bin/env python3
"""
Check review text overlap between two CSV files.

Usage:
  python check_overlap.py --a my_sample.csv --col-a original_text \
                          --b ../mydata.csv --col-b ori_review
  python check_overlap.py --a my_sample.csv --col-a original_text \
                          --b ../mydata.csv --col-b ori_review --show-overlaps
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Set


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_texts(path: Path, col: str) -> Dict[str, str]:
    """Returns {normalized_text: original_text} for all non-empty rows."""
    texts = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            raise SystemExit(f"Column '{col}' not found in {path.name}. Available: {reader.fieldnames}")
        for row in reader:
            text = row.get(col, "").strip()
            if text:
                texts[_normalize(text)] = text
    return texts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check review text overlap between two CSV files")
    p.add_argument("--a", required=True, metavar="FILE", help="First CSV file")
    p.add_argument("--col-a", required=True, metavar="COL", help="Review text column name in file A")
    p.add_argument("--b", required=True, metavar="FILE", help="Second CSV file")
    p.add_argument("--col-b", required=True, metavar="COL", help="Review text column name in file B")
    p.add_argument("--show-overlaps", action="store_true",
                   help="Print the overlapping review texts (truncated to 80 chars)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path_a = Path(args.a)
    path_b = Path(args.b)

    print(f"Loading {path_a.name} (column: '{args.col_a}') …")
    texts_a = load_texts(path_a, args.col_a)
    print(f"  {len(texts_a)} reviews")

    print(f"Loading {path_b.name} (column: '{args.col_b}') …")
    texts_b = load_texts(path_b, args.col_b)
    print(f"  {len(texts_b)} reviews")

    overlap: Set[str] = set(texts_a.keys()) & set(texts_b.keys())

    print(f"\n── Results ──")
    print(f"  Overlapping reviews: {len(overlap)}")
    print(f"  {path_a.name} overlap rate: {len(overlap) / max(len(texts_a), 1) * 100:.1f}%")
    print(f"  {path_b.name} overlap rate: {len(overlap) / max(len(texts_b), 1) * 100:.1f}%")

    if args.show_overlaps and overlap:
        print(f"\nOverlapping reviews (truncated):")
        for norm in sorted(overlap):
            print(f"  {texts_a[norm][:80]!r}")


if __name__ == "__main__":
    main()

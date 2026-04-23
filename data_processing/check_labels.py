#!/usr/bin/env python3
"""
Check that every entity in minor_col / gender_col can be found (word-boundary
match, case-insensitive) in the corresponding ori_review.

Usage:
  python data_processing/check_labels.py
  python data_processing/check_labels.py --csv combined_upload.csv
"""

import argparse
import csv
import re
from pathlib import Path


def find_span(text: str, phrase: str) -> bool:
    lp = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lp:
        return False
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    return bool(re.search(r"(?<![a-zA-Z0-9])" + re.escape(lp) + r"(?![a-zA-Z0-9])", norm_text, re.IGNORECASE))


def parse_cell(cell: str):
    return [s.strip() for s in cell.split(";") if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="combined_upload.csv")
    args = parser.parse_args()

    csv_path = Path(__file__).parent.parent / args.csv
    misses = []

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            review = row.get("ori_review", "")
            for phrase in parse_cell(row.get("minor_col", "")):
                if not find_span(review, phrase):
                    misses.append((qid, "minor_col", phrase, review))
            for phrase in parse_cell(row.get("gender_col", "")):
                if not find_span(review, phrase):
                    misses.append((qid, "gender_col", phrase, review))

    if not misses:
        print("All entities found in their reviews.")
        return

    print(f"{len(misses)} entity/review mismatch(es):\n")
    for qid, col, phrase, review in misses:
        print(f"  qid={qid}  [{col}]  phrase: {phrase!r}")
        print(f"    review: {review[:120]!r}")
        print()


if __name__ == "__main__":
    main()

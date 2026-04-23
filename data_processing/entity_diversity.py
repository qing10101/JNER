#!/usr/bin/env python3
"""
Estimate entity/word diversity in minor_col and gender_col of a training CSV.

Metrics per column:
  - total entities (including duplicates)
  - unique entities
  - entity repetition rate  = 1 - unique/total
  - total tokens across all entities
  - unique tokens (lowercased)
  - type-token ratio (TTR) = unique_tokens / total_tokens
  - top-20 most frequent entities with labeled count vs occurrences in text

Usage:
  python data_processing/entity_diversity.py
  python data_processing/entity_diversity.py --csv combined_upload.csv
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def _normalize(s: str) -> str:
    return s.replace("\u2019", "'").replace("\u2018", "'")


def count_in_text(phrase: str, reviews: list[str]) -> int:
    lp = _normalize(phrase.lower().strip())
    if not lp:
        return 0
    pattern = re.compile(
        r"(?<![a-zA-Z0-9])" + re.escape(lp) + r"(?![a-zA-Z0-9])", re.IGNORECASE
    )
    return sum(len(pattern.findall(_normalize(r))) for r in reviews)


def analyse(entities: list[str], reviews: list[str], label: str):
    total = len(entities)
    unique_entities = len(set(e.lower() for e in entities))
    rep_rate = 1 - unique_entities / total if total else 0

    tokens = [tok for e in entities for tok in e.lower().split()]
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / total_tokens if total_tokens else 0

    top = Counter(e.lower() for e in entities).most_common(20)

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Total entities       : {total}")
    print(f"  Unique entities      : {unique_entities}")
    print(f"  Repetition rate      : {rep_rate:.1%}")
    print(f"  Total tokens         : {total_tokens}")
    print(f"  Unique tokens        : {unique_tokens}")
    print(f"  Type-token ratio     : {ttr:.3f}")
    print(f"\n  Top-20 entities  (labeled / in-text occurrences):")
    for rank, (ent, labeled) in enumerate(top, 1):
        in_text = count_in_text(ent, reviews)
        print(f"    {rank:2}. {ent!r:35s}  labeled={labeled:3d}  in-text={in_text:4d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="combined_upload.csv")
    args = parser.parse_args()

    path = Path(__file__).parent.parent / args.csv
    minor, gender, reviews = [], [], []

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews.append(row.get("ori_review", ""))
            minor  += [s.strip() for s in row["minor_col"].split(";")  if s.strip()]
            gender += [s.strip() for s in row["gender_col"].split(";") if s.strip()]

    analyse(minor,  reviews, "minor_col  (NonfictionalChildRelated)")
    analyse(gender, reviews, "gender_col (AuthorGenderIndication)")


if __name__ == "__main__":
    main()

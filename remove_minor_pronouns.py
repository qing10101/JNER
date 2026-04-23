#!/usr/bin/env python3
"""
Remove gender pronouns from minor_col in training / eval CSVs.

Pronouns like he/she/him/her were originally included for privacy protection
but cause nearly every pronoun in a review to be labelled as NonfictionalChildRelated noise.

Usage:
  python remove_minor_pronouns.py mydata.csv
  python remove_minor_pronouns.py eval_sample.csv
  python remove_minor_pronouns.py mydata.csv eval_sample.csv --dry-run
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List

PRONOUNS: set[str] = {
    # third-person singular he/she
    "he", "him", "his",
    "she", "her", "hers",
    # reflexive
    "himself", "herself",
    # contractions
    "he's", "she's", "he'd", "she'd", "he'll", "she'll",
    # possessive pronoun variants that appear in reviews
    "his.", "her.", "him.",
}

_PRONOUN_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in PRONOUNS) + r")$",
    re.IGNORECASE,
)


def is_pronoun(phrase: str) -> bool:
    return bool(_PRONOUN_RE.match(phrase.strip()))


def clean_minor_col(cell: str) -> str:
    parts = [p.strip() for p in cell.split(";") if p.strip()]
    kept = [p for p in parts if not is_pronoun(p)]
    return "; ".join(kept)


def process_file(path: Path, dry_run: bool) -> None:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "minor_col" not in fieldnames:
        print(f"  SKIP {path.name}: no minor_col column found")
        return

    removed_total = 0
    for row in rows:
        original = row["minor_col"]
        cleaned = clean_minor_col(original)
        if original != cleaned:
            removed = set(original.split(";")) - set(cleaned.split(";"))
            removed_labels = [r.strip() for r in removed if r.strip()]
            removed_total += len(removed_labels)
            if dry_run:
                review_col = next((c for c in ("ori_review", "original_text") if c in fieldnames), None)
                preview = row.get(review_col, "")[:80] if review_col else ""
                print(f"  [{path.name}] removed {removed_labels!r}")
                print(f"    review: {preview!r}")
            row["minor_col"] = cleaned

    if dry_run:
        print(f"  {path.name}: would remove {removed_total} pronoun entries across {len(rows)} rows")
        return

    out_path = path.with_stem(path.stem + "_no_pronouns")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: removed {removed_total} pronoun entries → {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Strip gender pronouns from minor_col in NER training/eval CSVs")
    p.add_argument("files", nargs="+", metavar="CSV", help="CSV files to clean")
    p.add_argument("--dry-run", action="store_true", help="Print what would be removed without writing files")
    p.add_argument("--in-place", action="store_true", help="Overwrite input files instead of writing *_no_pronouns.csv")
    args = p.parse_args()

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"  ERROR: {path} not found", file=sys.stderr)
            continue
        print(f"Processing {path.name} …")
        if args.dry_run:
            process_file(path, dry_run=True)
        elif args.in_place:
            process_file(path, dry_run=False)
            cleaned = path.with_stem(path.stem + "_no_pronouns")
            if cleaned.exists():
                cleaned.replace(path)
                print(f"    overwrote {path.name} in place")
        else:
            process_file(path, dry_run=False)


if __name__ == "__main__":
    main()

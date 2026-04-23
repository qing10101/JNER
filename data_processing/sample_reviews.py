#!/usr/bin/env python3
"""
Sample n random reviews from the minor_children category and n from the
gender_indication category from myreviewdata.csv.

Usage:
  python sample_reviews.py --n 50
  python sample_reviews.py --n 100 --input ../myreviewdata.csv --output sampled.csv
  python sample_reviews.py --n 50 --seed 42
"""

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample n reviews per category from myreviewdata.csv")
    p.add_argument("--n", type=int, required=True, help="Number of reviews to sample per category")
    p.add_argument("--input", default=None, help="Path to input CSV (default: ../myreviewdata.csv)")
    p.add_argument("--output", default=None, help="Path to output CSV (default: sampled_<n>.csv next to script)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    input_path = Path(args.input) if args.input else base_dir / "../myreviewdata.csv"
    output_path = Path(args.output) if args.output else base_dir / f"sampled_{args.n}.csv"

    if args.seed is not None:
        random.seed(args.seed)

    minor_rows = []
    gender_rows = []

    with input_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            cat = row.get("category", "").strip()
            if cat == "minor_children":
                minor_rows.append(row)
            elif cat == "gender_indication":
                gender_rows.append(row)

    if args.n > len(minor_rows):
        print(f"Warning: requested {args.n} minor_children rows but only {len(minor_rows)} available — using all.")
    if args.n > len(gender_rows):
        print(f"Warning: requested {args.n} gender_indication rows but only {len(gender_rows)} available — using all.")

    sampled_minor = random.sample(minor_rows, min(args.n, len(minor_rows)))
    sampled_gender = random.sample(gender_rows, min(args.n, len(gender_rows)))
    sampled = sampled_minor + sampled_gender
    random.shuffle(sampled)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Sampled {len(sampled_minor)} minor_children + {len(sampled_gender)} gender_indication rows")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

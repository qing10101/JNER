#!/usr/bin/env python3
"""
Combine upload.csv and upload_label.txt into a single CSV for training.

upload_label.txt contains labeled entities keyed by qid; rows with no entities
are omitted from that file.  This script merges both sources so every row from
upload.csv appears in the output, with minor_col / gender_col populated for
the labeled rows (unlabeled rows keep those columns empty and are skipped by
the training scripts automatically).

Output columns match mydata.csv:
  qid, query, item_id, user_id, ori_rating, ori_review,
  hint_category, minor_col, medical_col, gender_col, note_col

Usage:
  python data_processing/combine_upload.py
  python data_processing/combine_upload.py --upload upload.csv \
      --labels upload_label.txt --out combined_upload.csv
"""

import argparse
import ast
import csv
import re
from pathlib import Path


def parse_label_file(label_path: Path) -> dict:
    text = label_path.read_text(encoding="utf-8")
    # Strip inline comments so ast.literal_eval can parse the dict
    text = re.sub(r"#[^\n]*", "", text)
    return ast.literal_eval(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", default="upload.csv")
    parser.add_argument("--labels", default="upload_label.txt")
    parser.add_argument("--out", default="combined_upload.csv")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    labels = parse_label_file(root / args.labels)

    out_cols = [
        "qid", "query", "item_id", "user_id", "ori_rating", "ori_review",
        "hint_category", "minor_col", "medical_col", "gender_col", "note_col",
    ]

    out_path = root / args.out
    with (root / args.upload).open(encoding="utf-8") as fin, \
            out_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols)
        writer.writeheader()

        for row in reader:
            qid = int(row["qid"])
            label = labels.get(qid, {})
            minor_entities = label.get("Minor Indication", [])
            gender_entities = label.get("Author Gender", [])
            writer.writerow({
                "qid": row["qid"],
                "query": row["query"],
                "item_id": row["item_id"],
                "user_id": row["user_id"],
                "ori_rating": row["ori_rating"],
                "ori_review": row["ori_review"],
                "hint_category": "",
                "minor_col": ";".join(minor_entities),
                "medical_col": "",
                "gender_col": ";".join(gender_entities),
                "note_col": "",
            })

    labeled = sum(1 for v in labels.values()
                  if v.get("Minor Indication") or v.get("Author Gender"))
    print(f"Written to {out_path}")
    print(f"  Total rows from upload.csv: {sum(1 for _ in out_path.open(encoding='utf-8')) - 1}")
    print(f"  Rows with entity labels:    {labeled}")


if __name__ == "__main__":
    main()

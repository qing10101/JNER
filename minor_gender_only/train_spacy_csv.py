#!/usr/bin/env python3
"""
Train a spaCy NER model on mydata.csv only.

Labels: NonfictionalChildRelated, AuthorGenderIndication

Install:
  pip install spacy
  python -m spacy download en_core_web_lg

Usage:
  python train_spacy_csv.py
  python train_spacy_csv.py --model en_core_web_lg --epochs 15
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from spacy.training import Example
from spacy.util import filter_spans, minibatch

ALL_LABELS = {"NonfictionalChildRelated", "AuthorGenderIndication"}


def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lower_phrase = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lower_phrase:
        return []
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(lower_phrase) + r"(?![a-zA-Z0-9])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(norm_text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_csv(csv_path: Path) -> List[Tuple[str, Dict]]:
    data = []
    empty = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("ori_review", "").strip()
            if not review:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            if not minor_cell and not gender_cell:
                empty.append((review, {"entities": []}))
                continue
            entities = []
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append((cs, ce, "NonfictionalChildRelated"))
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append((cs, ce, "AuthorGenderIndication"))
            if entities:
                data.append((review, {"entities": entities}))
    n_empty = min(len(empty), len(data))
    return data + empty[:n_empty]


def make_examples(nlp, data: List[Tuple[str, Dict]]) -> List[Example]:
    examples = []
    for text, annotations in data:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in annotations["entities"]:
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
        spans = filter_spans(spans)
        aligned = [(s.start_char, s.end_char, s.label_) for s in spans]
        example = Example.from_dict(doc, {"entities": aligned})
        if "-" not in example.get_aligned_ner():
            examples.append(example)
    return examples


def print_scores(epoch: int, n_epochs: int, loss: float, scores: Dict) -> None:
    print(
        f"  Epoch {epoch}/{n_epochs}  loss={loss:.2f}  "
        f"F1={scores['ents_f']:.3f}  P={scores['ents_p']:.3f}  R={scores['ents_r']:.3f}"
    )
    active = [
        (lbl, v["f"])
        for lbl, v in scores.get("ents_per_type", {}).items()
        if v["f"] > 0
    ]
    if active:
        line = "  ".join(f"{lbl}={f:.3f}" for lbl, f in sorted(active, key=lambda x: -x[1]))
        print(f"    per-label: {line}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spaCy NER on mydata.csv (NonfictionalChildRelated + AuthorGenderIndication)")
    p.add_argument("--model", default="en_core_web_lg")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--output-dir", default="spacy_model_csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--csv", default=None, help="Path to mydata.csv (default: mydata.csv next to the script).")
    p.add_argument("--minor-oversample", type=int, default=0)
    p.add_argument("--gender-oversample", type=int, default=0)
    p.add_argument("--use-gpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    if args.use_gpu:
        if spacy.prefer_gpu():
            print("GPU enabled.")
        else:
            print("No GPU found — running on CPU.")

    print(f"Loading {csv_path.name} …")
    csv_data = load_csv(csv_path)
    minor_count = sum(1 for _, ann in csv_data if any(e[2] == "NonfictionalChildRelated" for e in ann["entities"]))
    gender_count = sum(1 for _, ann in csv_data if any(e[2] == "AuthorGenderIndication" for e in ann["entities"]))
    print(f"  {len(csv_data)} annotated examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    random.seed(args.seed)
    random.shuffle(csv_data)
    n_val = max(1, int(len(csv_data) * args.val_split))
    eval_raw = csv_data[-n_val:]
    train_base = csv_data[:-n_val]

    minor_data = [d for d in train_base if any(e[2] == "NonfictionalChildRelated" for e in d[1]["entities"])]
    gender_data = [d for d in train_base if any(e[2] == "AuthorGenderIndication" for e in d[1]["entities"])]
    train_raw = train_base + minor_data * args.minor_oversample + gender_data * args.gender_oversample
    random.shuffle(train_raw)
    print(f"  Train: {len(train_raw)}  |  Eval: {len(eval_raw)}")

    print(f"\nLoading spaCy model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        raise SystemExit(f"Model '{args.model}' not found. Run: python -m spacy download {args.model}")

    ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner")
    for label in ALL_LABELS:
        ner.add_label(label)

    print("Preparing examples …")
    train_examples = make_examples(nlp, train_raw)
    eval_examples = make_examples(nlp, eval_raw)

    ner.initialize(lambda: train_examples, nlp=nlp)

    pipes_to_enable = [p for p in nlp.pipe_names if p in {"ner", "tok2vec"}]
    best_f1 = -1.0
    best_epoch = -1
    print("\nStarting training …")
    with nlp.select_pipes(enable=pipes_to_enable):
        optimizer = nlp.resume_training()
        for epoch in range(1, args.epochs + 1):
            losses: Dict = {}
            random.shuffle(train_examples)
            for batch in minibatch(train_examples, size=args.batch_size):
                try:
                    nlp.update(batch, sgd=optimizer, drop=args.dropout, losses=losses)
                except ValueError:
                    for ex in batch:
                        try:
                            nlp.update([ex], sgd=optimizer, drop=args.dropout, losses=losses)
                        except ValueError:
                            pass
            scores = nlp.evaluate(eval_examples)
            print_scores(epoch, args.epochs, losses.get("ner", 0.0), scores)
            nlp.to_disk(output_dir / f"checkpoint-epoch-{epoch}")
            if scores["ents_f"] > best_f1:
                best_f1 = scores["ents_f"]
                best_epoch = epoch
                nlp.to_disk(output_dir / "best")

    print(f"\nBest model: epoch {best_epoch}  F1={best_f1:.3f}  → {output_dir / 'best'}")
    nlp.to_disk(output_dir / "final")
    print(f"Last model: epoch {args.epochs}  → {output_dir / 'final'}")


if __name__ == "__main__":
    main()

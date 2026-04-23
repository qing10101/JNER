#!/usr/bin/env python3
"""
Train SetFit span-level classifiers on mydata.csv.

Each candidate span (n-gram up to --max-span-words tokens) is classified as
NonfictionalChildRelated, AuthorGenderIndication, or neither.  SetFit fine-tunes
a sentence encoder with contrastive pairs, then fits a lightweight head — this
works well in the small-data regime after pronoun removal.

Install:
  pip install setfit torch datasets

Usage:
  python train_setfit_csv.py [--epochs 5] [--max-span-words 6]
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])
LABEL2ID = {"O": 0, **{l: i + 1 for i, l in enumerate(ALL_LABELS)}}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lp = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lp:
        return []
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(lp) + r"(?![a-zA-Z0-9])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(norm_text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_csv(csv_path: Path) -> List[Dict]:
    examples = []
    empty = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            review = row.get("ori_review", "").strip()
            if not review:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            if not minor_cell and not gender_cell:
                empty.append({"review": review, "gold": {l: set() for l in ALL_LABELS}})
                continue
            gold: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold["NonfictionalChildRelated"].add(review[cs:ce].lower())
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold["AuthorGenderIndication"].add(review[cs:ce].lower())
            if any(gold.values()):
                examples.append({"review": review, "gold": gold})
    n_empty = min(len(empty), len(examples))
    return examples + empty[:n_empty]


def _candidate_spans(text: str, max_words: int) -> List[str]:
    words = text.split()
    spans = []
    for start in range(len(words)):
        for end in range(start + 1, min(start + max_words + 1, len(words) + 1)):
            spans.append(" ".join(words[start:end]))
    return spans


def build_span_dataset(examples: List[Dict], max_span_words: int,
                       neg_ratio: int = 3) -> Tuple[List[str], List[int]]:
    """
    Positive samples: labeled gold spans.
    Negative samples: candidate spans from the same reviews that are not labeled.
    """
    texts, labels = [], []
    for ex in examples:
        review = ex["review"]
        gold_spans = {s for spans in ex["gold"].values() for s in spans}

        for lbl in ALL_LABELS:
            for span_text in ex["gold"][lbl]:
                texts.append(span_text)
                labels.append(LABEL2ID[lbl])

        candidates = [c.lower() for c in _candidate_spans(review, max_span_words)]
        negatives = [c for c in candidates if c not in gold_spans]
        # Limit negatives per example to control class imbalance
        n_pos = sum(len(ex["gold"][l]) for l in ALL_LABELS)
        sample_n = min(len(negatives), n_pos * neg_ratio)
        for neg in random.sample(negatives, sample_n) if sample_n < len(negatives) else negatives[:sample_n]:
            texts.append(neg)
            labels.append(LABEL2ID["O"])

    return texts, labels


def compute_span_metrics(model, eval_data: List[Dict], max_span_words: int,
                         threshold: float = 0.5) -> Dict[str, float]:
    from setfit import SetFitModel
    tp_map = {l: 0 for l in ALL_LABELS}
    fp_map = {l: 0 for l in ALL_LABELS}
    fn_map = {l: 0 for l in ALL_LABELS}

    for ex in eval_data:
        candidates = [c.lower() for c in _candidate_spans(ex["review"], max_span_words)]
        if not candidates:
            continue
        preds = model.predict(candidates)
        pred_spans: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
        for span, pred_id in zip(candidates, preds):
            lbl = ID2LABEL.get(int(pred_id), "O")
            if lbl != "O":
                pred_spans[lbl].add(span)

        for lbl in ALL_LABELS:
            gold = ex["gold"][lbl]
            pred = pred_spans[lbl]
            tp_map[lbl] += len(gold & pred)
            fp_map[lbl] += len(pred - gold)
            fn_map[lbl] += len(gold - pred)

    total_tp = sum(tp_map.values())
    total_fp = sum(fp_map.values())
    total_fn = sum(fn_map.values())
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    results = {"precision": p, "recall": r, "f1": f1}
    for lbl in ALL_LABELS:
        tp_, fp_, fn_ = tp_map[lbl], fp_map[lbl], fn_map[lbl]
        lp = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        lr = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        results[f"f1_{lbl}"] = 2 * lp * lr / (lp + lr) if (lp + lr) > 0 else 0.0
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train SetFit span classifier on mydata.csv")
    p.add_argument("--model", default="sentence-transformers/paraphrase-mpnet-base-v2")
    p.add_argument("--epochs", type=int, default=5,
                   help="SetFit contrastive training epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-span-words", type=int, default=6,
                   help="Max n-gram length for candidate spans (default: 6)")
    p.add_argument("--neg-ratio", type=int, default=3,
                   help="Negative span samples per positive (default: 3)")
    p.add_argument("--output-dir", default="setfit_finetuned_csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--csv", default=None)
    p.add_argument("--minor-oversample", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    print(f"Loading {csv_path.name} ...")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if ex["gold"]["NonfictionalChildRelated"])
    gender_count = sum(1 for ex in csv_examples if ex["gold"]["AuthorGenderIndication"])
    print(f"  {len(csv_examples)} examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    minor_csv = [ex for ex in csv_examples if ex["gold"]["NonfictionalChildRelated"]]
    all_examples = csv_examples + minor_csv * args.minor_oversample
    random.seed(args.seed)
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    print(f"\nBuilding span dataset (max {args.max_span_words} words per span) ...")
    train_texts, train_labels = build_span_dataset(train_data, args.max_span_words, args.neg_ratio)
    print(f"  {len(train_texts)} span samples  "
          f"(pos: {sum(1 for l in train_labels if l != 0)}, neg: {sum(1 for l in train_labels if l == 0)})")

    from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitArgs
    from datasets import Dataset

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    print(f"\nLoading SetFit model: {args.model}")
    model = SetFitModel.from_pretrained(
        args.model,
        num_classes=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = SetFitArgs(
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        report_to="none",
    )

    trainer = SetFitTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    print("\nStarting contrastive training ...")
    trainer.train()

    print("\nEvaluating on eval set ...")
    metrics = compute_span_metrics(model, eval_data, args.max_span_words)
    print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    for lbl in ALL_LABELS:
        f = metrics[f"f1_{lbl}"]
        if f > 0:
            print(f"    {lbl}  F1={f:.3f}")

    model.save_pretrained(str(output_dir / "best"))
    print(f"Model saved to {output_dir / 'best'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train DeBERTa-v3-base token-classification NER on mydata.csv.

Labels: NonfictionalChildRelated, AuthorGenderIndication

Install:
  pip install transformers torch datasets seqeval

Usage:
  python train_deberta_csv.py [--epochs 10] [--batch-size 8]
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])
LABEL_LIST = ["O"] + [f"B-{l}" for l in ALL_LABELS] + [f"I-{l}" for l in ALL_LABELS]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, spans = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def char_to_token_span(cs, ce, token_spans) -> Optional[Tuple[int, int]]:
    first = last = None
    for i, (ts, te) in enumerate(token_spans):
        if te > cs and ts < ce:
            if first is None:
                first = i
            last = i
    return None if first is None else (first, last)


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
            tokens, token_spans = tokenize(review)
            if not tokens:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            if not minor_cell and not gender_cell:
                empty.append({"tokenized_text": tokens, "ner": []})
                continue
            ner: List[List] = []
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    r = char_to_token_span(cs, ce, token_spans)
                    if r:
                        ner.append([r[0], r[1], "NonfictionalChildRelated"])
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    r = char_to_token_span(cs, ce, token_spans)
                    if r:
                        ner.append([r[0], r[1], "AuthorGenderIndication"])
            if ner:
                examples.append({"tokenized_text": tokens, "ner": ner})
    n_empty = min(len(empty), len(examples))
    return examples + empty[:n_empty]


def spans_to_bio(tokens: List[str], ner: List[List]) -> List[str]:
    tags = ["O"] * len(tokens)
    for s, e, label in sorted(ner, key=lambda x: x[0]):
        if s < len(tokens):
            tags[s] = f"B-{label}"
            for i in range(s + 1, min(e + 1, len(tokens))):
                tags[i] = f"I-{label}"
    return tags


def to_hf_dataset(examples: List[Dict]) -> Dataset:
    data: Dict[str, list] = {"tokens": [], "ner_tags": []}
    for ex in examples:
        tags = spans_to_bio(ex["tokenized_text"], ex["ner"])
        data["tokens"].append(ex["tokenized_text"])
        data["ner_tags"].append([LABEL2ID[t] for t in tags])
    return Dataset.from_dict(data)


def align_labels(examples, tokenizer):
    enc = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
    )
    all_label_ids = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(labels[wid])
            else:
                orig = LABEL_LIST[labels[wid]]
                label_ids.append(LABEL2ID["I-" + orig[2:]] if orig.startswith("B-") else labels[wid])
            prev = wid
        all_label_ids.append(label_ids)
    enc["labels"] = all_label_ids
    return enc


def compute_metrics(p):
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    logits, labels = p
    preds = np.argmax(logits, axis=2)
    true_lbls = [[LABEL_LIST[l] for l in row if l != -100] for row in labels]
    pred_lbls = [
        [LABEL_LIST[pred] for pred, lbl in zip(pr, lb) if lbl != -100]
        for pr, lb in zip(preds, labels)
    ]
    report = classification_report(true_lbls, pred_lbls, output_dict=True, zero_division=0)
    metrics = {
        "precision": precision_score(true_lbls, pred_lbls),
        "recall": recall_score(true_lbls, pred_lbls),
        "f1": f1_score(true_lbls, pred_lbls),
    }
    for label in ALL_LABELS:
        if label in report:
            metrics[f"f1_{label}"] = report[label]["f1-score"]
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train DeBERTa-v3-base NER on mydata.csv")
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output-dir", default="deberta_finetuned_csv")
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
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "AuthorGenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    random.seed(args.seed)
    random.shuffle(csv_examples)
    n_val = max(1, int(len(csv_examples) * args.val_split))
    eval_data = csv_examples[-n_val:]
    train_base = csv_examples[:-n_val]

    minor_train = [ex for ex in train_base if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"])]
    train_data = train_base + minor_train * args.minor_oversample
    random.shuffle(train_data)
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = to_hf_dataset(train_data).map(
        lambda ex: align_labels(ex, tokenizer), batched=True,
        remove_columns=["tokens", "ner_tags"],
    )
    eval_ds = to_hf_dataset(eval_data).map(
        lambda ex: align_labels(ex, tokenizer), batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    print(f"Loading model: {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting training ...")
    trainer.train()

    # load_best_model_at_end=True means model now holds the best weights
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"Best model saved to {output_dir / 'best'}")

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Final model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()

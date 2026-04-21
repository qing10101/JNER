#!/usr/bin/env python3
"""
Train a SpanMarker NER model on mydata.csv only.

Labels: MinorChild, GenderIndication

Install:
  pip install span-marker torch datasets

Usage:
  python train_spanmarker_csv.py [--epochs 5] [--batch-size 8]
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ALL_LABELS = sorted(["MinorChild", "GenderIndication"])
BIO_LABELS = ["O"] + [f"B-{l}" for l in ALL_LABELS] + [f"I-{l}" for l in ALL_LABELS]


def tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, spans = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def char_to_token_span(
    char_start: int,
    char_end: int,
    token_spans: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    first = last = None
    for i, (ts, te) in enumerate(token_spans):
        if te > char_start and ts < char_end:
            if first is None:
                first = i
            last = i
    if first is None:
        return None
    return first, last


def chunk_examples(examples: List[Dict], max_words: int = 150) -> List[Dict]:
    result = []
    for ex in examples:
        tokens = ex["tokenized_text"]
        ner = ex["ner"]
        if len(tokens) <= max_words:
            result.append(ex)
            continue
        for start in range(0, len(tokens), max_words):
            end = min(start + max_words, len(tokens))
            chunk_ner = [
                [s - start, e - start, lbl]
                for s, e, lbl in ner
                if s >= start and e < end
            ]
            result.append({"tokenized_text": tokens[start:end], "ner": chunk_ner})
    return result


def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lower_phrase = phrase.lower().strip()
    if not lower_phrase:
        return []
    pattern = re.compile(r"\b" + re.escape(lower_phrase) + r"\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_csv(csv_path: Path) -> List[Dict]:
    examples = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("ori_review", "").strip()
            if not review:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            if not minor_cell and not gender_cell:
                continue
            tokens, token_spans = tokenize(review)
            if not tokens:
                continue
            ner: List[List] = []
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    result = char_to_token_span(cs, ce, token_spans)
                    if result:
                        ner.append([result[0], result[1], "MinorChild"])
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    result = char_to_token_span(cs, ce, token_spans)
                    if result:
                        ner.append([result[0], result[1], "GenderIndication"])
            if ner:
                examples.append({"tokenized_text": tokens, "ner": ner})
    return examples


def spans_to_bio(tokens: List[str], spans: List) -> List[str]:
    tags = ["O"] * len(tokens)
    for s, e, label in spans:
        if s >= len(tokens) or e >= len(tokens):
            continue
        if tags[s] == "O":
            tags[s] = f"B-{label}"
            for i in range(s + 1, e + 1):
                if tags[i] == "O":
                    tags[i] = f"I-{label}"
    return tags


def to_hf_dataset(examples: List[Dict]):
    from datasets import ClassLabel, Dataset, Features, Sequence, Value

    label2id = {l: i for i, l in enumerate(BIO_LABELS)}
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=BIO_LABELS)),
    })

    data: Dict[str, list] = {"tokens": [], "ner_tags": []}
    for ex in examples:
        tokens = ex["tokenized_text"]
        tags = spans_to_bio(tokens, ex["ner"])
        data["tokens"].append(tokens)
        data["ner_tags"].append([label2id[t] for t in tags])
    return Dataset.from_dict(data, features=features)


def _make_per_label_callback(eval_data: List[Dict], output_dir: Path):
    from transformers import TrainerCallback

    best_f1 = [-1.0]

    class _PerLabelCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            if model is None:
                return
            texts = [" ".join(ex["tokenized_text"]) for ex in eval_data]
            predictions = model.predict(texts)

            tp_map = {lbl: 0 for lbl in ALL_LABELS}
            fp_map = {lbl: 0 for lbl in ALL_LABELS}
            fn_map = {lbl: 0 for lbl in ALL_LABELS}

            for ex, preds in zip(eval_data, predictions):
                tokens = ex["tokenized_text"]
                gold = {lbl: set() for lbl in ALL_LABELS}
                for s, e, lbl in ex["ner"]:
                    if lbl in gold:
                        gold[lbl].add(" ".join(tokens[s:e + 1]).lower())
                pred = {lbl: set() for lbl in ALL_LABELS}
                for p in preds:
                    if p["label"] in pred:
                        pred[p["label"]].add(p["span"].lower())
                for lbl in ALL_LABELS:
                    tp_map[lbl] += len(gold[lbl] & pred[lbl])
                    fp_map[lbl] += len(pred[lbl] - gold[lbl])
                    fn_map[lbl] += len(gold[lbl] - pred[lbl])

            total_tp = sum(tp_map.values())
            total_fp = sum(fp_map.values())
            total_fn = sum(fn_map.values())
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            overall_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"  Eval (span-match)  F1={overall_f1:.3f}  P={precision:.3f}  R={recall:.3f}")

            per_label = []
            for lbl in ALL_LABELS:
                tp, fp, fn = tp_map[lbl], fp_map[lbl], fn_map[lbl]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > 0:
                    per_label.append((lbl, f1))
            if per_label:
                print("    per-label: " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in per_label))

            if overall_f1 > best_f1[0]:
                best_f1[0] = overall_f1
                model.save_pretrained(str(output_dir / "best"))
                print(f"    ↑ new best F1={overall_f1:.3f} — saved to {output_dir / 'best'}")

    return _PerLabelCallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpanMarker NER on mydata.csv (MinorChild + GenderIndication)")
    p.add_argument("--model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output-dir", default="spanmarker_finetuned_csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--csv", default=None, help="Path to mydata.csv (default: mydata.csv next to the script).")
    p.add_argument("--minor-oversample", type=int, default=2)
    p.add_argument("--entity-max-length", type=int, default=8)
    p.add_argument("--model-max-length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    print(f"Loading {csv_path.name} …")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "GenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} annotated examples  (MinorChild: {minor_count}, GenderIndication: {gender_count})")

    minor_csv = [ex for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"])]
    all_examples = chunk_examples(csv_examples + minor_csv * args.minor_oversample)
    print(f"  {len(all_examples)} examples after chunking + oversampling")

    random.seed(args.seed)
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    train_dataset = to_hf_dataset(train_data)
    eval_dataset = to_hf_dataset(eval_data)

    from span_marker import SpanMarkerModel, Trainer, TrainingArguments

    print(f"\nLoading base model: {args.model}")
    model = SpanMarkerModel.from_pretrained(
        args.model,
        labels=BIO_LABELS,
        model_max_length=args.model_max_length,
        entity_max_length=args.entity_max_length,
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
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[_make_per_label_callback(eval_data, output_dir)],
    )

    print("\nStarting training …")
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    print(f"\nFinal model saved to {output_dir / 'final'}")
    print(f"Best model saved to {output_dir / 'best'} (highest eval_overall_f1)")


if __name__ == "__main__":
    main()

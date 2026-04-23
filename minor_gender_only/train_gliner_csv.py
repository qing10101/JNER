#!/usr/bin/env python3
"""
Train a GLiNER model on mydata.csv only.

Labels: NonfictionalChildRelated, AuthorGenderIndication

Install:
  pip install gliner torch

Usage:
  python train_gliner_csv.py [--epochs 10] [--batch-size 8]
"""

import argparse
import csv
import json
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])


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
    lower_phrase = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lower_phrase:
        return []
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(lower_phrase) + r"(?![a-zA-Z0-9])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(norm_text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def _resolve_overlaps(ner: List[List]) -> List[List]:
    """Keep longest span when token ranges overlap; ties broken by earlier start."""
    sorted_spans = sorted(ner, key=lambda x: (-(x[1] - x[0]), x[0]))
    kept = []
    for s, e, lbl in sorted_spans:
        if not any(s <= ke and ks <= e for ks, ke, _ in kept):
            kept.append([s, e, lbl])
    return kept


def load_csv(csv_path: Path) -> List[Dict]:
    examples = []
    empty = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
                    result = char_to_token_span(cs, ce, token_spans)
                    if result:
                        ner.append([result[0], result[1], "NonfictionalChildRelated"])
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    result = char_to_token_span(cs, ce, token_spans)
                    if result:
                        ner.append([result[0], result[1], "AuthorGenderIndication"])
            if ner:
                examples.append({"tokenized_text": tokens, "ner": _resolve_overlaps(ner)})
    n_empty = min(len(empty), len(examples))
    return examples + empty[:n_empty]


def compute_ner_metrics(
    model, eval_data: List[Dict], labels: List[str], threshold: float = 0.5
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    tp_map: Dict[str, int] = {lbl: 0 for lbl in labels}
    fp_map: Dict[str, int] = {lbl: 0 for lbl in labels}
    fn_map: Dict[str, int] = {lbl: 0 for lbl in labels}

    with torch.no_grad():
        for ex in eval_data:
            tokens = ex["tokenized_text"]
            text = " ".join(tokens)

            gold: Dict[str, set] = {lbl: set() for lbl in labels}
            for s, e, lbl in ex["ner"]:
                if lbl in gold:
                    gold[lbl].add(" ".join(tokens[s: e + 1]).lower())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = model.predict_entities(text, labels, threshold=threshold)

            pred: Dict[str, set] = {lbl: set() for lbl in labels}
            for p in preds:
                if p["label"] in pred:
                    pred[p["label"]].add(p["text"].lower())

            for lbl in labels:
                tp_map[lbl] += len(gold[lbl] & pred[lbl])
                fp_map[lbl] += len(pred[lbl] - gold[lbl])
                fn_map[lbl] += len(gold[lbl] - pred[lbl])

    if was_training:
        model.train()

    total_tp = sum(tp_map.values())
    total_fp = sum(fp_map.values())
    total_fn = sum(fn_map.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results: Dict[str, float] = {"precision": precision, "recall": recall, "f1": f1}
    for lbl in labels:
        tp, fp, fn = tp_map[lbl], fp_map[lbl], fn_map[lbl]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[f"f1_{lbl}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return results


def _print_ner_metrics(metrics: Dict[str, float], labels: List[str]) -> None:
    print(
        f"  Eval  F1={metrics['f1']:.3f}  "
        f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
    )
    active = [(lbl, metrics[f"f1_{lbl}"]) for lbl in labels if metrics[f"f1_{lbl}"] > 0]
    if active:
        print("    per-label: " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in active))


def _make_ner_callback(eval_data: List[Dict], labels: List[str], output_dir: Path, threshold: float = 0.5):
    from transformers import TrainerCallback

    best_f1 = [-1.0]

    class _NERMetricsCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            metrics = compute_ner_metrics(model, eval_data, labels, threshold)
            _print_ner_metrics(metrics, labels)
            if metrics["f1"] > best_f1[0]:
                best_f1[0] = metrics["f1"]
                model.save_pretrained(str(output_dir / "best"))
                print(f"    ↑ new best F1={metrics['f1']:.3f} — saved to {output_dir / 'best'}")

    return _NERMetricsCallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GLiNER on mydata.csv (NonfictionalChildRelated + AuthorGenderIndication)")
    p.add_argument("--model", default="EmergentMethods/gliner_medium_news-v2.1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--output-dir", default="gliner_finetuned_csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--csv", default=None, help="Path to mydata.csv (default: mydata.csv next to the script).")
    p.add_argument("--minor-oversample", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    print(f"Loading {csv_path.name} …")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "AuthorGenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} annotated examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    minor_csv = [ex for ex in csv_examples if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"])]
    all_examples = chunk_examples(csv_examples + minor_csv * args.minor_oversample)
    print(f"  {len(all_examples)} examples after chunking + oversampling")

    random.seed(args.seed)
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    (output_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    (output_dir / "eval.json").write_text(json.dumps(eval_data, indent=2))

    eval_labels = sorted({lbl for ex in eval_data for _, _, lbl in ex["ner"]})

    from gliner import GLiNER

    print(f"\nLoading base model: {args.model}")
    model = GLiNER.from_pretrained(args.model)

    try:
        from gliner.training import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=args.lr,
            weight_decay=0.01,
            others_lr=3e-5,
            others_weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            dataloader_num_workers=0,
            use_cpu=not torch.cuda.is_available(),
            report_to="none",
        )

        try:
            from gliner.data_processing.collator import SpanDataCollator
            data_collator = SpanDataCollator(model.config, model.data_processor)
        except ImportError:
            from gliner.data_processing.collator import DataCollatorWithPadding
            data_collator = DataCollatorWithPadding(model.config, model.data_processor)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            callbacks=[_make_ner_callback(eval_data, eval_labels, output_dir)],
        )

        print("\nStarting training …")
        trainer.train()

    except (ImportError, TypeError):
        _manual_train(model, train_data, eval_data, eval_labels, args, output_dir)

    model.save_pretrained(str(output_dir / "final"))
    print(f"Last model saved to {output_dir / 'final'}")


def _manual_train(
    model, train_data, eval_data, labels: List[str], args, output_dir: Path
) -> None:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        from gliner.data_processing.collator import SpanDataCollator
        collator = SpanDataCollator(model.config, model.data_processor)
    except ImportError:
        from gliner.data_processing.collator import DataCollatorWithPadding
        collator = DataCollatorWithPadding(model.config, model.data_processor)

    optimizer = AdamW(
        [
            {"params": model.token_rep_layer.parameters(), "lr": args.lr},
            {"params": model.prompt_rep_layer.parameters(), "lr": args.lr},
            {"params": [p for n, p in model.named_parameters()
                        if "token_rep_layer" not in n and "prompt_rep_layer" not in n],
             "lr": 3e-5},
        ],
        weight_decay=0.01,
    )
    total_steps = len(train_data) // args.batch_size * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    best_f1 = -1.0
    best_epoch = -1
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)
        total_loss = 0.0
        steps = 0
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i: i + args.batch_size]
            batch_input = collator(batch)
            batch_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_input.items()}
            optimizer.zero_grad()
            loss = model(**batch_input).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")
        metrics = compute_ner_metrics(model, eval_data, labels)
        _print_ner_metrics(metrics, labels)
        model.save_pretrained(str(output_dir / f"checkpoint-epoch-{epoch}"))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            model.save_pretrained(str(output_dir / "best"))
            print(f"    ↑ new best F1={best_f1:.3f} — saved to {output_dir / 'best'}")

    print(f"Manual training complete. Best: epoch {best_epoch}  F1={best_f1:.3f}")


if __name__ == "__main__":
    main()

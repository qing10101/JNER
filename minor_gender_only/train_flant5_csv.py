#!/usr/bin/env python3
"""
Fine-tune FLAN-T5-base as a seq2seq NER model on mydata.csv.

The descriptive label names (NonfictionalChildRelated, AuthorGenderIndication) are
included verbatim in the prompt — T5's instruction-tuning gives it a head start on
understanding the semantics without any examples.

Install:
  pip install transformers torch datasets

Usage:
  python train_flant5_csv.py [--epochs 10] [--batch-size 8]
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])

PROMPT_PREFIX = (
    "Extract named entities from the text below. "
    "Return a JSON array where each element has 'text' (exact span) and 'label' "
    f"(one of: {', '.join(ALL_LABELS)}). Return only the JSON array.\n\nText: "
)


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
                empty.append({"review": review, "entities": []})
                continue
            entities: List[Dict] = []
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append({"text": review[cs:ce], "label": "NonfictionalChildRelated"})
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append({"text": review[cs:ce], "label": "AuthorGenderIndication"})
            # Deduplicate
            seen: Set[tuple] = set()
            unique = []
            for e in entities:
                key = (e["text"].lower(), e["label"])
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            if unique:
                examples.append({"review": review, "entities": unique})
    n_empty = min(len(empty), len(examples))
    return examples + empty[:n_empty]


def to_hf_dataset(examples: List[Dict], tokenizer, max_input: int, max_target: int) -> Dataset:
    inputs, targets = [], []
    for ex in examples:
        inputs.append(PROMPT_PREFIX + ex["review"])
        targets.append(json.dumps(ex["entities"], ensure_ascii=False))

    enc = tokenizer(inputs, max_length=max_input, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        dec = tokenizer(targets, max_length=max_target, truncation=True, padding=False)

    enc["labels"] = dec["input_ids"]
    return Dataset.from_dict(enc)


def _parse_response(response: str) -> List[Dict]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", response, re.DOTALL)
        try:
            return json.loads(m.group()) if m else []
        except (json.JSONDecodeError, AttributeError):
            return []


def eval_span_f1(model, tokenizer, eval_data: List[Dict], device, max_input: int,
                 max_new_tokens: int) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    tp_map = {l: 0 for l in ALL_LABELS}
    fp_map = {l: 0 for l in ALL_LABELS}
    fn_map = {l: 0 for l in ALL_LABELS}

    with torch.no_grad():
        for ex in eval_data:
            prompt = PROMPT_PREFIX + ex["review"]
            enc = tokenizer(prompt, return_tensors="pt", max_length=max_input,
                            truncation=True).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens)
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            parsed = _parse_response(response)

            gold: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
            for e in ex["entities"]:
                if e["label"] in gold:
                    gold[e["label"]].add(e["text"].lower())

            pred: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
            for e in parsed:
                if isinstance(e, dict) and e.get("label") in pred:
                    pred[e["label"]].add(e["text"].lower())

            for lbl in ALL_LABELS:
                tp_map[lbl] += len(gold[lbl] & pred[lbl])
                fp_map[lbl] += len(pred[lbl] - gold[lbl])
                fn_map[lbl] += len(gold[lbl] - pred[lbl])

    total_tp = sum(tp_map.values())
    total_fp = sum(fp_map.values())
    total_fn = sum(fn_map.values())
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    metrics = {"precision": p, "recall": r, "f1": f1}
    for lbl in ALL_LABELS:
        tp_, fp_, fn_ = tp_map[lbl], fp_map[lbl], fn_map[lbl]
        lp = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        lr = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        metrics[f"f1_{lbl}"] = 2 * lp * lr / (lp + lr) if (lp + lr) > 0 else 0.0
    return metrics


def make_eval_callback(eval_data, tokenizer, output_dir, device, max_input, max_new_tokens):
    best_f1 = [-1.0]

    class _EvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            metrics = eval_span_f1(model, tokenizer, eval_data, device, max_input, max_new_tokens)
            print(
                f"  Eval  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
            )
            active = [(l, metrics[f"f1_{l}"]) for l in ALL_LABELS if metrics[f"f1_{l}"] > 0]
            if active:
                print("    per-label: " + "  ".join(f"{l}={v:.3f}" for l, v in active))
            if metrics["f1"] > best_f1[0]:
                best_f1[0] = metrics["f1"]
                model.save_pretrained(str(output_dir / "best"))
                tokenizer.save_pretrained(str(output_dir / "best"))
                print(f"    ↑ new best F1={metrics['f1']:.3f} — saved to {output_dir / 'best'}")

    return _EvalCallback()


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune FLAN-T5 seq2seq NER on mydata.csv")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output-dir", default="flant5_finetuned_csv")
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
    minor_count = sum(1 for ex in csv_examples if any(e["label"] == "NonfictionalChildRelated" for e in ex["entities"]))
    gender_count = sum(1 for ex in csv_examples if any(e["label"] == "AuthorGenderIndication" for e in ex["entities"]))
    print(f"  {len(csv_examples)} examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    random.seed(args.seed)
    random.shuffle(csv_examples)
    n_val = max(1, int(len(csv_examples) * args.val_split))
    eval_data = csv_examples[-n_val:]
    train_base = csv_examples[:-n_val]

    minor_train = [ex for ex in train_base if any(e["label"] == "NonfictionalChildRelated" for e in ex["entities"])]
    train_data = train_base + minor_train * args.minor_oversample
    random.shuffle(train_data)
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    print(f"\nLoading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = to_hf_dataset(train_data, tokenizer, args.max_input_length, args.max_target_length)
    eval_ds = to_hf_dataset(eval_data, tokenizer, args.max_input_length, args.max_target_length)

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        callbacks=[make_eval_callback(eval_data, tokenizer, output_dir, device,
                                      args.max_input_length, args.max_new_tokens)],
    )

    print("\nStarting training ...")
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Final model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()

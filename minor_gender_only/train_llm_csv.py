#!/usr/bin/env python3
"""
Fine-tune Qwen3.5-9B with LoRA on mydata.csv only for NER.

Labels: NonfictionalChildRelated, AuthorGenderIndication

Install:  pip install -r requirements-llm.txt
Usage:    python train_llm_csv.py [--epochs 3] [--batch-size 2]
"""

import argparse
import copy
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])

SYSTEM_PROMPT = (
    "You are a named entity recognition assistant. "
    "Extract all named entities from the given text and return them as a JSON array. "
    'Each element must have exactly two fields: "text" (the exact span as it appears) '
    'and "label" (one of: ' + ", ".join(ALL_LABELS) + "). "
    "Return only the JSON array with no explanation or markdown."
)


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
    return None if first is None else (first, last)


def chunk_examples(examples: List[Dict], max_words: int = 150) -> List[Dict]:
    result = []
    for ex in examples:
        tokens, ner = ex["tokenized_text"], ex["ner"]
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
    lp = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lp:
        return []
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(lp) + r"(?![a-zA-Z0-9])", re.IGNORECASE)
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


def _ner_to_entities(ex: Dict) -> List[Dict]:
    tokens = ex["tokenized_text"]
    seen: Set[Tuple[str, str]] = set()
    entities = []
    for s, e, label in ex["ner"]:
        span_text = " ".join(tokens[s:e + 1])
        key = (span_text.lower(), label)
        if key not in seen:
            entities.append({"text": span_text, "label": label})
            seen.add(key)
    return entities


def build_hf_dataset(raw_examples: List[Dict], tokenizer, max_seq_length: int) -> Dataset:
    records: Dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    skipped = 0
    for ex in raw_examples:
        text = " ".join(ex["tokenized_text"])
        entities = _ner_to_entities(ex)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities:\n{text}"},
            {"role": "assistant", "content": json.dumps(entities, ensure_ascii=False)},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_enc = tokenizer(
            full_text, max_length=max_seq_length, truncation=True, add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text, max_length=max_seq_length, truncation=True, add_special_tokens=False,
        )["input_ids"]

        input_ids = full_enc["input_ids"]
        labels = copy.copy(input_ids)
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len):
            labels[i] = -100

        if all(l == -100 for l in labels):
            skipped += 1
            continue

        records["input_ids"].append(input_ids)
        records["attention_mask"].append(full_enc["attention_mask"])
        records["labels"].append(labels)

    if skipped:
        print(f"  [dataset] Skipped {skipped} examples (response truncated by max_seq_length)")
    return Dataset.from_dict(records)


def make_collator(pad_token_id: int):
    from torch.nn.utils.rnn import pad_sequence

    def collate(batch):
        input_ids = pad_sequence(
            [torch.tensor(ex["input_ids"]) for ex in batch],
            batch_first=True, padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(ex["attention_mask"]) for ex in batch],
            batch_first=True, padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(ex["labels"]) for ex in batch],
            batch_first=True, padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def compute_ner_metrics(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_new_tokens: int = 256,
    n_samples: Optional[int] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    model.config.use_cache = True

    samples = (
        eval_data if n_samples is None
        else random.sample(eval_data, min(n_samples, len(eval_data)))
    )

    tp_map = {lbl: 0 for lbl in ALL_LABELS}
    fp_map = {lbl: 0 for lbl in ALL_LABELS}
    fn_map = {lbl: 0 for lbl in ALL_LABELS}
    parse_errors = 0

    for ex in tqdm(samples, desc="NER eval", leave=False):
        tokens = ex["tokenized_text"]
        text = " ".join(tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities:\n{text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        pred_entities: Set[Tuple[str, str]] = set()
        try:
            parsed = json.loads(response)
            pred_entities = {
                (e["text"].lower(), e["label"])
                for e in parsed
                if isinstance(e, dict) and "text" in e and "label" in e and e["label"] in ALL_LABELS
            }
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    pred_entities = {
                        (e["text"].lower(), e["label"])
                        for e in parsed
                        if isinstance(e, dict) and "text" in e and "label" in e and e["label"] in ALL_LABELS
                    }
                except (json.JSONDecodeError, TypeError):
                    pass

        gold: Dict[str, Set[str]] = {lbl: set() for lbl in ALL_LABELS}
        for s, e, lbl in ex["ner"]:
            if lbl in gold:
                gold[lbl].add(" ".join(tokens[s:e + 1]).lower())

        pred: Dict[str, Set[str]] = {lbl: set() for lbl in ALL_LABELS}
        for span_text, lbl in pred_entities:
            if lbl in pred:
                pred[lbl].add(span_text)

        for lbl in ALL_LABELS:
            tp_map[lbl] += len(gold[lbl] & pred[lbl])
            fp_map[lbl] += len(pred[lbl] - gold[lbl])
            fn_map[lbl] += len(gold[lbl] - pred[lbl])

    model.config.use_cache = False
    if was_training:
        model.train()

    if parse_errors:
        print(f"    [eval] JSON parse errors: {parse_errors}/{len(samples)}")

    total_tp = sum(tp_map.values())
    total_fp = sum(fp_map.values())
    total_fn = sum(fn_map.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results: Dict[str, float] = {"precision": precision, "recall": recall, "f1": f1}
    for lbl in ALL_LABELS:
        tp, fp, fn = tp_map[lbl], fp_map[lbl], fn_map[lbl]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[f"f1_{lbl}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return results


def _print_ner_metrics(metrics: Dict[str, float]) -> None:
    print(
        f"  Eval  F1={metrics['f1']:.3f}  "
        f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
    )
    active = [(lbl, metrics[f"f1_{lbl}"]) for lbl in ALL_LABELS if metrics[f"f1_{lbl}"] > 0]
    if active:
        print("    per-label: " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in active))


def make_eval_callback(
    eval_data: List[Dict],
    tokenizer,
    output_dir: Path,
    n_samples: int,
    max_new_tokens: int,
):
    best_f1 = [-1.0]

    class _NERCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            if n_samples == 0:
                model.save_pretrained(str(output_dir / f"checkpoint-epoch-{state.epoch:.0f}"))
                return
            n = n_samples if n_samples > 0 else len(eval_data)
            print(f"\nEpoch {state.epoch:.0f} NER eval ({n} samples) …")
            torch.cuda.empty_cache()
            metrics = compute_ner_metrics(
                model, tokenizer, eval_data,
                max_new_tokens=max_new_tokens,
                n_samples=n_samples if n_samples > 0 else None,
            )
            _print_ner_metrics(metrics)
            if metrics["f1"] > best_f1[0]:
                best_f1[0] = metrics["f1"]
                model.save_pretrained(str(output_dir / "best"))
                tokenizer.save_pretrained(str(output_dir / "best"))
                print(f"    ↑ new best F1={metrics['f1']:.3f} — saved to {output_dir / 'best'}")
            torch.cuda.empty_cache()

    return _NERCallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LLM (LoRA) on mydata.csv for NonfictionalChildRelated + AuthorGenderIndication NER")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output-dir", default="llm_finetuned_csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--minor-oversample", type=int, default=1)
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--csv", default=None, help="Path to mydata.csv (default: mydata.csv next to the script).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    print(f"Loading {csv_path.name} …")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "AuthorGenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} annotated examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    all_chunked = chunk_examples(csv_examples)
    print(f"  {len(all_chunked)} examples after chunking")

    random.shuffle(all_chunked)
    n_val = max(1, int(len(all_chunked) * args.val_split))
    eval_data = all_chunked[-n_val:]
    train_base = all_chunked[:-n_val]

    minor_train = [ex for ex in train_base if any(s[2] == "NonfictionalChildRelated" for s in ex["ner"])]
    train_data = train_base + minor_train * args.minor_oversample
    random.shuffle(train_data)
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    (output_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    (output_dir / "eval.json").write_text(json.dumps(eval_data, indent=2))

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing training examples …")
    train_dataset = build_hf_dataset(train_data, tokenizer, args.max_seq_length)
    print(f"  {len(train_dataset)} tokenized training examples")

    use_4bit = not args.no_4bit and torch.cuda.is_available()
    print(f"\nLoading base model: {args.model}  (4-bit QLoRA: {use_4bit})")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.bfloat16,
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collator(tokenizer.pad_token_id),
        callbacks=[
            make_eval_callback(
                eval_data, tokenizer, output_dir,
                n_samples=args.eval_samples,
                max_new_tokens=args.max_new_tokens,
            )
        ],
    )

    print("\nStarting training …")
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nFinal LoRA adapter saved to  {output_dir / 'final'}")
    print(f"Best LoRA adapter saved to   {output_dir / 'best'}")


if __name__ == "__main__":
    main()

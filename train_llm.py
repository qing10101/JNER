#!/usr/bin/env python3
"""
Fine-tune Qwen3.5-9B with LoRA on MACCROBAT + Corona2 + mydata.csv for NER.

The model is trained as a text-to-text task: given a text chunk, generate a
JSON array of {"text": ..., "label": ...} entity objects.  Only the LoRA
adapter weights are saved (~hundreds of MB, not the full 9B).

Labels produced:
  MedicalCondition   — from MACCROBAT / Corona2
  ClinicalProcedure  — from MACCROBAT
  ClinicalEvent      — from MACCROBAT
  MinorChild         — from MACCROBAT (Age < 18 + pronoun injection) + mydata.csv
  GenderIndication   — from mydata.csv

Install:  pip install -r requirements-llm.txt
Usage:    python train_llm.py [--epochs 3] [--batch-size 2] [--model Qwen/Qwen3.5-9B]
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
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

MACCROBAT_LABEL_MAP = {
    "Sign_symptom":          "MedicalCondition",
    "Disease_disorder":      "MedicalCondition",
    "Diagnostic_procedure":  "ClinicalProcedure",
    "Therapeutic_procedure": "ClinicalProcedure",
    "Clinical_event":        "ClinicalEvent",
}

CORONA_LABEL_MAP = {
    "MedicalCondition": "MedicalCondition",
}

ALL_LABELS = sorted(
    set(MACCROBAT_LABEL_MAP.values())
    | set(CORONA_LABEL_MAP.values())
    | {"MinorChild", "GenderIndication"}
)

SYSTEM_PROMPT = (
    "You are a named entity recognition assistant. "
    "Extract all named entities from the given text and return them as a JSON array. "
    'Each element must have exactly two fields: "text" (the exact span as it appears) '
    'and "label" (one of: ' + ", ".join(ALL_LABELS) + "). "
    "Return only the JSON array with no explanation or markdown."
)

# ---------------------------------------------------------------------------
# Age → MinorChild helper
# ---------------------------------------------------------------------------

_MINOR_KEYWORDS = {
    "newborn", "neonate", "neonatal", "infant", "baby", "toddler",
    "child", "children", "pediatric", "paediatric",
    "adolescent", "teen", "teenager", "juvenile",
}


def _is_minor_age(span_text: str) -> bool:
    text = span_text.lower()
    if any(kw in text for kw in _MINOR_KEYWORDS):
        return True
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) < 18 if m else False


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MACCROBAT BratStandoff parser
# ---------------------------------------------------------------------------

_MINOR_PRONOUN_RE = re.compile(
    r"\b(he|she|his|her|him|the\s+(?:child|patient|boy|girl|infant|baby|toddler|teen|adolescent))\b",
    re.IGNORECASE,
)


def _inject_minor_pronouns(
    text: str, entities: List[Tuple[int, int, str]]
) -> Tuple[List[Tuple[int, int, str]], int]:
    if not any(lbl == "MinorChild" for _, _, lbl in entities):
        return entities, 0
    existing: Set[Tuple[int, int]] = {(s, e) for s, e, _ in entities}
    extra: List[Tuple[int, int, str]] = []
    for m in _MINOR_PRONOUN_RE.finditer(text):
        span = (m.start(), m.end())
        if span not in existing:
            extra.append((m.start(), m.end(), "MinorChild"))
            existing.add(span)
    return entities + extra, len(extra)


def parse_ann(ann_path: Path, text: str) -> List[Tuple[int, int, str]]:
    entities: List[Tuple[int, int, str]] = []
    sex_spans: List[Tuple[int, int]] = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        fields = parts[1].split()
        if len(fields) < 3:
            continue
        label = fields[0]
        if label == "Age":
            span_text = parts[2].strip() if len(parts) > 2 else ""
            if not _is_minor_age(span_text):
                continue
            mapped = "MinorChild"
        elif label == "Sex":
            mapped = None
        elif label in MACCROBAT_LABEL_MAP:
            mapped = MACCROBAT_LABEL_MAP[label]
        else:
            continue
        try:
            char_start = int(fields[1])
            raw_end = fields[-1]
            char_end = int(raw_end.split(";")[-1]) if ";" in raw_end else int(raw_end)
        except ValueError:
            continue
        if mapped is None:
            sex_spans.append((char_start, char_end))
        else:
            entities.append((char_start, char_end, mapped))
    # Extend MinorChild spans to absorb an immediately adjacent Sex token so
    # span boundaries match mydata.csv, which labels the full noun phrase
    # (e.g. "16-year-old boy") rather than just the age fragment ("16-year-old").
    if sex_spans:
        merged = []
        for cs, ce, lbl in entities:
            if lbl == "MinorChild":
                for ss, se in sex_spans:
                    if 0 < ss - ce <= 2 and text[ce:ss].strip() == "":
                        ce = se
                        break
            merged.append((cs, ce, lbl))
        return merged
    return entities


def load_maccrobat(root: Path) -> List[Dict]:
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(subdirs) > 1:
        print(f"  [MACCROBAT] Found {len(subdirs)} subdirs — using only '{subdirs[0].name}'")
        subdirs = subdirs[:1]
    examples, docs_with_minor, total_injected = [], 0, 0
    for subdir in subdirs:
        for txt_file in sorted(subdir.glob("*.txt")):
            ann_file = txt_file.with_suffix(".ann")
            if not ann_file.exists():
                continue
            text = txt_file.read_text(encoding="utf-8")
            tokens, token_spans = tokenize(text)
            if not tokens:
                continue
            char_entities = parse_ann(ann_file, text)
            char_entities, n = _inject_minor_pronouns(text, char_entities)
            if n > 0:
                docs_with_minor += 1
                total_injected += n
            ner = []
            for cs, ce, label in char_entities:
                result = char_to_token_span(cs, ce, token_spans)
                if result:
                    ner.append([result[0], result[1], label])
            examples.append({"tokenized_text": tokens, "ner": ner})
    print(
        f"  [MACCROBAT] Pronoun injection: {docs_with_minor} docs affected, "
        f"{total_injected} spans added as MinorChild"
    )
    return examples


# ---------------------------------------------------------------------------
# Corona2.json parser
# ---------------------------------------------------------------------------

def load_corona(json_path: Path) -> List[Dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    examples = []
    for ex in data.get("examples", []):
        text = ex.get("content", "")
        if not text:
            continue
        tokens, spans = tokenize(text)
        if not tokens:
            continue
        ner = []
        for ann in ex.get("annotations", []):
            label = ann.get("tag_name", "")
            if label not in CORONA_LABEL_MAP:
                continue
            cs, ce = ann.get("start"), ann.get("end")
            if cs is None or ce is None:
                continue
            result = char_to_token_span(cs, ce, spans)
            if result:
                ner.append([result[0], result[1], CORONA_LABEL_MAP[label]])
        examples.append({"tokenized_text": tokens, "ner": ner})
    return examples


# ---------------------------------------------------------------------------
# mydata.csv loader
# ---------------------------------------------------------------------------

def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lp = phrase.lower().strip()
    if not lp:
        return []
    pattern = re.compile(r"\b" + re.escape(lp) + r"\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_csv(csv_path: Path) -> List[Dict]:
    examples, skipped_medical = [], 0
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("ori_review", "").strip()
            if not review:
                continue
            if row.get("medical_col", "").strip():
                skipped_medical += 1
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
    if skipped_medical:
        print(f"  [CSV] Skipped {skipped_medical} rows with medical_col entries (avoid false negatives)")
    return examples


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt / dataset preparation
# ---------------------------------------------------------------------------

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


def build_hf_dataset(
    raw_examples: List[Dict], tokenizer, max_seq_length: int
) -> Dataset:
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
            full_text,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = full_enc["input_ids"]
        labels = copy.copy(input_ids)
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len):
            labels[i] = -100

        # Skip examples where truncation removed the entire response
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_new_tokens: int = 256,
    n_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Entity-level precision, recall, micro-F1 (lowercase exact span + label match)."""
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

    for ex in samples:
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
        enc = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)

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
                if isinstance(e, dict)
                and "text" in e
                and "label" in e
                and e["label"] in ALL_LABELS
            }
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            # Try extracting a JSON array from a partially-formatted response
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    pred_entities = {
                        (e["text"].lower(), e["label"])
                        for e in parsed
                        if isinstance(e, dict)
                        and "text" in e
                        and "label" in e
                        and e["label"] in ALL_LABELS
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


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3.5-9B (LoRA) for NER")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B",
                   help="HuggingFace model ID (default: Qwen/Qwen3.5-9B)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=768,
                   help="Max tokens per example (prompt + response)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate during NER evaluation")
    p.add_argument("--output-dir", default="llm_finetuned")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--minor-oversample", type=int, default=2)
    p.add_argument("--eval-samples", type=int, default=100,
                   help="Eval examples per epoch for NER generation (0 = all; each sample "
                        "requires one forward+generate pass, so keep this low for speed)")
    p.add_argument("--no-4bit", action="store_true",
                   help="Load in bfloat16 instead of 4-bit QLoRA (needs more VRAM)")
    p.add_argument("--data-dir", default=None,
                   help="MACCROBAT root dir (default: 9764942/ next to script)")
    p.add_argument("--corona", default=None,
                   help="Path to Corona2.json (default: Corona2.json next to script)")
    p.add_argument("--csv", default=None,
                   help="Path to mydata.csv (default: mydata.csv next to script)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "9764942"
    corona_path = Path(args.corona) if args.corona else base_dir / "Corona2.json"
    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading MACCROBAT from {data_dir} …")
    mac_examples = load_maccrobat(data_dir)
    print(f"  {len(mac_examples)} documents")

    print(f"Loading Corona2 from {corona_path} …")
    cor_examples = load_corona(corona_path)
    print(f"  {len(cor_examples)} documents")

    print(f"Loading {csv_path.name} …")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "GenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} annotated examples (MinorChild: {minor_count}, GenderIndication: {gender_count})")

    minor_csv = [ex for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"])]
    all_examples = chunk_examples(
        mac_examples + cor_examples + csv_examples + minor_csv * args.minor_oversample
    )
    print(f"\n  Total: {len(all_examples)} examples after chunking")

    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    (output_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    (output_dir / "eval.json").write_text(json.dumps(eval_data, indent=2))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build HF dataset ──────────────────────────────────────────────────────
    print("Tokenizing training examples …")
    train_dataset = build_hf_dataset(train_data, tokenizer, args.max_seq_length)
    print(f"  {len(train_dataset)} tokenized training examples")

    # ── Load model ────────────────────────────────────────────────────────────
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
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
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
    print("\nTo load for inference:")
    print(f"  from peft import PeftModel")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.model}', ...)")
    print(f"  model = PeftModel.from_pretrained(model, '{output_dir / 'best'}')")


if __name__ == "__main__":
    main()

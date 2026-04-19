#!/usr/bin/env python3
"""
Train a SpanMarker NER model on MACCROBAT + Corona2 + mydata.csv in a single pass.

Labels produced:
  MedicalCondition   — from MACCROBAT / Corona2
  ClinicalProcedure  — from MACCROBAT
  ClinicalEvent      — from MACCROBAT
  MinorChild         — from MACCROBAT (Age < 18 + pronoun injection) + mydata.csv
  GenderIndication   — from mydata.csv

Install dependencies before running:
  pip install -r requirements-spanmarker.txt

Usage:
  python train_spanmarker.py [--epochs 5] [--batch-size 8] [--model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext]
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    if m:
        return int(m.group(1)) < 18
    return False


# ---------------------------------------------------------------------------
# Entity types to keep from each dataset
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
BIO_LABELS = ["O"] + [f"B-{l}" for l in ALL_LABELS] + [f"I-{l}" for l in ALL_LABELS]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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
    if first is None:
        return None
    return first, last


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


def parse_ann(ann_path: Path) -> List[Tuple[int, int, str]]:
    entities: List[Tuple[int, int, str]] = []
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
        entities.append((char_start, char_end, mapped))
    return entities


def load_maccrobat(root: Path) -> List[Dict]:
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(subdirs) > 1:
        print(f"  [MACCROBAT] Found {len(subdirs)} subdirs — using only '{subdirs[0].name}'")
        subdirs = subdirs[:1]

    examples = []
    docs_with_minor = 0
    total_pronouns_injected = 0
    for subdir in subdirs:
        for txt_file in sorted(subdir.glob("*.txt")):
            ann_file = txt_file.with_suffix(".ann")
            if not ann_file.exists():
                continue
            text = txt_file.read_text(encoding="utf-8")
            tokens, token_spans = tokenize(text)
            if not tokens:
                continue
            char_entities = parse_ann(ann_file)
            char_entities, n_injected = _inject_minor_pronouns(text, char_entities)
            if n_injected > 0:
                docs_with_minor += 1
                total_pronouns_injected += n_injected
            ner = []
            for cs, ce, label in char_entities:
                result = char_to_token_span(cs, ce, token_spans)
                if result:
                    ner.append([result[0], result[1], label])
            examples.append({"tokenized_text": tokens, "ner": ner})

    print(
        f"  [MACCROBAT] Pronoun injection: {docs_with_minor} docs affected, "
        f"{total_pronouns_injected} pronoun spans added as MinorChild"
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
    lower_phrase = phrase.lower().strip()
    if not lower_phrase:
        return []
    pattern = re.compile(r"\b" + re.escape(lower_phrase) + r"\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_csv(csv_path: Path) -> List[Dict]:
    examples = []
    skipped_medical = 0
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
# BIO conversion + HuggingFace Dataset
# ---------------------------------------------------------------------------

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
    from datasets import ClassLabel, Dataset, Features, Sequence, Value  # type: ignore[import]

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpanMarker NER on biomedical + review data")
    p.add_argument("--model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                   help="HuggingFace encoder model ID")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output-dir", default="spanmarker_finetuned")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--corona", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--minor-oversample", type=int, default=2)
    p.add_argument("--entity-max-length", type=int, default=8,
                   help="Max tokens per entity span (default: 8)")
    p.add_argument("--model-max-length", type=int, default=256,
                   help="Max input sequence length (default: 256)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "9764942"
    corona_path = Path(args.corona) if args.corona else base_dir / "Corona2.json"
    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading MACCROBAT data from {data_dir} …")
    mac_examples = load_maccrobat(data_dir)
    print(f"  {len(mac_examples)} documents from MACCROBAT")

    print(f"Loading Corona2 data from {corona_path} …")
    cor_examples = load_corona(corona_path)
    print(f"  {len(cor_examples)} documents from Corona2")

    print(f"Loading {csv_path.name} …")
    csv_examples = load_csv(csv_path)
    minor_count = sum(1 for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"]))
    gender_count = sum(1 for ex in csv_examples if any(s[2] == "GenderIndication" for s in ex["ner"]))
    print(f"  {len(csv_examples)} annotated examples  (MinorChild: {minor_count}, GenderIndication: {gender_count})")

    minor_csv = [ex for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"])]
    all_examples = chunk_examples(
        mac_examples + cor_examples + csv_examples + minor_csv * args.minor_oversample
    )
    print(f"\n  Total: {len(all_examples)} examples after chunking")

    # ── Train / eval split ─────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    train_dataset = to_hf_dataset(train_data)
    eval_dataset = to_hf_dataset(eval_data)

    # ── Load SpanMarker model ──────────────────────────────────────────────
    from span_marker import SpanMarkerModel, Trainer, TrainingArguments  # type: ignore[import]

    print(f"\nLoading base model: {args.model}")
    model = SpanMarkerModel.from_pretrained(
        args.model,
        labels=BIO_LABELS,
        model_max_length=args.model_max_length,
        entity_max_length=args.entity_max_length,
    )

    # ── Training ───────────────────────────────────────────────────────────
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
        metric_for_best_model="eval_overall_f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\nStarting training …")
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    print(f"\nFinal model saved to {output_dir / 'final'}")
    print(f"Best model saved to {output_dir / 'best'} (highest eval_overall_f1)")


if __name__ == "__main__":
    main()

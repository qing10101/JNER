#!/usr/bin/env python3
"""
Train a GLiNER model on MACCROBAT + Corona2 + mydata.csv in a single pass.

Labels produced:
  MedicalCondition   — from MACCROBAT / Corona2
  ClinicalProcedure  — from MACCROBAT
  ClinicalEvent      — from MACCROBAT
  Medicine           — from Corona2
  MinorChild         — from MACCROBAT (Age < 18 + pronoun injection) + mydata.csv
  GenderIndication   — from mydata.csv

Install dependencies before running:
  pip install -r requirements.txt

Usage:
  python train_gliner.py [--epochs 10] [--batch-size 8] [--model EmergentMethods/gliner_medium_news-v2.1]
"""

import argparse
import csv
import json
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

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

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_examples(examples: List[Dict], max_words: int = 150) -> List[Dict]:
    """
    Split examples whose tokenized_text is longer than max_words into
    non-overlapping chunks of that size.  150 words × ~2.5 subword tokens
    ≈ 375 subword tokens, safely under the model's 384-token limit.
    NER spans that straddle a chunk boundary are dropped.
    """
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
    """Whitespace-split tokenisation that also preserves character spans."""
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
    """Map a character span to an inclusive [token_start, token_end] pair."""
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

# Pronouns that corefer to a single patient in a clinical case report.
# Plural pronouns (they/them/their) are excluded — in clinical notes they
# typically refer to the medical team, parents, or family, not the patient.
_MINOR_PRONOUN_RE = re.compile(
    r"\b(he|she|his|her|him|the\s+(?:child|patient|boy|girl|infant|baby|toddler|teen|adolescent))\b",
    re.IGNORECASE,
)


def _inject_minor_pronouns(
    text: str, entities: List[Tuple[int, int, str]]
) -> Tuple[List[Tuple[int, int, str]], int]:
    """
    If the document already has at least one MinorChild span (from Age), find
    all third-person pronouns/noun phrases and add them as MinorChild.
    Returns (updated entity list, number of pronouns injected).
    Single-case clinical reports have one patient, so all such pronouns
    corefer to that patient.
    """
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
    """
    Parse entity lines (T…) from a BratStandoff .ann file.
    Returns list of (char_start, char_end, consolidated_label).
    Labels not in MACCROBAT_LABEL_MAP are silently dropped.
    Handles discontinuous spans by using the overall extent.
    """
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
    """
    Load MACCROBAT. The 2018 and 2020 subdirectories are identical, so only
    the first subdirectory (alphabetically) is used to avoid duplicate training.
    Prints a summary of how many documents were affected by pronoun injection.
    """
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(subdirs) > 1:
        # Deduplicate: keep only the first subdir (2018 == 2020 content-wise)
        print(f"  [MACCROBAT] Found {len(subdirs)} subdirs with identical content — using only '{subdirs[0].name}'")
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
            char_entities = parse_ann(ann_file, text)
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
    # Use word-boundary regex so short phrases like "his" don't match inside
    # "this", "history", etc.
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
            # Rows with medical entities must be excluded: we don't annotate
            # medical_col, so those spans would receive false negative gradients
            # that contradict MACCROBAT/Corona2 training.
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
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    model, eval_data: List[Dict], labels: List[str], threshold: float = 0.5
) -> Dict[str, float]:
    """Entity-level precision, recall, micro-F1 (exact span text + label match)."""
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
    from transformers import TrainerCallback  # type: ignore[import]

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GLiNER on biomedical + review NER data")
    p.add_argument("--model", default="EmergentMethods/gliner_medium_news-v2.1",
                   help="HuggingFace model ID to fine-tune (default: numind/NuNER_Zero-span)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--output-dir", default="gliner_finetuned",
                   help="Directory to save the fine-tuned model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of data to use for validation")
    p.add_argument(
        "--data-dir", default=None,
        help="Directory containing MACCROBAT2018/ and MACCROBAT2020/ subdirs. "
             "Defaults to '9764942/' next to the script.",
    )
    p.add_argument("--corona", default=None,
                   help="Path to Corona2.json (default: Corona2.json next to the script).")
    p.add_argument("--csv", default=None,
                   help="Path to mydata.csv (default: mydata.csv next to the script).")
    p.add_argument("--minor-oversample", type=int, default=2,
                   help="How many extra times to repeat MinorChild CSV examples (default: 2).")
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

    # Oversample CSV examples that contain MinorChild to counteract class imbalance.
    minor_csv = [ex for ex in csv_examples if any(s[2] == "MinorChild" for s in ex["ner"])]
    all_examples = chunk_examples(mac_examples + cor_examples + csv_examples + minor_csv * args.minor_oversample)
    print(f"\n  Total: {len(all_examples)} examples after chunking")

    # ── Train / eval split ─────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_split))
    train_data = all_examples[:-n_val]
    eval_data = all_examples[-n_val:]
    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    (output_dir / "train.json").write_text(json.dumps(train_data, indent=2))
    (output_dir / "eval.json").write_text(json.dumps(eval_data, indent=2))

    eval_labels = sorted({lbl for ex in eval_data for _, _, lbl in ex["ner"]})

    # ── Load base GLiNER model ─────────────────────────────────────────────
    from gliner import GLiNER  # type: ignore[import]

    print(f"\nLoading base model: {args.model}")
    model = GLiNER.from_pretrained(args.model)

    # ── Training ───────────────────────────────────────────────────────────
    try:
        from gliner.training import Trainer, TrainingArguments  # type: ignore[import]

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
            from gliner.data_processing.collator import SpanDataCollator  # type: ignore[import]
            data_collator = SpanDataCollator(model.config, model.data_processor)
        except ImportError:
            from gliner.data_processing.collator import DataCollatorWithPadding  # type: ignore[import]
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
        # ImportError  → gliner.training not available in this version
        # TypeError    → accelerate/transformers version mismatch
        _manual_train(model, train_data, eval_data, eval_labels, args, output_dir)

    model.save_pretrained(str(output_dir / "final"))
    print(f"Last model saved to {output_dir / 'final'}")


def _manual_train(
    model, train_data, eval_data, labels: List[str], args, output_dir: Path
) -> None:
    """Minimal training loop for GLiNER versions without a built-in Trainer."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        from gliner.data_processing.collator import SpanDataCollator  # type: ignore[import]
        collator = SpanDataCollator(model.config, model.data_processor)
    except ImportError:
        from gliner.data_processing.collator import DataCollatorWithPadding  # type: ignore[import]
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
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                         total_iters=total_steps)

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

#!/usr/bin/env python3
"""
Train a spaCy NER model on MACCROBAT + Corona2 + mydata.csv in a single pass.

Labels produced:
  MedicalCondition   — from MACCROBAT / Corona2
  ClinicalProcedure  — from MACCROBAT
  ClinicalEvent      — from MACCROBAT
  Medicine           — from Corona2
  MinorChild         — from MACCROBAT (Age < 18 + pronoun injection) + mydata.csv
  GenderIndication   — from mydata.csv

Install:
  pip install spacy
  python -m spacy download en_core_web_lg        # default, general English
  # Recommended for biomedical text:
  pip install scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

Usage:
  python train_spacy.py
  python train_spacy.py --model en_core_sci_lg --epochs 15
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

import spacy
from spacy.training import Example
from spacy.util import filter_spans, minibatch

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
# Entity types
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

ALL_LABELS = (
    set(MACCROBAT_LABEL_MAP.values())
    | set(CORONA_LABEL_MAP.values())
    | {"MinorChild", "GenderIndication"}
)

# ---------------------------------------------------------------------------
# MACCROBAT loading
# ---------------------------------------------------------------------------

# Pronouns that corefer to a single patient in a clinical case report.
# Plural pronouns (they/them/their) are excluded — in clinical notes they
# typically refer to the medical team, parents, or family, not the patient.
_MINOR_PRONOUN_RE = re.compile(
    r"\b(he|she|his|her|him|the\s+(?:child|patient|boy|girl|infant|baby|toddler|teen|adolescent))\b",
    re.IGNORECASE,
)


def _inject_minor_pronouns(text: str, entities: List[Tuple[int, int, str]]) -> Tuple[List[Tuple[int, int, str]], int]:
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
    entities = []
    sex_spans = []
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


def load_maccrobat(root: Path) -> List[Tuple[str, Dict]]:
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

    data = []
    docs_with_minor = 0
    total_pronouns_injected = 0
    for subdir in subdirs:
        for txt_file in sorted(subdir.glob("*.txt")):
            ann_file = txt_file.with_suffix(".ann")
            if not ann_file.exists():
                continue
            text = txt_file.read_text(encoding="utf-8")
            entities = parse_ann(ann_file, text)
            entities, n_injected = _inject_minor_pronouns(text, entities)
            if n_injected > 0:
                docs_with_minor += 1
                total_pronouns_injected += n_injected
            data.append((text, {"entities": entities}))

    print(
        f"  [MACCROBAT] Pronoun injection: {docs_with_minor} docs affected, "
        f"{total_pronouns_injected} pronoun spans added as MinorChild"
    )
    return data


# ---------------------------------------------------------------------------
# Corona2 loading
# ---------------------------------------------------------------------------

def load_corona(json_path: Path) -> List[Tuple[str, Dict]]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    data = []
    for ex in raw.get("examples", []):
        text = ex.get("content", "")
        if not text:
            continue
        entities = []
        for ann in ex.get("annotations", []):
            label = ann.get("tag_name", "")
            if label not in CORONA_LABEL_MAP:
                continue
            cs, ce = ann.get("start"), ann.get("end")
            if cs is not None and ce is not None:
                entities.append((cs, ce, CORONA_LABEL_MAP[label]))
        data.append((text, {"entities": entities}))
    return data


# ---------------------------------------------------------------------------
# mydata.csv loading
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


def load_csv(csv_path: Path) -> List[Tuple[str, Dict]]:
    data = []
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
            entities = []
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append((cs, ce, "MinorChild"))
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    entities.append((cs, ce, "GenderIndication"))
            if entities:
                data.append((review, {"entities": entities}))
    if skipped_medical:
        print(f"  [CSV] Skipped {skipped_medical} rows with medical_col entries (avoid false negatives)")
    return data


# ---------------------------------------------------------------------------
# spaCy Example helpers
# ---------------------------------------------------------------------------

def make_examples(nlp, data: List[Tuple[str, Dict]]) -> List[Example]:
    """
    Convert (text, annotations) pairs to spaCy Examples.
    Spans that don't align with spaCy's tokenisation are dropped;
    overlapping spans are resolved with filter_spans.
    """
    examples = []
    for text, annotations in data:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in annotations["entities"]:
            # Trim leading/trailing whitespace from the annotation boundary
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
        # Drop examples where any entity token has a '-' (misaligned) BILUO tag
        if "-" not in example.get_aligned_ner():
            examples.append(example)
    return examples


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spaCy NER on biomedical + review data")
    p.add_argument(
        "--model", default="en_core_web_lg",
        help="spaCy base model (default: en_core_web_lg). "
             "Use en_core_sci_lg for better biomedical accuracy (requires scispacy).",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--output-dir", default="spacy_model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument(
        "--data-dir", default=None,
        help="Directory containing MACCROBAT2018/ and MACCROBAT2020/. "
             "Defaults to 9764942/ next to the script.",
    )
    p.add_argument("--corona", default=None,
                   help="Path to Corona2.json (default: Corona2.json next to the script).")
    p.add_argument("--csv", default=None,
                   help="Path to mydata.csv (default: mydata.csv next to the script).")
    p.add_argument("--minor-oversample", type=int, default=2,
                   help="How many extra times to repeat MinorChild CSV examples (default: 2).")
    p.add_argument(
        "--use-gpu", action="store_true",
        help="Enable GPU acceleration. Uses CUDA if available; on Apple Silicon "
             "uses MPS via PyTorch (requires thinc>=8.2.0). Falls back to CPU silently.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    if args.use_gpu:
        if spacy.prefer_gpu():
            print("GPU enabled.")
        else:
            print("No GPU found — running on CPU.")

    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "9764942"
    corona_path = Path(args.corona) if args.corona else base_dir / "Corona2.json"
    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading MACCROBAT from {data_dir} …")
    mac_data = load_maccrobat(data_dir)
    print(f"  {len(mac_data)} documents")

    print(f"Loading Corona2 from {corona_path} …")
    cor_data = load_corona(corona_path)
    print(f"  {len(cor_data)} documents")

    print(f"Loading {csv_path.name} …")
    csv_data = load_csv(csv_path)
    minor_count = sum(1 for _, ann in csv_data if any(e[2] == "MinorChild" for e in ann["entities"]))
    gender_count = sum(1 for _, ann in csv_data if any(e[2] == "GenderIndication" for e in ann["entities"]))
    print(f"  {len(csv_data)} annotated examples  (MinorChild: {minor_count}, GenderIndication: {gender_count})")

    # Oversample CSV examples that contain MinorChild to counteract class imbalance
    # (MedicalCondition/ClinicalProcedure dominate with far more training signal).
    minor_csv = [d for d in csv_data if any(e[2] == "MinorChild" for e in d[1]["entities"])]
    all_data = mac_data + cor_data + csv_data + minor_csv * args.minor_oversample
    random.seed(args.seed)
    random.shuffle(all_data)
    n_val = max(1, int(len(all_data) * args.val_split))
    train_raw = all_data[:-n_val]
    eval_raw = all_data[-n_val:]
    print(f"\n  Total — Train: {len(train_raw)}  |  Eval: {len(eval_raw)}")

    # ── Load spaCy model ───────────────────────────────────────────────────
    print(f"\nLoading spaCy model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        raise SystemExit(
            f"Model '{args.model}' not found. "
            f"Run: python -m spacy download {args.model}"
        )

    ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner")
    for label in ALL_LABELS:
        ner.add_label(label)

    # ── Build Examples ─────────────────────────────────────────────────────
    print("Preparing examples …")
    train_examples = make_examples(nlp, train_raw)
    eval_examples = make_examples(nlp, eval_raw)

    # Initialize only the NER component so the transition system learns all new
    # labels.  We call ner.initialize() directly (not nlp.initialize()) to avoid
    # resetting the pretrained tok2vec weights.  tok2vec must remain in the
    # pipeline during this call so the NER listener can resolve its input shape.
    ner.initialize(lambda: train_examples, nlp=nlp)

    # ── Train ──────────────────────────────────────────────────────────────
    # Keep tok2vec enabled — NER is a listener and needs its representations.
    # Disable only components that contribute nothing to NER training.
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

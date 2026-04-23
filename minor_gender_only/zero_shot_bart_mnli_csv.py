#!/usr/bin/env python3
"""
Zero-shot BART MNLI NER baseline on mydata.csv / eval CSV.

No training required.  Uses NLI entailment to detect entity-bearing sentences,
then extracts the specific span using keyword matching within positive sentences.

Hypothesis templates (one per label):
  NonfictionalChildRelated  → "This text mentions a child or minor."
  AuthorGenderIndication    → "This text indicates the gender of a person."

Usage:
  python zero_shot_bart_mnli_csv.py --csv mydata.csv
  python zero_shot_bart_mnli_csv.py --csv eval_sample.csv --text-col original_text
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])

HYPOTHESES: Dict[str, str] = {
    "NonfictionalChildRelated": "This text mentions a child or minor.",
    "AuthorGenderIndication": "This text indicates the gender of a person.",
}

# Fallback keyword extractor when entailment fires but no finer span is found
_KEYWORD_MINOR = re.compile(
    r"\b(son|daughter|child|children|kid|kids|boy|girl|baby|babies|infant|toddler|"
    r"teen|teenager|adolescent|minor|youth|newborn|grandchild|grandson|granddaughter|"
    r"preschooler|juvenile)\b",
    re.IGNORECASE,
)
_KEYWORD_GENDER = re.compile(
    r"\b(husband|wife|boyfriend|girlfriend|brother|sister|father|mother|dad|mom|"
    r"grandfather|grandmother|uncle|aunt|nephew|niece|man|woman|male|female|"
    r"he|she|his|her|him|himself|herself|mr|mrs|ms|sir|gentleman|lady)\b",
    re.IGNORECASE,
)
_KEYWORD_RE: Dict[str, re.Pattern] = {
    "NonfictionalChildRelated": _KEYWORD_MINOR,
    "AuthorGenderIndication": _KEYWORD_GENDER,
}


def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lp = phrase.lower().strip()
    if not lp:
        return []
    return [(m.start(), m.end()) for m in
            re.compile(r"\b" + re.escape(lp) + r"\b", re.IGNORECASE).finditer(text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_eval_data(csv_path: Path, text_col: str) -> List[Tuple[str, Dict[str, Set[str]]]]:
    examples = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            review = row.get(text_col, "").strip()
            if not review:
                continue
            gold: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
            for phrase in _parse_cell(row.get("minor_col", "")):
                for cs, ce in _find_all_spans(review, phrase):
                    gold["NonfictionalChildRelated"].add(review[cs:ce].lower())
            for phrase in _parse_cell(row.get("gender_col", "")):
                for cs, ce in _find_all_spans(review, phrase):
                    gold["AuthorGenderIndication"].add(review[cs:ce].lower())
            examples.append((review, gold))
    return examples


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def predict_spans(
    classifier,
    texts: List[str],
    threshold: float,
    batch_size: int,
) -> List[Dict[str, Set[str]]]:
    """Run NLI on sentence windows, extract keyword spans from positive sentences."""
    from tqdm import tqdm
    results: List[Dict[str, Set[str]]] = []

    for text in tqdm(texts, desc="BART MNLI inference"):
        pred: Dict[str, Set[str]] = {l: set() for l in ALL_LABELS}
        sentences = split_sentences(text)

        for lbl, hypothesis in HYPOTHESES.items():
            # Batch sentences for this label
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i: i + batch_size]
                outputs = classifier(batch, hypothesis, multi_label=False)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for sent, out in zip(batch, outputs):
                    if out["scores"][0] >= threshold:
                        # Extract keyword spans within this sentence
                        for m in _KEYWORD_RE[lbl].finditer(sent):
                            pred[lbl].add(m.group().lower())
        results.append(pred)

    return results


def compute_metrics(
    golds: List[Dict[str, Set[str]]],
    preds: List[Dict[str, Set[str]]],
) -> Dict[str, float]:
    tp_map = {l: 0 for l in ALL_LABELS}
    fp_map = {l: 0 for l in ALL_LABELS}
    fn_map = {l: 0 for l in ALL_LABELS}
    for gold, pred in zip(golds, preds):
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
    results = {"precision": p, "recall": r, "f1": f1}
    for lbl in ALL_LABELS:
        tp_, fp_, fn_ = tp_map[lbl], fp_map[lbl], fn_map[lbl]
        lp = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        lr = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        results[f"f1_{lbl}"] = 2 * lp * lr / (lp + lr) if (lp + lr) > 0 else 0.0
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot BART MNLI NER baseline")
    p.add_argument("--model", default="facebook/bart-large-mnli")
    p.add_argument("--csv", default=None, help="Eval CSV path (default: mydata.csv next to script)")
    p.add_argument("--text-col", default="ori_review", help="CSV column for review text")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Entailment score threshold (default: 0.7)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Sentences per NLI batch (default: 16)")
    p.add_argument("--annotated-only", action="store_true",
                   help="Evaluate only on rows with at least one gold entity")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent
    csv_path = Path(args.csv) if args.csv else base_dir / "mydata.csv"

    print(f"Loading eval data from {csv_path.name} ...")
    examples = load_eval_data(csv_path, args.text_col)
    if args.annotated_only:
        examples = [(t, g) for t, g in examples if any(g.values())]
        print(f"  --annotated-only: {len(examples)} rows")
    else:
        print(f"  {len(examples)} rows")

    print(f"\nLoading zero-shot classifier: {args.model}")
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model=args.model)

    texts = [t for t, _ in examples]
    golds = [g for _, g in examples]

    print(f"Running NLI inference (threshold={args.threshold}) ...")
    preds = predict_spans(classifier, texts, args.threshold, args.batch_size)

    metrics = compute_metrics(golds, preds)
    print(f"\n── BART MNLI (zero-shot) ──")
    print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    for lbl in ALL_LABELS:
        f = metrics[f"f1_{lbl}"]
        if f > 0:
            print(f"    {lbl}  F1={f:.3f}")


if __name__ == "__main__":
    main()

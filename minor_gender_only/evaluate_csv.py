#!/usr/bin/env python3
"""
Evaluate trained NER models on mydata.csv using consistent span-text exact match.

All models are optional — only those with a provided path are evaluated.

Usage:
  python evaluate_csv.py --spacy spacy_model_csv/best --gliner gliner_finetuned_csv/best
  python evaluate_csv.py --spacy spacy_model_csv/best --spanmarker spanmarker_finetuned_csv/best
  python evaluate_csv.py --llm llm_finetuned_csv/best --llm-base Qwen/Qwen3.5-9B --llm-samples 200
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

ALL_LABELS = sorted(["MinorChild", "GenderIndication"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lp = phrase.lower().strip()
    if not lp:
        return []
    pattern = re.compile(r"\b" + re.escape(lp) + r"\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_eval_data(csv_path: Path) -> List[Tuple[str, Set[Tuple[str, str]]]]:
    """Returns list of (review_text, gold_spans) for ALL rows. gold_spans is empty for unannotated rows."""
    examples = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("ori_review", "").strip()
            if not review:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            gold: Set[Tuple[str, str]] = set()
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold.add((review[cs:ce].lower(), "MinorChild"))
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold.add((review[cs:ce].lower(), "GenderIndication"))
            examples.append((review, gold))
    return examples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    golds: List[Set[Tuple[str, str]]],
    preds: List[Set[Tuple[str, str]]],
) -> Dict[str, float]:
    tp_map = {lbl: 0 for lbl in ALL_LABELS}
    fp_map = {lbl: 0 for lbl in ALL_LABELS}
    fn_map = {lbl: 0 for lbl in ALL_LABELS}

    for gold, pred in zip(golds, preds):
        gold_by_lbl = {lbl: {s for s, l in gold if l == lbl} for lbl in ALL_LABELS}
        pred_by_lbl = {lbl: {s for s, l in pred if l == lbl} for lbl in ALL_LABELS}
        for lbl in ALL_LABELS:
            tp_map[lbl] += len(gold_by_lbl[lbl] & pred_by_lbl[lbl])
            fp_map[lbl] += len(pred_by_lbl[lbl] - gold_by_lbl[lbl])
            fn_map[lbl] += len(gold_by_lbl[lbl] - pred_by_lbl[lbl])

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


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n── {name} ──")
    print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    per = [(lbl, metrics[f"f1_{lbl}"]) for lbl in ALL_LABELS if metrics[f"f1_{lbl}"] > 0]
    if per:
        print("  per-label: " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in per))


# ---------------------------------------------------------------------------
# Inference per model
# ---------------------------------------------------------------------------

def eval_spacy(model_path: str, texts: List[str]) -> List[Set[Tuple[str, str]]]:
    import spacy
    nlp = spacy.load(model_path)
    preds = []
    for doc in nlp.pipe(texts, batch_size=32):
        preds.append({(ent.text.lower(), ent.label_) for ent in doc.ents if ent.label_ in ALL_LABELS})
    return preds


def _chunk_text(text: str, max_words: int = 150) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def eval_gliner(model_path: str, texts: List[str], threshold: float) -> List[Set[Tuple[str, str]]]:
    from gliner import GLiNER
    model = GLiNER.from_pretrained(model_path)
    model.eval()
    preds = []
    for text in texts:
        pred_set: Set[Tuple[str, str]] = set()
        for chunk in _chunk_text(text):
            entities = model.predict_entities(chunk, ALL_LABELS, threshold=threshold)
            pred_set.update((e["text"].lower(), e["label"]) for e in entities if e["label"] in ALL_LABELS)
        preds.append(pred_set)
    return preds


def eval_spanmarker(model_path: str, texts: List[str], batch_size: int = 32) -> List[Set[Tuple[str, str]]]:
    from span_marker import SpanMarkerModel
    from tqdm import tqdm
    model = SpanMarkerModel.from_pretrained(model_path)
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SpanMarker inference"):
        batch = texts[i: i + batch_size]
        for pred_list in model.predict(batch):
            preds.append({(p["span"].lower(), p["label"]) for p in pred_list if p["label"] in ALL_LABELS})
    return preds


def eval_llm(
    adapter_path: str,
    base_model: str,
    texts: List[str],
    max_new_tokens: int,
) -> List[Set[Tuple[str, str]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    system_prompt = (
        "You are a named entity recognition assistant. "
        "Extract all named entities from the given text and return them as a JSON array. "
        'Each element must have exactly two fields: "text" (the exact span as it appears) '
        'and "label" (one of: ' + ", ".join(ALL_LABELS) + "). "
        "Return only the JSON array with no explanation or markdown."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    preds = []
    for text in tqdm(texts, desc="LLM inference"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract entities:\n{text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        pred_set: Set[Tuple[str, str]] = set()
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", response, re.DOTALL)
            try:
                parsed = json.loads(m.group()) if m else []
            except (json.JSONDecodeError, AttributeError):
                parsed = []
        for e in parsed:
            if isinstance(e, dict) and "text" in e and "label" in e and e["label"] in ALL_LABELS:
                pred_set.add((e["text"].lower(), e["label"]))
        preds.append(pred_set)
    return preds


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NER models on mydata.csv with consistent span-text exact match")
    p.add_argument("--csv", default=None, help="Path to eval CSV (default: ../mydata.csv)")
    p.add_argument("--spacy", default=None, metavar="PATH", help="spaCy model dir")
    p.add_argument("--gliner", default=None, metavar="PATH", help="GLiNER model dir")
    p.add_argument("--spanmarker", default=None, metavar="PATH", help="SpanMarker model dir")
    p.add_argument("--llm", default=None, metavar="PATH", help="LLM LoRA adapter dir")
    p.add_argument("--llm-base", default="Qwen/Qwen3.5-9B", help="Base model ID for LLM (default: Qwen/Qwen3.5-9B)")
    p.add_argument("--threshold", type=float, default=0.5, help="GLiNER prediction threshold (default: 0.5)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for LLM generation (default: 256)")
    p.add_argument("--llm-samples", type=int, default=None,
                   help="Limit LLM eval to N random samples — LLM inference is slow (default: all)")
    p.add_argument("--annotated-only", action="store_true",
                   help="Only evaluate on rows that contain at least one entity from either category")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    csv_path = Path(args.csv) if args.csv else base_dir / "../mydata.csv"

    print(f"Loading eval data from {csv_path} …")
    examples = load_eval_data(csv_path)
    minor_count = sum(1 for _, g in examples if any(l == "MinorChild" for _, l in g))
    gender_count = sum(1 for _, g in examples if any(l == "GenderIndication" for _, l in g))
    print(f"  {len(examples)} annotated examples  (MinorChild: {minor_count}, GenderIndication: {gender_count})")

    if args.annotated_only:
        examples = [(t, g) for t, g in examples if g]
        print(f"  --annotated-only: filtered to {len(examples)} rows containing at least one entity")

    if not any([args.spacy, args.gliner, args.spanmarker, args.llm]):
        print("\nNo models specified. Use --spacy, --gliner, --spanmarker, or --llm.")
        return

    texts = [text for text, _ in examples]
    golds = [gold for _, gold in examples]

    if args.spacy:
        print(f"\nEvaluating spaCy from {args.spacy} …")
        preds = eval_spacy(args.spacy, texts)
        print_metrics("spaCy", compute_metrics(golds, preds))

    if args.gliner:
        print(f"\nEvaluating GLiNER from {args.gliner} …")
        preds = eval_gliner(args.gliner, texts, args.threshold)
        print_metrics("GLiNER", compute_metrics(golds, preds))

    if args.spanmarker:
        print(f"\nEvaluating SpanMarker from {args.spanmarker} …")
        preds = eval_spanmarker(args.spanmarker, texts)
        print_metrics("SpanMarker", compute_metrics(golds, preds))

    if args.llm:
        llm_texts, llm_golds = texts, golds
        if args.llm_samples:
            import random
            indices = random.sample(range(len(texts)), min(args.llm_samples, len(texts)))
            llm_texts = [texts[i] for i in indices]
            llm_golds = [golds[i] for i in indices]
        print(f"\nEvaluating LLM from {args.llm} on {len(llm_texts)} examples …")
        preds = eval_llm(args.llm, args.llm_base, llm_texts, args.max_new_tokens)
        print_metrics("LLM", compute_metrics(llm_golds, preds))


if __name__ == "__main__":
    main()

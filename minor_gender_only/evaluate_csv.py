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

ALL_LABELS = sorted(["NonfictionalChildRelated", "AuthorGenderIndication"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_all_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    lp = phrase.lower().strip().replace("\u2019", "'").replace("\u2018", "'")
    if not lp:
        return []
    norm_text = text.replace("\u2019", "'").replace("\u2018", "'")
    pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(lp) + r"(?![a-zA-Z0-9])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(norm_text)]


def _parse_cell(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_eval_data(csv_path: Path, text_col: str = "ori_review") -> List[Tuple[str, Set[Tuple[str, str]]]]:
    """Returns list of (review_text, gold_spans) for ALL rows. gold_spans is empty for unannotated rows."""
    examples = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get(text_col, "").strip()
            if not review:
                continue
            minor_cell = row.get("minor_col", "").strip()
            gender_cell = row.get("gender_col", "").strip()
            gold: Set[Tuple[str, str]] = set()
            for phrase in _parse_cell(minor_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold.add((review[cs:ce].lower(), "NonfictionalChildRelated"))
            for phrase in _parse_cell(gender_cell):
                for cs, ce in _find_all_spans(review, phrase):
                    gold.add((review[cs:ce].lower(), "AuthorGenderIndication"))
            examples.append((review, gold))
    return examples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_POSSESSIVE_RE = re.compile(r"'s?$|'s?$", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:my|a|an|the)\s+", re.IGNORECASE)


def _norm_text(text: str) -> str:
    text = _POSSESSIVE_RE.sub("", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text).strip()
    return text


def _norm(spans: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    return {(_norm_text(text), label) for text, label in spans}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    return inter / len(ta | tb) if inter else 0.0


def _is_numeric(text: str) -> bool:
    return all(t.isdigit() for t in text.split()) and bool(text.strip())


def _soft_tp(gold_texts: Set[str], pred_texts: Set[str]) -> float:
    """Greedy 1-to-1 matching. Exact = 1.0; token-Jaccard overlap > 0 = partial credit.
    Purely numeric spans are excluded from partial credit (exact match only)."""
    if not gold_texts or not pred_texts:
        return 0.0
    exact = gold_texts & pred_texts
    score = float(len(exact))
    rem_g, rem_p = gold_texts - exact, pred_texts - exact
    if not rem_g or not rem_p:
        return score
    pairs = sorted(
        (
            (g, p, _token_jaccard(g, p))
            for g in rem_g for p in rem_p
            if not _is_numeric(g) and not _is_numeric(p)
        ),
        key=lambda x: -x[2],
    )
    used_g: Set[str] = set()
    used_p: Set[str] = set()
    for g, p, s in pairs:
        if s == 0.0:
            break
        if g not in used_g and p not in used_p:
            score += s
            used_g.add(g)
            used_p.add(p)
    return score


def _row_prf(gold: Set[Tuple[str, str]], pred: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    gold, pred = _norm(gold), _norm(pred)
    soft_tp = n_gold = n_pred = 0.0
    for lbl in ALL_LABELS:
        g = {s for s, l in gold if l == lbl}
        p = {s for s, l in pred if l == lbl}
        soft_tp += _soft_tp(g, p)
        n_gold += len(g)
        n_pred += len(p)
    prec = soft_tp / n_pred if n_pred > 0 else 1.0
    rec = soft_tp / n_gold if n_gold > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_row_metrics(
    texts: List[str],
    golds: List[Set[Tuple[str, str]]],
    preds: List[Set[Tuple[str, str]]],
) -> List[Dict]:
    rows = []
    for text, gold, pred in zip(texts, golds, preds):
        p, r, f1 = _row_prf(gold, pred)
        row: Dict = {
            "review": text,
            "gold": "; ".join(sorted(f"{s}|{l}" for s, l in gold)),
            "pred": "; ".join(sorted(f"{s}|{l}" for s, l in pred)),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        for lbl in ALL_LABELS:
            lbl_gold = {(s, l) for s, l in gold if l == lbl}
            lbl_pred = {(s, l) for s, l in pred if l == lbl}
            lp, lr, lf = _row_prf(lbl_gold, lbl_pred)
            row[f"precision_{lbl}"] = round(lp, 4)
            row[f"recall_{lbl}"] = round(lr, 4)
            row[f"f1_{lbl}"] = round(lf, 4)
        rows.append(row)
    return rows


def write_row_csv(path: Path, rows: List[Dict]) -> None:
    per_lbl_cols = [f"{m}_{lbl}" for lbl in ALL_LABELS for m in ("precision", "recall", "f1")]
    fieldnames = ["review", "gold", "pred", "precision", "recall", "f1"] + per_lbl_cols
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Row-level results written to {path}")


def compute_metrics(
    golds: List[Set[Tuple[str, str]]],
    preds: List[Set[Tuple[str, str]]],
) -> Dict[str, float]:
    stp_map: Dict[str, float] = {lbl: 0.0 for lbl in ALL_LABELS}
    ng_map: Dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}
    np_map: Dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}

    for gold, pred in zip(golds, preds):
        gold, pred = _norm(gold), _norm(pred)
        for lbl in ALL_LABELS:
            g = {s for s, l in gold if l == lbl}
            p = {s for s, l in pred if l == lbl}
            stp_map[lbl] += _soft_tp(g, p)
            ng_map[lbl] += len(g)
            np_map[lbl] += len(p)

    total_stp = sum(stp_map.values())
    total_ng = sum(ng_map.values())
    total_np = sum(np_map.values())
    precision = total_stp / total_np if total_np > 0 else 0.0
    recall = total_stp / total_ng if total_ng > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results: Dict[str, float] = {"precision": precision, "recall": recall, "f1": f1}
    for lbl in ALL_LABELS:
        p = stp_map[lbl] / np_map[lbl] if np_map[lbl] > 0 else 0.0
        r = stp_map[lbl] / ng_map[lbl] if ng_map[lbl] > 0 else 0.0
        results[f"precision_{lbl}"] = p
        results[f"recall_{lbl}"] = r
        results[f"f1_{lbl}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return results


def print_metrics(name: str, metrics: Dict[str, float], row_rows: List[Dict] = None) -> None:
    print(f"\n── {name} ──")
    print(f"  Span (corpus)  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    for lbl in ALL_LABELS:
        p = metrics.get(f"precision_{lbl}", 0.0)
        r = metrics.get(f"recall_{lbl}", 0.0)
        f = metrics.get(f"f1_{lbl}", 0.0)
        if p or r or f:
            print(f"    {lbl}  F1={f:.3f}  P={p:.3f}  R={r:.3f}")
    if row_rows:
        n = len(row_rows)
        mean_p = sum(r["precision"] for r in row_rows) / n
        mean_r = sum(r["recall"] for r in row_rows) / n
        mean_f1 = sum(r["f1"] for r in row_rows) / n
        print(f"  Row-avg  F1={mean_f1:.3f}  P={mean_p:.3f}  R={mean_r:.3f}  (n={n})")
        for lbl in ALL_LABELS:
            lp = sum(r[f"precision_{lbl}"] for r in row_rows) / n
            lr = sum(r[f"recall_{lbl}"] for r in row_rows) / n
            lf = sum(r[f"f1_{lbl}"] for r in row_rows) / n
            if lp or lr or lf:
                print(f"    {lbl}  F1={lf:.3f}  P={lp:.3f}  R={lr:.3f}")


# ---------------------------------------------------------------------------
# Keyword baseline
# ---------------------------------------------------------------------------

_KEYWORD_MINOR: List[str] = [
    # multi-word first so they match before their sub-tokens
    "little one", "little ones", "grand child", "grandchildren", "grand children",
    "grandson", "granddaughter", "grand son", "grand daughter",
    "baby girl", "baby boy",
    # single-word
    "son", "daughter", "child", "children", "kid", "kids",
    "boy", "girl", "baby", "babies", "infant", "infants",
    "toddler", "toddlers", "teen", "teens", "teenager", "teenagers",
    "adolescent", "adolescents", "minor", "minors", "youth", "youths",
    "newborn", "newborns", "preschooler", "preschoolers",
    "grandchild", "juvenile", "juveniles",
]

_KEYWORD_GENDER: List[str] = [
    # multi-word first
    "boy friend", "girl friend",
    # titles
    # "mr.", "mrs.", "ms.", "mr", "mrs", "ms",
    # relationship/family
    "husband", "wife", "boyfriend", "girlfriend", "fiancé", "fiancée",
    # "fiance", "fiancee",
    # "brother", "sister", "father", "mother", "dad", "mom",
    # "grandfather", "grandmother", "grandpa", "grandma",
    # "uncle", "aunt", "nephew", "niece",
    # "stepfather", "stepmother", "stepson", "stepdaughter",
    # "stepbrother", "stepsister",
    # general gender terms
    # "man", "woman", "male", "female", "gentleman", "lady",
    # "guy", "gal", "sir", "madam",
    # # pronouns
    # "he", "she", "his", "her", "him", "himself", "herself",
]

# Compile once: longest patterns first to avoid sub-match shadowing
def _build_keyword_re(keywords: List[str]) -> re.Pattern:
    sorted_kw = sorted(keywords, key=len, reverse=True)
    alts = "|".join(re.escape(k) for k in sorted_kw)
    return re.compile(r"\b(?:" + alts + r")\b", re.IGNORECASE)

_RE_MINOR = _build_keyword_re(_KEYWORD_MINOR)
_RE_GENDER = _build_keyword_re(_KEYWORD_GENDER)


def eval_keyword_baseline(texts: List[str]) -> List[Set[Tuple[str, str]]]:
    preds = []
    for text in texts:
        spans: Set[Tuple[str, str]] = set()
        for m in _RE_MINOR.finditer(text):
            spans.add((m.group().lower(), "NonfictionalChildRelated"))
        for m in _RE_GENDER.finditer(text):
            spans.add((m.group().lower(), "AuthorGenderIndication"))
        preds.append(spans)
    return preds


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


def eval_spanmarker(model_path: str, texts: List[str], threshold: float = 0.5, batch_size: int = 32) -> List[Set[Tuple[str, str]]]:
    from span_marker import SpanMarkerModel
    from tqdm import tqdm
    model = SpanMarkerModel.from_pretrained(model_path)
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SpanMarker inference"):
        batch = texts[i: i + batch_size]
        for pred_list in model.predict(batch):
            preds.append({
                (p["span"].lower(), p["label"])
                for p in pred_list
                if p["label"] in ALL_LABELS and p.get("score", 1.0) >= threshold
            })
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
    p.add_argument("--text-col", default="ori_review", help="CSV column name for review text (default: ori_review)")
    p.add_argument("--spacy", default=None, metavar="PATH", help="spaCy model dir")
    p.add_argument("--gliner", default=None, metavar="PATH", help="GLiNER model dir")
    p.add_argument("--spanmarker", default=None, metavar="PATH", help="SpanMarker model dir")
    p.add_argument("--llm", default=None, metavar="PATH", help="LLM LoRA adapter dir")
    p.add_argument("--llm-base", default="Qwen/Qwen3.5-9B", help="Base model ID for LLM (default: Qwen/Qwen3.5-9B)")
    p.add_argument("--gliner-threshold", type=float, default=0.5, help="GLiNER confidence threshold (default: 0.5)")
    p.add_argument("--spanmarker-threshold", type=float, default=0.5, help="SpanMarker confidence threshold (default: 0.5)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for LLM generation (default: 256)")
    p.add_argument("--llm-samples", type=int, default=None,
                   help="Limit LLM eval to N random samples — LLM inference is slow (default: all)")
    p.add_argument("--annotated-only", action="store_true",
                   help="Only evaluate on rows that contain at least one entity from either category")
    p.add_argument("--row-output", default=None, metavar="PATH",
                   help="Write per-row precision/recall/f1 to this CSV path (one file per model, suffix added)")
    p.add_argument("--keyword-baseline", action="store_true",
                   help="Run keyword-matching baseline (no model required)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).parent
    csv_path = Path(args.csv) if args.csv else base_dir / "../mydata.csv"

    print(f"Loading eval data from {csv_path} …")
    examples = load_eval_data(csv_path, text_col=args.text_col)
    minor_count = sum(1 for _, g in examples if any(l == "NonfictionalChildRelated" for _, l in g))
    gender_count = sum(1 for _, g in examples if any(l == "AuthorGenderIndication" for _, l in g))
    print(f"  {len(examples)} annotated examples  (NonfictionalChildRelated: {minor_count}, AuthorGenderIndication: {gender_count})")

    if args.annotated_only:
        examples = [(t, g) for t, g in examples if g]
        print(f"  --annotated-only: filtered to {len(examples)} rows containing at least one entity")

    if not any([args.spacy, args.gliner, args.spanmarker, args.llm, args.keyword_baseline]):
        print("\nNo models specified. Use --spacy, --gliner, --spanmarker, --llm, or --keyword-baseline.")
        return

    texts = [text for text, _ in examples]
    golds = [gold for _, gold in examples]

    def _maybe_write_rows(name: str, t: List[str], g: List[Set], p: List[Set]) -> List[Dict]:
        rows = compute_row_metrics(t, g, p)
        if args.row_output:
            out = Path(args.row_output)
            write_row_csv(out.with_stem(out.stem + f"_{name.lower()}"), rows)
        return rows

    if args.keyword_baseline:
        print("\nEvaluating keyword baseline …")
        preds = eval_keyword_baseline(texts)
        rows = _maybe_write_rows("keyword_baseline", texts, golds, preds)
        print_metrics("Keyword Baseline", compute_metrics(golds, preds), rows)

    if args.spacy:
        print(f"\nEvaluating spaCy from {args.spacy} …")
        preds = eval_spacy(args.spacy, texts)
        rows = _maybe_write_rows("spacy", texts, golds, preds)
        print_metrics("spaCy", compute_metrics(golds, preds), rows)

    if args.gliner:
        print(f"\nEvaluating GLiNER from {args.gliner} …")
        preds = eval_gliner(args.gliner, texts, args.gliner_threshold)
        rows = _maybe_write_rows("gliner", texts, golds, preds)
        print_metrics("GLiNER", compute_metrics(golds, preds), rows)

    if args.spanmarker:
        print(f"\nEvaluating SpanMarker from {args.spanmarker} …")
        preds = eval_spanmarker(args.spanmarker, texts, args.spanmarker_threshold)
        rows = _maybe_write_rows("spanmarker", texts, golds, preds)
        print_metrics("SpanMarker", compute_metrics(golds, preds), rows)

    if args.llm:
        llm_texts, llm_golds = texts, golds
        if args.llm_samples:
            import random
            indices = random.sample(range(len(texts)), min(args.llm_samples, len(texts)))
            llm_texts = [texts[i] for i in indices]
            llm_golds = [golds[i] for i in indices]
        print(f"\nEvaluating LLM from {args.llm} on {len(llm_texts)} examples …")
        preds = eval_llm(args.llm, args.llm_base, llm_texts, args.max_new_tokens)
        rows = _maybe_write_rows("llm", llm_texts, llm_golds, preds)
        print_metrics("LLM", compute_metrics(llm_golds, preds), rows)


if __name__ == "__main__":
    main()

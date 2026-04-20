# JNER — Training Scripts

NER training pipeline for detecting **minor children**, **gender indications**, and **biomedical entities** in text. Three model backends are provided: [GLiNER](https://github.com/urchade/GLiNER) (transformer-based, retains zero-shot generalization), [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) (BERT span classifier, higher accuracy on fixed labels), and [spaCy](https://spacy.io/) (lighter, faster, fixed-label only).

---

## Labels

| Label | Sources |
|---|---|
| `MedicalCondition` | MACCROBAT, Corona2 |
| `ClinicalProcedure` | MACCROBAT |
| `ClinicalEvent` | MACCROBAT |
| `MinorChild` | MACCROBAT (Age < 18 + pronoun injection), mydata.csv |
| `GenderIndication` | mydata.csv |

---

## Data Sources

### MACCROBAT (required)
BratStandoff-format clinical case reports. Download and extract the zip — the expected layout is:
```
9764942/
  MACCROBAT2018/   *.txt + *.ann
  MACCROBAT2020/   *.txt + *.ann   ← identical to 2018, deduplicated automatically
```

### Corona2.json (required)
Medical NER dataset in JLiNER export format.  
Source: https://www.kaggle.com/datasets/finalepoch/medical-ner/data  
Place `Corona2.json` next to the training scripts.

### mydata.csv (required)
Internal annotation file with columns:
- `ori_review` — source text
- `minor_col` — semicolon-separated spans marking minor children (including pronouns)
- `gender_col` — semicolon-separated spans marking gender indications
- `medical_col` — rows with this column populated are **excluded** from CSV training to avoid false negative gradients on medical spans

---

## Installation

```bash
# GLiNER backend:
pip install -r requirements-gliner.txt

# SpanMarker backend:
pip install -r requirements-spanmarker.txt

# spaCy backend:
pip install -r requirements-spacy.txt
```

---

## Scripts

### `train_gliner.py` — GLiNER backend

Fine-tunes a GLiNER model on all three data sources in a single pass. The model retains its zero-shot generalization capabilities for entity types outside the training label set.

```bash
python train_gliner.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `EmergentMethods/gliner_medium_news-v2.1` | HuggingFace model ID |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `3e-5` | Learning rate |
| `--output-dir` | `gliner_finetuned` | Output directory |
| `--seed` | `42` | Random seed |
| `--val-split` | `0.1` | Fraction held out for validation |
| `--data-dir` | `9764942/` | Directory containing MACCROBAT subdirs |
| `--corona` | `Corona2.json` | Path to Corona2.json |
| `--csv` | `mydata.csv` | Path to mydata.csv |
| `--minor-oversample` | `2` | Extra copies of MinorChild CSV examples to counter class imbalance |

**Outputs:**
- `gliner_finetuned/best/` — checkpoint with highest validation F1
- `gliner_finetuned/final/` — last epoch checkpoint
- `gliner_finetuned/checkpoint-epoch-N/` — per-epoch checkpoints
- `gliner_finetuned/train.json` / `eval.json` — serialized split data

---

### `train_spanmarker.py` — SpanMarker backend

Fine-tunes a SpanMarker span classifier on all three data sources. Uses direct BIO supervision rather than contrastive learning, which typically yields higher F1 on fixed label sets. Automatically saves the best checkpoint by `eval_overall_f1`.

```bash
python train_spanmarker.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | HuggingFace encoder model ID |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `5e-5` | Learning rate |
| `--output-dir` | `spanmarker_finetuned` | Output directory |
| `--seed` | `42` | Random seed |
| `--val-split` | `0.1` | Fraction held out for validation |
| `--data-dir` | `9764942/` | Directory containing MACCROBAT subdirs |
| `--corona` | `Corona2.json` | Path to Corona2.json |
| `--csv` | `mydata.csv` | Path to mydata.csv |
| `--minor-oversample` | `2` | Extra copies of MinorChild CSV examples to counter class imbalance |
| `--entity-max-length` | `8` | Max tokens per entity span |
| `--model-max-length` | `256` | Max input sequence length |

**Outputs:**
- `spanmarker_finetuned/best/` — checkpoint with highest `eval_overall_f1`
- `spanmarker_finetuned/final/` — last epoch checkpoint

---

### `train_spacy.py` — spaCy backend

Trains a spaCy NER model on all three data sources in a single pass. The output model recognizes **only** the five labels above — base model generalist NER (PERSON, ORG, etc.) is not retained.

```bash
python train_spacy.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `en_core_sci_lg` | spaCy base model (requires scispacy) |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--dropout` | `0.2` | Dropout rate |
| `--output-dir` | `spacy_model` | Output directory |
| `--seed` | `42` | Random seed |
| `--val-split` | `0.1` | Fraction held out for validation |
| `--data-dir` | `9764942/` | Directory containing MACCROBAT subdirs |
| `--corona` | `Corona2.json` | Path to Corona2.json |
| `--csv` | `mydata.csv` | Path to mydata.csv |
| `--minor-oversample` | `2` | Extra copies of MinorChild CSV examples to counter class imbalance |
| `--use-gpu` | off | Enable GPU (CUDA or Apple MPS) |

**Outputs:**
- `spacy_model/best/` — checkpoint with highest validation F1
- `spacy_model/final/` — last epoch checkpoint
- `spacy_model/checkpoint-epoch-N/` — per-epoch checkpoints

---

## Design Notes

### Single-pass training
Both scripts train on MACCROBAT + Corona2 + mydata.csv in one combined pass. Splitting into two rounds risked the model learning that pronouns are *not* MinorChild during round 1 (clinical data only), then having to unlearn that in round 2 — a harder optimization problem.

### MinorChild pronoun injection (MACCROBAT)
Clinical case reports are single-patient documents. When a document contains an `Age < 18` annotation, all third-person singular pronouns (`he`, `she`, `his`, `her`, `him`) and noun phrases (`the child`, `the patient`, `the boy`, etc.) are automatically labeled `MinorChild`. Plural pronouns (`they`, `them`, `their`) are excluded as they typically refer to the medical team or family members.

### MinorChild Sex-span absorption (MACCROBAT)
MACCROBAT annotates age and sex as two separate spans — e.g., `"16-year-old"` (label `Age`) and `"boy"` (label `Sex`). mydata.csv, by contrast, labels the full noun phrase as a single `MinorChild` span (e.g., `"4 year old son"`). To align span boundaries across datasets, `parse_ann` absorbs an immediately adjacent `Sex` span into the `MinorChild` span when the two are separated by pure whitespace (≤ 2 characters). A comma, newline, or any non-whitespace character between them prevents the merge.

### MACCROBAT deduplication
MACCROBAT2018 and MACCROBAT2020 contain identical files. Only MACCROBAT2018 is loaded to avoid training on duplicate documents.

### medical_col exclusion
mydata.csv rows with a non-empty `medical_col` are excluded from training. Including them while only annotating `minor_col`/`gender_col` would produce false negative gradients on medical spans, contradicting MACCROBAT/Corona2 supervision.

### MinorChild oversampling
`MedicalCondition` and `ClinicalProcedure` appear many times per clinical document, while `MinorChild` is rare. `--minor-oversample` (default `2`) duplicates MinorChild-containing CSV examples before training to partially compensate.

### Backend comparison
| | GLiNER | SpanMarker | spaCy |
|---|---|---|---|
| Architecture | Transformer span model | BERT span classifier | CNN / Transformer token classifier |
| Generalist NER after fine-tuning | Retained | Lost | Lost |
| Inference speed | Medium | Medium | Fast |
| F1 on fixed labels | ~0.60 | 0.801 | 0.738 |
| Implicit/subtle entity detection | Better | Better | Weaker |

**spaCy per-label (en_core_sci_lg, F1=0.738 P=0.746 R=0.729):**

| Label | F1 |
|---|---|
| MinorChild | 0.933 |
| GenderIndication | 0.923 |
| ClinicalEvent | 0.708 |
| ClinicalProcedure | 0.673 |
| MedicalCondition | 0.635 |

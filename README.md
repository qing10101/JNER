# JLiNER Label Verifier — Training Scripts

NER training pipeline for detecting **minor children**, **gender indications**, and **biomedical entities** in text. Two model backends are provided: [GLiNER](https://github.com/urchade/GLiNER) (transformer-based, retains zero-shot generalization) and [spaCy](https://spacy.io/) (lighter, faster, fixed-label only).

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
pip install -r requirements.txt

# spaCy backend (choose one):
python -m spacy download en_core_web_lg       # CNN, faster
python -m spacy download en_core_web_trf      # transformer, better for implicit entities
pip install spacy-transformers                 # required for en_core_web_trf

# Optional — better biomedical tokenisation for spaCy:
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
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
| `--model` | `numind/NuNER_Zero-span` | HuggingFace model ID |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-5` | Learning rate |
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

### `train_spacy.py` — spaCy backend

Trains a spaCy NER model on all three data sources in a single pass. The output model recognizes **only** the five labels above — base model generalist NER (PERSON, ORG, etc.) is not retained.

```bash
python train_spacy.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `en_core_web_lg` | spaCy base model |
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

### MACCROBAT deduplication
MACCROBAT2018 and MACCROBAT2020 contain identical files. Only MACCROBAT2018 is loaded to avoid training on duplicate documents.

### medical_col exclusion
mydata.csv rows with a non-empty `medical_col` are excluded from training. Including them while only annotating `minor_col`/`gender_col` would produce false negative gradients on medical spans, contradicting MACCROBAT/Corona2 supervision.

### MinorChild oversampling
`MedicalCondition` and `ClinicalProcedure` appear many times per clinical document, while `MinorChild` is rare. `--minor-oversample` (default `2`) duplicates MinorChild-containing CSV examples before training to partially compensate.

### GLiNER vs. spaCy
| | GLiNER | spaCy |
|---|---|---|
| Architecture | Transformer span model | CNN / Transformer token classifier |
| Generalist NER after fine-tuning | Retained | Lost |
| Inference speed | Slower | Faster |
| Implicit/subtle entity detection | Better | Weaker (use `en_core_web_trf` to improve) |

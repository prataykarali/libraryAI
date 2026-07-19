# Pilot Corpus — CS / ML Department (Archipelago)

**Department:** Computer Science / Machine Learning  
**Purpose:** One curated departmental course corpus for the Archipelago pilot  
**Session:** 8 — Pilot Readiness

---

## Contents

| File | Role in curriculum | Approx. size |
|---|---|---|
| `pdfs/Vaswani2017_Attention_Is_All_You_Need.pdf` | Transformers / attention foundations | ~2.1 MB |
| `pdfs/Devlin2018_BERT.pdf` | Contextual language models | ~0.8 MB |
| `pdfs/Hu2021_LoRA.pdf` | Parameter-efficient fine-tuning | ~1.6 MB |
| `pdfs/Lewis2020_RAG.pdf` | Retrieval-augmented generation | ~0.9 MB |
| `pdfs/Edge2024_GraphRAG.pdf` | Graph-structured retrieval | ~6.9 MB |
| `pdfs/AI_ML_Archipelago_Corpus_Seed.md` | Local syllabus / bridge concepts | small |
| `pdfs/Deisenroth_Math_For_ML.pdf` *(optional, large)* | Math foundations (chs. 1–2 focus) | ~17 MB |

Default pilot set is **6 core items** (5 papers + syllabus seed). Math-for-ML is included in the full set when disk/time allow.

Files under `pilot_corpus/pdfs/` are **symlinks** into the main `pdfs/` tree so we do not duplicate large binaries.

---

## Gold evaluation sets

| Gold file | Focus |
|---|---|
| `gold/gold_lora.json` | LoRA / PEFT concept + edge inventory |
| `gold/gold_attention.json` | Attention / Transformer concept + edge inventory |
| `gold/gold_curriculum.json` | Union curriculum checklist for the pilot (concepts expected across papers) |

Run against the live Kùzu DB:

```bash
.venv/bin/python - <<'PY'
from okf.evaluate import evaluate_pipeline, print_report
import kuzu
conn = kuzu.Connection(kuzu.Database("okf_graph.db"))
for g in [
    "pilot_corpus/gold/gold_lora.json",
    "pilot_corpus/gold/gold_attention.json",
    "pilot_corpus/gold/gold_curriculum.json",
]:
    print("\n###", g)
    print_report(evaluate_pipeline(conn, g))
PY
```

---

## Expected graph scale (after full pilot ingest)

Documented targets for a healthy pilot graph (from prior full-corpus runs; re-measure after `./ingest_pilot_corpus.sh`):

| Metric | Target range | Notes |
|---|---|---|
| Concepts | **80 – 160** | Grounded concepts after cleanup |
| Edges (all relation types) | **80 – 150** | Requires / unlocks / related |
| Orphan rate | **&lt; 60%** | Prefer lower after relation pass |
| Self-loops | **0** | Structural audit must be clean |
| Placeholder / empty-summary nodes | **0** | See `graph_audit.json` |

See `expected_stats.json` for machine-readable targets used by readiness tooling.

---

## Ingestion

From the `libraryAI/` project root:

```bash
./ingest_pilot_corpus.sh
```

This walks `pilot_corpus/pdfs/` (and optional textbook if present) and calls the local pipeline:

```text
.venv/bin/python okf_pipeline.py --add <file> --local
```

**Requirements**

- CUDA venv at `.venv` (preferred) — never system `python3` for GPU ingest  
- Stop `inference_server` before bulk rebuild if it holds the Kùzu lock  
- Disk space for quarantine DBs during atomic swap  

After ingest, restart:

```bash
.venv/bin/python graph_server.py      # :5050
.venv/bin/python inference_server.py  # :5051
.venv/bin/python chat_server.py       # :5052
```

---

## License / use note

Papers are standard open-access research PDFs already present under `pdfs/`. The pilot corpus is for **local educational use** within the department. Do not redistribute the binary PDFs outside institutional policy.

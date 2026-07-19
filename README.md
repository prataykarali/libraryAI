# Archipelago: PDF → OKF → KuzuDB Knowledge Graph

A local-first, domain-agnostic knowledge graph pipeline. It extracts structured concepts from documents (PDFs, Markdown, plain text), canonicalises them, and ingests them into a KuzuDB graph database with full provenance tracking.

> **Domain note:** The repo is seeded with AI/ML papers only as a test corpus. The code, schema, prompts, and graph UI are generic and work with any subject.

## Architecture

```
Documents (PDF / MD / TXT)
        |
   [1] Section-Aware Chunking (archipelago.ingestion / pdf_ingestion.py shim)
        |
   [2] OKF Extraction (okf.* / okf_pipeline.py shim, local SLM / Ollama)
        |
   [2b] Post-Extraction Cleanup (okf.cleanup_parts: grounding, dedupe, cycles)
        |
   [3] Entity Canonicalization (okf.canonicalize)
        |
   [4] KuzuDB MERGE Ingestion (okf.graph)
        |
   [5] Accuracy Evaluation (okf.eval)
        |
   Outputs: okf_results.json, okf_graph.json, accuracy.json
        |
   [6] Inference / Chat RAG (archipelago.inference) + Graph UI (ui/graph)
```

Feature packages and dependency rules: **[docs/guides/ARCHITECTURE.md](docs/guides/ARCHITECTURE.md)**.

## OKF v1.6 Schema

Each concept extracted from a document contains:

| Field | Type | Description |
|-------|------|-------------|
| `concept_name` | string | Short noun phrase (aim for ≤5 words) |
| `concept_type` | enum | `method`, `metric`, `technique`, `theory`, `tool`, `dataset`, `result`, `definition` |
| `difficulty` | enum | `foundational`, `intermediate`, `advanced`, `expert` |
| `summary` | string | 1–2 sentence description |
| `prerequisites` | list[str] | Concepts needed BEFORE this one |
| `unlocks` | list[str] | Concepts this one ENABLES |
| `related_to` | list[obj] | `{concept, relation}` where relation is `uses`, `extends`, `contrasts_with`, `evaluated_by`, `variant_of`, `part_of` |
| `tags` | list[str] | Keyword tags (lowercase, hyphenated) |

Provenance fields are attached by the pipeline and must not be emitted by the model:

| Field | Description |
|-------|-------------|
| `doc_id` | Source document path |
| `chunk_id` | Chunk identifier |
| `page_number` | Source page number |
| `section_title` | Source section heading |
| `source_category` | `paper`, `textbook`, `markdown`, etc. |
| `source_passage` | The exact chunk text that produced the node |

## Setup

### Prerequisites

- Python 3.10+
- Ollama running locally with any small Instruct model (default: `qwen3.5:0.8b`)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pull a model (if using Ollama)

```bash
ollama pull qwen3.5:0.8b
```

Or place a local GGUF model in `models/`.

## Usage

### Run the full pipeline

```bash
# Process all files in pdfs/
python okf_pipeline.py

# Process a specific file
python okf_pipeline.py path/to/document.pdf

# Resume from saved results (skip re-extraction)
python okf_pipeline.py --resume
```

### Supported input formats

- `.pdf` - Section-aware chunking via PyMuPDF
- `.md` / `.markdown` - Split by headings
- `.txt` / `.text` - Paragraph-based chunking

### Add more documents

Drop any PDF, Markdown, or text file into `pdfs/` or one of its subfolders and re-run the pipeline. Concepts that resolve to the same canonical name across documents automatically share a node in the graph (cross-document bridges).

## Output Files

| File | Description |
|------|-------------|
| `okf_results.json` | All extracted OKF concepts with provenance |
| `okf_graph.json` | Full graph export (nodes + edges) |
| `accuracy.json` | Accuracy scores and breakdown |
| `okf_graph.db` | KuzuDB database file |

## Servers

**Pilot launch (one command):** `./scripts/ops/start_pilot.sh` — starts all
three services, runs the 7-query demo gate, and prints the pilot URL table.
Full pilot scope, safety defaults, and demo script: **[PILOT_LAUNCH.md](PILOT_LAUNCH.md)**.

```bash
python graph_server.py                          # Graph UI + API :5050
python inference_server.py                      # Chat / RAG API :5051
# or: python -m archipelago.apps.inference_app
python chat_server.py                           # Chat static UI :5052
```

Static UI lives under `ui/chat` and `ui/graph`. Compat symlinks `chat_ui` →
`ui/chat` and `graph_ui` → `ui/graph` keep existing server paths working.

### Known pilot caveats

- Citations can still be imperfect on already-ingested chunks until re-ingest
  (bibliography-skip + dominant-page fixes are in code; live data predates them).
- Agent/LangChain nodes are seeded, not extracted from full papers.
- Graph has ~450 concepts from a small source set — sparse for niche topics.
- Curriculum answers come from graph edges, not a real teacher.

## Training data

Scripts live under `training/`:

```bash
python training/prepare_okf_training_data.py
python training/split_okf_dataset.py
```

Output under `training_data/` (jsonl train/test pairs + reports). Splits are
chunk-held-out so source chunks do not leak between train and test.

See [docs/guides/FINE_TUNING_GUIDE.md](docs/guides/FINE_TUNING_GUIDE.md).

## Demo (Mock Data)

```bash
python okf_kuzu_graph.py    # Build graph from mock_data.py
python chat.py               # Interactive chat with the local model
pytest tests/                # Unit + integration suite
```

## Project Structure

```
libraryAI/
  archipelago/
    inference/          # chat RAG ranking curriculum citations synthesis
    apps/               # process entrypoints (inference_app)
    ingestion/          # PDF / MD / TXT / DOCX chunking
  okf/
    graph/              # Kùzu ingest / export / evidence
    cleanup_parts/      # grounding, dedupe, cycles
    eval/               # metrics, gold, structural audit
  docs/
    guides/             # ARCHITECTURE, OKF_SPEC, fine-tuning, privacy
    reports/            # audits, handoffs, pilot notes
  training/             # dataset + fine-tune scripts
  scripts/ops/          # bulk ingest, rebuild_graph, tidy_root, …
  ui/chat, ui/graph     # canonical static UIs
  chat_ui → ui/chat     # compat symlink
  graph_ui → ui/graph   # compat symlink
  tests/                # unit / integration / e2e
  pdfs/                 # drop documents here
  pilot_corpus/         # frozen pilot + gold
  jobs/                 # live upload quarantine
  data/                 # optional logs / artifacts staging

  # Thin root shims (keep for imports / CLI)
  okf_pipeline.py       # → okf.*
  pdf_ingestion.py      # → archipelago.ingestion.*
  inference_server.py   # → archipelago.inference.*

  # Small entrypoints (not shims)
  chat_server.py, graph_server.py
  ingestion_jobs.py, ingestion_worker.py

  # Legacy / demo (do not grow)
  okf_extraction.py, okf_kuzu_graph.py, mock_data.py, chat.py
```

Full dependency rules and root-clutter policy:
[docs/guides/ARCHITECTURE.md](docs/guides/ARCHITECTURE.md).

### Root clutter

- **Logs:** move with `bash scripts/ops/tidy_root.sh` → `data/logs/` (safe).
- **Backups** (`*.bak*`, `*.thin.*`, `*.pre_sync_*`): keep; archive under
  `data/backups/` only after checking scripts/docs. Do not delete casually.
- **Live graph data** (`okf_results.json`, `okf_graph.json`, `okf_graph.db`):
  required runtime artifacts at `BASE_DIR`.

## Configuration

Primary constants live in `okf/config.py` (re-exported via `okf_pipeline`):

- `MODEL_NAME` — Ollama model name
- `MAX_RETRIES` — retry count for failed extractions
- `ALIAS_MAP` — concept aliases (keep empty for domain-agnostic mode)

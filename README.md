# Archipelago: PDF -> OKF -> KuzuDB Knowledge Graph

A local-first knowledge graph pipeline that extracts structured concepts from documents (PDFs, Markdown, plain text), canonicalizes them, and ingests them into a KuzuDB graph database with full provenance tracking.

## Architecture

```
Documents (PDF/MD/TXT)
        |
   [1] Section-Aware Chunking (pdf_ingestion.py)
        |
   [2] OKF v1.5 Extraction (okf_pipeline.py, via Qwen 3.5 0.8B / Ollama)
        |
  [2b] Post-Extraction Cleanup (reference filter, self-loop removal, dedup)
        |
   [3] Entity Canonicalization (alias resolution, fuzzy dedup)
        |
   [4] KuzuDB MERGE Ingestion (no duplicate nodes across documents)
        |
   [5] Accuracy Evaluation (proxy metrics)
        |
   Outputs: okf_results.json, okf_graph.json, accuracy.json
```

## OKF v1.5 Schema

Each concept extracted from a document contains:

| Field | Type | Description |
|-------|------|-------------|
| `concept_name` | string | Short noun phrase (max 5 words) |
| `concept_type` | enum | `method`, `metric`, `technique`, `theory`, `tool`, `dataset`, `result`, `definition` |
| `difficulty` | enum | `foundational`, `intermediate`, `advanced`, `expert` |
| `summary` | string | 1-2 sentence description |
| `prerequisites` | list[str] | Concepts needed BEFORE this one |
| `unlocks` | list[str] | Concepts enabled AFTER learning this |
| `related_to` | list[obj] | `{concept, relation}` where relation is `uses`, `extends`, `contrasts_with`, `evaluated_by`, `variant_of`, `part_of` |
| `tags` | list[str] | Keyword tags (lowercase, hyphenated) |
| `doc_id` | string | Source document filename |
| `chunk_id` | string | Chunk identifier |
| `page_number` | int | Source page number |
| `section_title` | string | Source section heading |

## Setup

### Prerequisites
- Python 3.10+
- Ollama running locally with `qwen3.5:0.8b` model

### Install
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### Pull the model (if using Ollama)
```bash
ollama pull qwen3.5:0.8b
```

Or place a local GGUF model in `models/qwen3.5-0.8b.gguf`.

## Usage

### Run the full pipeline
```bash
# Process all files in pdfs/ folder
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
Drop any PDF, Markdown, or text file into `pdfs/` and re-run the pipeline. Concepts that resolve to the same canonical name across documents automatically share a node in the graph (cross-document bridges).

## Output Files

| File | Description |
|------|-------------|
| `okf_results.json` | All extracted OKF v1.5 concepts with provenance |
| `okf_graph.json` | Full graph export (nodes + edges) |
| `accuracy.json` | Accuracy scores and breakdown |
| `okf_graph.db` | KuzuDB database file |

## Demo (Mock Data)

The original mock data demo still works:
```bash
python okf_kuzu_graph.py    # Build graph from mock_data.py
python test_okf.py           # Test extraction on mock chunks
python test_relationships.py # Test relationship detection
python chat.py               # Interactive chat with Qwen 3.5
```

## Project Structure

```
libraryAI/
  okf_pipeline.py       # Main pipeline: PDF -> OKF v1.5 -> KuzuDB
  pdf_ingestion.py       # Section-aware document chunker
  okf_extraction.py      # Original OKF v1.0 extraction (legacy)
  okf_kuzu_graph.py      # Original KuzuDB demo (mock data)
  chat.py                # Interactive Qwen 3.5 chat
  mock_data.py           # Mock text chunks for testing
  test_okf.py            # Extraction tests
  test_okf_graph.py      # Graph relationship tests
  test_relationships.py  # Cross-reference tests
  requirements.txt       # Python dependencies
  pdfs/                  # Drop documents here
  models/                # Local GGUF model files
```

## Configuration

Edit the top of `okf_pipeline.py`:
- `MODEL_NAME` - Ollama model name (default: `qwen3.5:0.8b`)
- `MAX_RETRIES` - Retry count for failed extractions (default: 2)
- `ALIAS_MAP` - Custom concept name aliases for canonicalization

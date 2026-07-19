## TODO: PDF -> OKF -> Kuzu graph + accuracy validator

> **STATUS (2026-07-16):** This file is a historical build log. For the current
> state and prioritized flaw list see `FULL_AUDIT_REPORT.md`.
>
> **Done:** structure-aware chunking, OKF extraction + cleanup, Kùzu MERGE
> ingestion, chat UI (:5052), graph UI (:5050), inference server (:5051,
> embedder + graph traversal + synthesis + citations), pytest suite
> (unit/integration/e2e), pilot corpus + gold sets + readiness ops.
>
> **Left:** graph export unification (root okf_graph.json has 0 edges vs
> graph_ui's 125 — in progress this session), soft-anchor quality, multi-book
> curriculum answers, v4 model train/deploy (deferred), auth hardening beyond
> localhost bind + optional token.

### Step 1 -- Add PDF ingestion + chunking
- [x] Implement `pdf_ingestion.py`:
  - [x] Walk `pdfs/` directory
  - [x] Extract text with page numbers (via PyMuPDF)
  - [x] Create chunking using heading/page heuristics
  - [x] Output list of chunks: `{doc_id, chunk_id, page_number, text}`
  - [x] Support Markdown and plain text files too

### Step 2 -- OKF v1.5 extraction + canonicalization + MERGE ingestion
- [x] Implement `okf_pipeline.py`:
  - [x] Run `extract_okf_v15(text)` per chunk with expanded schema
  - [x] New OKF v1.5 fields: concept_type, difficulty, related_to, tags
  - [x] Canonicalize concept names (alias map + fuzzy dedup)
  - [x] Stable concept key/id generation
  - [x] Ingest into Kuzu using deterministic keys + MERGE semantics
  - [x] Store provenance metadata on nodes/edges (doc_id, chunk_id, page)
  - [x] Post-extraction cleanup (self-loop removal, reference filtering, dedup)
  - [x] --resume flag to skip re-extraction

### Step 3 -- Keep original graph builder as demo
- [x] Keep `okf_kuzu_graph.py` as demo (works with mock_data.py)
- [x] New pipeline in `okf_pipeline.py` handles real PDFs

### Step 4 -- Accuracy scoring
- [x] Accuracy evaluation built into `okf_pipeline.py`:
  - [x] JSON validity + schema completeness
  - [x] Relation consistency (prereqs/unlocks map to known canonical nodes)
  - [x] Cycle/self-loop detection
  - [x] Concept quality (name length, format)
  - [x] Graph connectivity + orphan detection
  - [x] Composite weighted score

### Step 5 -- Outputs
- [x] Pipeline writes:
  - [x] `okf_results.json` (extracted per chunk OKF v1.5 w/ provenance)
  - [x] `okf_graph.json` (exported graph structure)
  - [x] `accuracy.json` (scores + breakdown)
- [x] Update `README.md` with full pipeline docs

### Step 6 -- Run + verify
- [x] Run pipeline on `pdfs/probable.pdf`
- [x] Confirm `accuracy.json` and `okf_graph.json` connections
- [ ] Run existing tests to verify no regressions

### Step 6.5 -- Ingestion test set (6 real docs) -- DONE (quality caveats)
- [x] Download 6 open-access sources into `pdfs/` tree:
  - [x] `textbooks/Deisenroth_Math_For_ML.pdf` (full book; only ch.1-2 needed)
  - [x] `papers/Vaswani2017_Attention_Is_All_You_Need.pdf`
  - [x] `papers/Devlin2018_BERT.pdf`
  - [x] `papers/Hu2021_LoRA.pdf`
  - [x] `papers/Lewis2020_RAG.pdf`
  - [x] `papers/Edge2024_GraphRAG.pdf`
- [x] Design OKF v1.6 schema + extraction skill -> `OKF_SPEC.md`
- [x] Model confirmed: `qwen3.5:0.8b` is a real tag, pulled (1.0 GB). `MODEL_NAME` already correct.
- [x] Start Ollama (`ollama serve`) — running, v0.30.10
- [x] Smoke-test: raw model extracts valid JSON (5 concepts, ~15s/chunk)
- [ ] Add the few-shot example + `temperature=0.1` to the extraction prompt (see OKF_SPEC.md §3, §5)
- [ ] (optional) Limit Math-for-ML to pages 1-2's chapters so the book doesn't dominate the run
- [x] Run `python okf_pipeline.py` on the 6 docs; inspect `okf_results.json`
- [x] Cross-doc bridges checked — exist but sparse (~7/125 edges; see FULL_AUDIT_REPORT.md P0.3)
- [x] Record baseline `accuracy.json` (proxy overall 64.8%)

### Step 7 -- Inference half -- DONE
- [x] Embedder for query->concept matching (Snowflake Arctic Embed, `inference_server.py`)
- [x] Graph traversal (Cypher REQUIRES/UNLOCKS neighborhood, k≈2)
- [x] Context assembly for LLM
- [x] Natural language response generation (template + optional Ollama wording)
- [x] Chat interface integration (chat UI :5052 → inference API :5051)

### Step 8 -- Fine-tuning -- DATA DONE, TRAIN/DEPLOY DEFERRED
- [x] Generate training pairs from pipeline output + manual corrections (v4: 95 quality pairs)
- [ ] LoRA fine-tune Qwen 3.5 0.8B (v4 train + deploy — deferred)
- [ ] Export to GGUF, drop into models/
- [ ] Re-run pipeline, measure accuracy improvement

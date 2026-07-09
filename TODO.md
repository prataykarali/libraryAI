## TODO: PDF -> OKF -> Kuzu graph + accuracy validator

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

### Step 7 -- Inference half (NOT STARTED)
- [ ] Embedder for query->concept matching
- [ ] Graph traversal (BFS on prereqs/unlocks)
- [ ] Context assembly for LLM
- [ ] Natural language response generation
- [ ] Chat interface integration

### Step 8 -- Fine-tuning (NOT STARTED)
- [ ] Generate 200+ training pairs from pipeline output + manual corrections
- [ ] LoRA fine-tune Qwen 3.5 0.8B on Colab
- [ ] Export to GGUF, drop into models/
- [ ] Re-run pipeline, measure accuracy improvement

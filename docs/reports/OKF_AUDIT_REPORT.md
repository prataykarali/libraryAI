# OKF Pipeline Evaluation & Upgrade Report

**Date:** 2026-07-12  
**Pipeline:** `okf_pipeline.py` (OKF v1.6)  
**Model:** local SLM (default `qwen3.5:0.8b`, overridable via `OKF_MODEL_NAME`)  
**Inputs audited:** 7 source documents (current corpus seeded with AI/ML papers as a test set)  
**Output audited:** `okf_graph.json`, `okf_results.json`, `accuracy.json`

> **Domain-agnostic note:** The pipeline, schema, prompts, UI and canonicaliser have been made domain-agnostic. The only AI/ML-specific content left is the test corpus itself (the files dropped into `pdfs/`). See `DOMAIN_NOTE.md`.

> **Freshness note:** The graph-integrity figures in this report (invalid/metadata nodes, empty-summary nodes, edge-placeholder nodes, self-loops, reciprocal `REQUIRES` cycles) reflect the audit as of the current `graph_audit.json`. That file now reports **0** for every one of these categories:
> `invalid_nodes = 0`, `empty_summary_nodes = 0`, `placeholder_nodes = 0`, `self_edges = 0`, `reciprocal_requires = 0`.
> The narrative examples below (backward edges, alias variants, metadata-node samples) are retained as *upstream failure modes the runtime cleanup now contains*, not as counts still present in the current graph. The concept-type and difficulty distributions in §2 are **not** part of `graph_audit.json` and have not been re-verified against the current run.

---

## 0. Domain-agnostic cleanup

- `OKF_EXTRACTOR_PROMPT` / `OKF_SPEC.md` now use a generic *Scientific Method / Peer Review* example instead of Transformer/Self-Attention.
- `ALIAS_MAP` in `okf_pipeline.py` is empty by default; the pipeline can be pointed at any domain without carrying AI/ML aliases.
- `_FOUNDATIONAL_PRIORITY` is empty by default; direct 2-node cycles are resolved with a generic difficulty heuristic (`foundational < intermediate < advanced < expert`).
- `graph_server.py` no longer reports hard-coded node/edge counts or model brand.
- `graph_ui/index.html` schema panel no longer references the extraction model by name.
- `README.md` and `DOMAIN_NOTE.md` explain that the repo is domain-agnostic and the AI/ML corpus is only a seed.

---

## 1. Graph UI — Animated graph disabled

- The canvas/holographic viewer (`graph_ui/neon.html`) has been disabled in `graph_server.py`. Both `/neon` and `/neon.html` now return HTTP 410 with a message directing users to the standard D3/SVG graph.
- All visible graph nodes in the live UI come from `/api/graph`, which reads `okf_graph.json` → `visualization.nodes`. No graph nodes are rendered from hard-coded HTML.
- The Standard Graph (`graph_ui/index.html`) is kept as the canonical viewer.
- `/archipelago_graph.html` still responds but now serves the live data viewer; the old static snapshot with hard-coded nodes is no longer served.
- `/neon` and `/neon.html` return 410 Gone.

---

## 2. OKF v1.6 Parameter & Schema Verification

The schema enforced by the pipeline is structurally valid.

| Field | Type | Values / Rule | Status | Notes |
|---|---|---|---|---|
| `concept_name` | string | ≤5 words, Title Case, reusable noun phrase | Valid | `concept_quality` = 99.7% in `accuracy.json`; a few long names escaped cleanup |
| `concept_type` | enum | `method`, `metric`, `technique`, `theory`, `tool`, `dataset`, `result`, `definition` | Valid | 0 invalid enum values in `okf_results.json` |
| `difficulty` | enum | `foundational`, `intermediate`, `advanced`, `expert` | Valid | 0 invalid enum values |
| `summary` | string | 1–2 sentences, definition-focused | Valid | Current `graph_audit.json` reports 0 empty-summary nodes and 0 edge-placeholder nodes (previously 191); placeholder targets are now backfilled by `summary_by_name` |
| `prerequisites` | list[str] | → `REQUIRES` edges | Warning | Direction is unreliable; model uses it as “related concepts” too often |
| `unlocks` | list[str] | → `UNLOCKS` edges | Warning | Frequently empty (forward dependencies are hard from local chunks) |
| `related_to` | list[obj] | `{concept, relation}` with 7 valid relations | Valid | Normalized to `uses` if relation invalid |
| `tags` | list[str] | lowercase-hyphenated | Valid | Cleaned by pipeline |

**Verdict:** The schema parameters are correct; the **values inside relations** (edge direction, alias consistency, semantic correctness) need strict runtime containment and fine-tuning.

### Distribution of extracted concept types

```
method      169
        technique   107
metric       52
dataset      27
result       22
definition   13
theory        8
tool          6
```

### Difficulty distribution

```
intermediate  213
advanced      185
expert          5
foundational    1
```

The `foundational` bucket is almost empty even for basic linear-algebra chunks, which suggests the model is biased toward `intermediate`/`advanced`.

---

## 3. Where the Qwen 0.8B SLM Failed

The current run has **zero self-loops** after pipeline cleanup, but the upstream failure modes are still visible in edge topology and source-page verification.

### A. Non-concept / metadata nodes

The extraction engine repeatedly treats document metadata and table labels as concepts.

Examples found in `okf_graph.json`:

| Bad node | Why it is wrong | Source |
|---|---|---|
| `Author Name` | Not a teachable concept | `papers/Edge2024_GraphRAG.pdf:chunk_045:p12` |
| `Contributor Name` | Not a teachable concept | `papers/Edge2024_GraphRAG.pdf:chunk_045:p12` |
| `Authors` / `Contributors` | Bibliographic metadata | `papers/Edge2024_GraphRAG.pdf:chunk_045:p12` |
| `Best Model Without Gold Access Underlined` | Table caption / formatting artifact | `papers/Lewis2020_RAG.pdf:chunk_019:p6` |
| `Canada CIFAR AI Chairs program` / `Canada Research Chair …` | Funding/affiliation text | `probable.pdf:chunk_028:p10` |

Root cause: the small model cannot reliably distinguish prose definitions from captions, front-matter and acknowledgements, despite `chunk_kind` filtering.

### B. Numeric / metric artifacts

- `15% of Tokens in Batch` — a parameter value, not a concept.
- `Model Size Comparison (BERTBASE vs GPT)` — a comparison phrase, not a stable concept.
- `Feed-Forward/Filter Size (4H)` — an architectural detail, not a node.

Root cause: the prompt allows ≤5 words, so the model packages values and comparisons into pseudo-concepts.

### C. Alias redundancy and over-canonicalization

Raw extraction contains variants such as:

- `Transformer` vs `Transformer Architecture` vs `Transformer Model`
- `Self-Attention` vs `Self-Attention Mechanism` vs `Self-Attention Module` vs `Self-Attention Head`
- `BERT` vs `Bert` vs `BERTBASE` vs `Bertbase`
- `Masked Language Modeling (LM)` vs `Masked Language Modeling`
- `Next Sentence Prediction` vs `Next Sentence Prediction (NSP)`

The pipeline substring canonicalizer collapses *too much*: e.g. “Transformer Encoder” was merged into a generic “Encoder” node, losing semantic specificity needed for a knowledge graph.

### D. Edge-direction confusion and circular dependencies

The graph is not a DAG. Many prerequisite relations run backwards:

```
Self-Attention  REQUIRES  Transformer
Transformer     REQUIRES  Self-Attention
Attention       REQUIRES  Transformer
Transformer     REQUIRES  Attention
```

This means the model uses `prerequisites` as “related concepts” rather than strict “must-know-first” dependencies. Result: a learner following `REQUIRES` edges will loop instead of progressing.

Also, highly questionable dependencies exist:

```
Bert  REQUIRES  LoRA        # BERT does not require LoRA
LoRA  REQUIRES  Bert        # plausible as an application, not prerequisite
```

### E. Empty summary placeholders

Historically, `okf_graph.json` contained nodes that appeared only as prerequisites/unlocks/related targets from other chunks. They had no own extraction record, so the graph stored only the name, which hurt the flashcard UX.

**Current status:** `graph_audit.json` now reports **0 empty-summary nodes and 0 edge-placeholder nodes** (down from the 191 reported in an earlier run). The `summary_by_name` backfill fills placeholder nodes from a summary extracted elsewhere, so this failure mode is currently contained.

### F. Cross-document bridges are weak

Target chains from `OKF_SPEC.md`:

```
Attention Mechanism → Self-Attention → Transformer → BERT → Fine-Tuning → LoRA
Transformer → RAG → GraphRAG
```

Actual state:

- Core attention/transformer/BERT concepts exist (`attention_mechanism`, `self_attention`, `transformer`, `bert`, `lora`).
- `RAG` node does not exist; only `retrieval_augmented_generation` exists.
- `graph_rag` does not exist; only `graphrag` exists.
- `Dense Passage Retrieval` does not exist.

So the semantic bridges are *partially* present but node naming is inconsistent, and directed prerequisite chains are unreliable.

---

## 4. Did it Extract Correctly from the Docs? (Source audit)

A source-page audit (`source_audit.json`) was run, comparing extracted concept names against the text of the pages they cite.

**Summary of sampled pages:**

| Source | Good extraction | Main problems |
|---|---|---|
| Deisenroth *Math for ML* | Foundational terms are mostly captured when they appear in prose | Graph/math-heavy pages produce noisy or empty outputs; almost no `foundational` difficulty tags |
| Vaswani “Attention Is All You Need” | `Self-Attention`, `Multi-Head Attention`, `Positional Encoding`, `Transformer` present | Direction of prerequisite edges is reversed; many table/figure-derived artifacts |
| Devlin BERT | `BERT`, `Masked Language Modeling`, `Next Sentence Prediction`, ` CLS Token` present | BERT/GPT comparison table captions become nodes; over-splitting of encoder/attention variants |
| Hu LoRA | `LoRA`, `Low-Rank Adaptation`, parameter-efficiency terms present | Math appendix (SVD, Frobenius norm, Grassmann distance) is mined for fake concepts; “Su- superGLUE” parsing error |
| Lewis RAG | `Retrieval-Augmented Generation`, `Seq2Seq`, retriever vocabulary present | Re-ranking/decoding artifacts; missing explicit `RAG` acronym node |
| Edge GraphRAG | `GraphRAG`, `Community Detection`, `Community Summary` present | Author/acknowledgement section leaked concepts; missing `Knowledge Graph` bridge |
| Syllabus seed | Broad AI concepts captured | Includes out-of-scope concepts (Mixture-of-Experts, GANs) with weak source grounding |

**Bottom line:** The model is good at spotting *what a paragraph is about* but bad at:
1. Deciding if something is a real, reusable concept.
2. Respecting prerequisite directionality.
3. Connecting variants and acronyms (`RAG` ↔ `Retrieval-Augmented Generation`).

---

## 5. Provenance upgrade: Page numbers + highlighted source passage

### 5.1 Schema change

The pipeline already stores provenance metadata (`doc_id`, `chunk_id`, `page_number`, `section_title`). The missing piece is the exact highlighted snippet.

`okf_pipeline.py` has been updated so every extraction result now carries:

```jsonc
{
  "concept_name": "Low-Rank Adaptation",
  "...semantic fields...": "...",
  "doc_id": "papers/Hu2021_LoRA.pdf",
  "chunk_id": "chunk_005",
  "page_number": 1,
  "section_title": "Introduction",
  "source_passage": "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices..."
}
```

The model is **not** asked to emit `source_passage`; it is added by the pipeline from the original chunk text.

### 5.2 UI change

`graph_ui/index.html` now renders the highlighted passage in the flashcard under each source entry. The flashcard shows:

- Document file name
- Chunk id, page number, section title
- The exact source passage (`source_passage`) that produced the node

This gives users the “Read Highlighted Section” behaviour without requiring a PDF renderer.

### 5.3 Future: clickable deep-link into PDF

To jump the user directly to the highlighted area:

1. Store a bounding box during PDF ingestion (PyMuPDF can give text rectangles).
2. Serve the source PDF at `/sources/<path>`.
3. In the UI, build a URL such as `/viewer?file=...&page=N&search=<text>` using PDF.js or pdf-highlight-js.

This is a follow-up feature; the current change provides the data layer (`page_number` + `source_passage`) needed for it.

---

## 6. Fine-tuning dataset prepared

A 597-case cleaned training set has been generated:

```
training_data/
  okf_training_pairs.jsonl   # Alpaca-format JSONL
  okf_training_pairs.json    # Human-reviewable pretty JSON
  okf_dataset_report.json    # Generation counts + discarded examples
  okf_train_pairs.jsonl      # Train split
  okf_test_pairs.jsonl       # Chunk-held-out test split
```

Composition:

| Split | Count | Purpose |
|---|---|---|
| Chunk-level multi-concept examples | 80 | Teaches the real task: extract 1–5 concepts from a paragraph |
| Single-concept snippet examples | 288 | Dense supervision around grounded mentions of a concept |
| Empty-response examples (math/refs/tables) | 229 | Teaches the model to return `[]` for non-prose |
| **Total** | **597** | |

Train/test split:

| Split | Count |
|---|---:|
| Train rows | 505 |
| Test rows | 92 |
| Train chunks | 263 |
| Test chunks | 46 |

The split is chunk-held-out: exact `(doc_id, chunk_id)` examples do not cross train/test, but the same source document can appear in both splits. Treat the test set as a schema/format validation set, not a true held-out-document benchmark.

### Cleanups applied to every training target

- Invalid enums fixed or dropped.
- Self-loops removed.
- Author/chair/grant metadata nodes removed.
- Numeric artifacts and table-caption pseudo-concepts removed.
- Reference targets are scrubbed so junk concepts do not appear in `prerequisites`/`unlocks`/`related_to`.
- Lightweight acronym aliases applied (`bert` → `BERT`, `lora` → `LoRA`, `rag` → `RAG`, etc.).
- Positive examples are emitted only from prose chunks; non-prose chunks become `[]` examples.
- Single-concept snippet examples are kept only when the concept or known alias is grounded in the prompt text.

The dataset still requires **manual review** to fix:

- Reversed prerequisite/unlock directions.
- Semantically weak but schema-valid summaries inherited from the source extraction.
- Concepts that are technically grounded but too generic for your final graph ontology.

Suggested training recipe: follow `FINE_TUNING_GUIDE.md` (Unsloth LoRA, Qwen 2.5 0.5B–1.5B, `r=16`, `lora_alpha=32`, loss only on the JSON output tokens).

---

## 7. Code fixes applied (not only fine-tuning)

Fine-tuning alone will **not** fix every problem. The following runtime fixes are now in the code so the pipeline is usable before and after fine-tuning:

| Problem | Fix |
|---|---|
| Author / grant / chair metadata nodes leaked into graph | `_JUNK_NAME_RE` + `is_valid_concept_name` filter in extraction + post-cleanup |
| Over-aggressive substring canonicalisation collapsing distinct concepts | `build_canonical_map` now uses `thefuzz` ratio ≥90; `ALIAS_MAP` is empty by default |
| AI/ML-specific hardcoding in prompt/spec | Generic *Scientific Method / Peer Review* example; `OKF_SPEC.md` rewritten |
| Reciprocal prerequisite cycles | `_resolve_reciprocal_cycles` resolves 2-node cycles via optional priority list or generic difficulty heuristic |
| Empty placeholder summaries | `summary_by_name` lookup fills placeholder nodes when a summary exists elsewhere |
| Hard-coded node/edge counts in server | `graph_server.py` comments and schema endpoint made generic/dynamic |

## 8. Immediate action check-list

1. ✅ Animated graph disabled; standard graph retained.
2. ✅ Live UI loads nodes from `/api/graph`; `/archipelago_graph.html` now serves the live viewer too.
3. ✅ No hard-coded example nodes in the active UI; the old static snapshot is no longer served.
4. ✅ `source_passage` provenance added to extraction output and UI.
5. ✅ Schema parameters verified; pipeline made domain-agnostic.
6. ✅ 597-case cleaned training set prepared (`training_data/okf_training_pairs.jsonl`).
7. ✅ Reciprocal-cycle breaker and junk filter added to runtime.
8. ⏳ Re-run `python okf_pipeline.py` to populate `source_passage` in `okf_results.json` / `okf_graph.json`.
9. ⏳ Fine-tune any local SLM on `training_data/okf_train_pairs.jsonl`, evaluate on `training_data/okf_test_pairs.jsonl`, and re-run the pipeline to measure improvement.

---

## 8. Auditor verdict

**Is the current map valid?** Partially.

- The graph contains the expected high-level concepts from the source papers.
- The OKF schema is valid.
- Directional prerequisite edges historically formed cycles and backward arrows; the current `graph_audit.json` reports 0 self-loops and 0 reciprocal `REQUIRES` cycles after runtime cleanup, though semantic direction still warrants manual review.
- Alias/deduplication is too aggressive in places and too weak in others.
- In the current run, `graph_audit.json` reports **0 invalid/metadata nodes and 0 empty-summary nodes** (earlier runs showed ~10 metadata nodes and 191 empty-summary nodes); the junk-name filter and summary backfill now contain these.

**Recommendation:** Use the current output as training material, not as a production knowledge map. After one fine-tuning iteration focused on edge direction, canonical naming and empty-reference suppression, re-audit and re-run.

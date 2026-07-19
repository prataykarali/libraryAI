# Archipelago — Full Multi-Agent Audit Report

**Date:** 2026-07-16 (updated after true **5/5** subagent completion)  
**Repo:** `/home/pratay-karali/Desktop/libraryAI/libraryAI`  
**Method:** Five specialized audits (3 ingestion/graph/ops + 2 inference/UX).  
**Agents (all completed as real subagents):**

| ID | Role | Status |
|----|------|--------|
| **G1** | OKF extraction, chunking, cleanup, training | **Completed** |
| **G2** | Kùzu topology, edges, export consistency | **Completed** (re-run after rate limit) |
| **G3** | Corpus, pilot ops, sessions, privacy, tests | **Completed** |
| **I1** | Retrieval, routing, synthesis, live chat | **Completed** (re-run after rate limit) |
| **I2** | UI/UX vs product vision | **Completed** |

**Note:** First attempt: G2/I1 rate-limited. Second attempt: both finished cleanly. I1 live probes reported **services DOWN** during re-run; earlier smoke + on-disk graph stats still apply.

---

## Done vs left (today) — headline numbers

| Scope | **Done** | **Left** | Notes |
|-------|---------:|---------:|-------|
| **Full Archipelago vision** (topology RAG + curriculum GPS + future OS) | **~52%** | **~48%** | Plumbing strong; bridges/edges/pedagogy weak |
| **Local departmental pilot MVP** (chat + graph + PDFs + tests + privacy) | **~70%** | **~30%** | Demo-able; quality & export consistency conditional |
| **Research “Automated Roadmap Maker” promise** | **~40%** | **~60%** | Needs real multi-hop edges + answer format |
| **Future OS** (agents, multimodal, enterprise, spatial) | **~10%** | **~90%** | Specs/scratch only |

### Done (working today)
- PDF/MD chunking with pages, sections, kinds  
- OKF extract + cleanup + Kùzu MERGE codepath  
- Chat UI + graph UI + inference API (when processes running)  
- Embedder ranking + soft/strong/chitchat routing  
- Citations with `doc` + `page` + `#page=` links (mainly right rail)  
- Upload jobs + atomic swap design  
- Large pytest suite + pilot readiness package  
- v4 training **dataset** built  

### Left (blocks “full Archipelago”)
- **Unify graph exports** (root 0 edges vs UI 125 edges) — **#1 blocker**  
- Dense, live **cross-doc REQUIRES/UNLOCKS** from current model (not bak2 graft)  
- Chat answers as multi-book curriculum paths with in-bubble links  
- Correct soft anchors; gold F1; lower orphans  
- v4 **model train + deploy + re-ingest**  
- Auth, delete-doc, unified app, agentic/multimodal/enterprise  

### Single biggest truth
Plumbing is real. **Learning edges and consistent graph artifacts are not.** Inference loads root `okf_graph.json` (**135 concepts / 0 edges**); UI snapshot has **138 / 125**. Chat often cannot do multi-hop curriculum even when the graph UI looks connected.

---

## Progress by layer (done % / left %)

| # | Layer (from vision) | Done | Left | What “done” means today | What’s left |
|---|---------------------|-----:|-----:|-------------------------|-------------|
| 1 | Structure-aware chunking + provenance | **85%** | 15% | Pages, sections, kinds, offsets; tests | Printed-page maps everywhere; non-PDF upload in UI |
| 2 | OKF extract + cleanup + grounding | **70%** | 30% | Pipeline + filters + dedupe | Grammar-constrained JSON; live model quality |
| 3 | Relations from **current** SLM | **25%** | 75% | Code for 2nd pass exists | v4 train/deploy; stop relying on bak2 graft |
| 4 | Kùzu MERGE + schema + DAG gates | **75%** | 25% | Schema, inventory gate, cycle break | Concurrent safety, silent edge fail logging |
| 5 | **One graph truth** for UI + inference | **30%** | 70% | Rich snapshot exists under `graph_ui/` | Sync root JSON + DB + `_graph_*` + worker |
| 6 | Cross-document bridges | **30%** | 70% | ~7/125 true cross-doc edges | Dense paper↔paper REQUIRES/UNLOCKS |
| 7 | Embedder top-K anchors | **70%** | 30% | Snowflake rank + soft routing | Wrong soft anchors (LoRA family, etc.) |
| 8 | k-hop curriculum in answers | **35%** | 65% | Cypher path coded | Needs edges in the DB inference uses |
| 9 | Citations + PDF deep-links | **60%** | 40% | Rail links `#page=N` | In-bubble pedagogy + books list |
| 10 | Generator (Model #2) | **45%** | 55% | Ollama wording + template | Aura default; true curriculum voice |
| 11 | Fine-tune loop | **45%** | 55% | v4 **data** ready | Train, deploy, re-ingest |
| 12 | Pilot ops (tests, privacy, scripts) | **75%** | 25% | Session 8 package | Auth, delete-doc, full re-ingest proof |
| 13 | Chat + graph product UI | **60%** | 40% | 3-port pilot works | Single app; mode/temp honesty |
| 14 | Agentic / multimodal / enterprise / spatial | **10%** | 90% | Scratch plans only | All of it |

### Weighted roll-up (how we get ~52% overall)

| Bucket | Weight | Score | Contribution |
|--------|-------:|------:|-------------:|
| Ingest + cleanup + chunking | 20% | 75% | 15.0 |
| Topology (edges + export truth + bridges) | 25% | 35% | 8.8 |
| Inference (retrieve + answer + cites) | 25% | 50% | 12.5 |
| Pilot ops + tests + privacy | 15% | 75% | 11.3 |
| Future OS | 15% | 10% | 1.5 |
| **Total** | 100% | | **~49–52%** |

### If you only care about “can we demo a pilot tomorrow?”

| Item | Status |
|------|--------|
| Start graph + inference + chat + Ollama | Ready (when processes up) |
| Ask LoRA / RAG / fine-tuning style questions | Ready (summary + related + some cites) |
| See graph of books/concepts | Ready **if** UI JSON is the rich one |
| Multi-hop “learn X needs Y then Z across papers” | **Not reliable** (0 edges on inference export) |
| Production dept rollout without caveats | **Not ready** (auth, gold F1, model relations) |

### Live numbers (disk, 2026-07-16)

| Artifact | Concepts | Edges |
|----------|---------:|------:|
| `okf_graph.json` (inference + graph_server) | 135 | **0** |
| `graph_ui/okf_graph.json` | 138 | **125** (50 REQUIRES / 6 UNLOCKS / 69 RELATED) |
| `accuracy.json` (last rebuild) | 138 | 125 |
| Cross-doc edges (among 125) | — | **~7 (~6%)** |
| Orphans | — | **~54%** |
| Proxy accuracy overall | — | **64.8%** |
| Gold concept F1 (pilot sets) | — | **~4–11%** |

---

# PART A — What exists (full inventory)

## A1. Code modules

### Core package `okf/`
| File | Role |
|------|------|
| `config.py` | Model name, prompts, enums, paths |
| `extraction.py` | Local HF + Ollama extract, JSON normalize |
| `cleanup.py` | Grounding, dedupe, cycles, junk filter |
| `canonicalize.py` | Alias + fuzzy merge |
| `relations.py` | Second-pass relation extraction |
| `pipeline.py` | Full / `--add` / staged worker pipeline |
| `graph_db.py` | Kùzu schema, MERGE, export, evidence |
| `evaluate.py` | Proxy accuracy + gold P/R/F1 + structural audit |
| `exports.py` | Graph export / audit helpers |
| `util.py` | Shared helpers |
| `gold/gold_attention.json`, `gold_lora.json` | Gold checklists |

### Root services & tools
| File | Role |
|------|------|
| `pdf_ingestion.py` | Section-aware chunking, spans, chunk_kind |
| `okf_pipeline.py` | CLI shim |
| `rebuild_graph.py` | Rebuild from `okf_results.json` without re-extract |
| `import_bak2_relations.py` | Graft old-model relations |
| `ingestion_worker.py` | Background jobs + GraphLock + atomic swap |
| `ingestion_jobs.py` | Job store / stages |
| `inference_server.py` | Embed + rank + traverse + chat API :5051 |
| `graph_server.py` | Graph API + UI :5050 |
| `chat_server.py` | Chat static UI :5052 |
| `finetune_qwen35_okf.py`, `continue_finetune.py` | Fine-tune tooling |
| `bulk_ingest.sh`, `ingest_pilot_corpus.sh`, `pilot_readiness.sh` | Ops scripts |

### UIs
| Path | Port | Role |
|------|------|------|
| `chat_ui/index.html` | 5052 | Chat, upload, citations rail, modes, concepts |
| `graph_ui/index.html` | 5050 | Books / Concepts / Cross-Book D3 graph |
| `graph_ui/neon.html` | 5050 | Alternate skin |
| Diagnostic HTML in `inference_server.py` | 5051 | Model status + test form |

## A2. Corpus

### Main `pdfs/`
- **Papers:** Vaswani (Attention), Devlin (BERT), Hu (LoRA), Lewis (RAG), Edge (GraphRAG)
- **Textbook:** Deisenroth Math for ML (~17MB)
- **Syllabus:** `AI_ML_Archipelago_Corpus_Seed.md`
- **Extras:** `probable.pdf`, `synthetic_concept_x_lora.pdf` (test)

### `pilot_corpus/`
- README, `expected_stats.json`, gold (attention/lora/curriculum), `gold_eval_results.json`
- Symlinks to core 6 + optional Math-for-ML
- Scripts: `ingest_pilot_corpus.sh`

### Docs
- `OKF_SPEC.md`, `PRIVACY_POLICY.md`, `PILOT_READINESS_REPORT.md`, `NEXT_SESSION.md`, `DOMAIN_NOTE.md`, `FINE_TUNING_GUIDE.md`, plans under `scratch/`

## A3. Tests (`tests/` — complete list)

### Root
- `conftest.py`, `__init__.py`

### Unit (11 modules)
- `test_chunk_spans.py` — multipage spans, offsets, section titles  
- `test_citations.py` — Session 3 citation contract  
- `test_cleanup.py` — grounding/dedupe/orphan  
- `test_evaluate.py` — gold compare + structural  
- `test_fixtures.py` — fixture sanity  
- `test_ingestion_jobs.py` — job store  
- `test_pdf_chunking.py` — chunking/kinds  
- `test_pipeline_logic.py` — cycles/merge/canonicalize  
- `test_quality_and_eval.py` — DAG/orphan/provenance  
- `test_relations.py` — relation filters  

### Integration (7 modules)
- `test_citation_correctness.py` — Kùzu evidence  
- `test_graph_db.py` — graph ingest helpers  
- `test_graph_quality.py` — audit + gold helpers  
- `test_inference_contract.py` — readiness/stream shape  
- `test_ingestion_safety.py` — failed write isolation  
- `test_ingestion_worker.py` — worker/lock/swap  
- `test_query_pipeline.py` — chat path on tmp graph  

### E2E (3 modules + conftest)
- `conftest.py` — live URLs, waiters, synthetic PDF  
- `test_latency.py` — Session 8 p95 gates  
- `test_live_model.py` — readiness, RAG, citations, ingest  
- `test_upload_flow.py` — upload curl docs  

**~20 pytest modules** under `tests/` (plus scratch tests outside suite).

## A4. Training

| Item | Status |
|------|--------|
| v3 pairs | Large, bad stats (empty/mode-collapse) |
| v4 pairs | **95 quality pairs** built (72/23), report + V4_CHANGES |
| v4 fine-tune deployed as live extractor | **Not confirmed / not done per V4_CHANGES + NEXT_SESSION** |
| aura-qwen weights | Present (`aura-qwen/` / parent) |

---

# PART B — Agent reports (condensed)

## G1 — Extraction / OKF (~45% of vision)

### Works
- Full pipeline architecture: chunk → extract → cleanup → Kùzu  
- Chunking (pages, sections, `chunk_kind`, prose-only to SLM) well tested  
- Grounding filter, mode-collapse dedupe, inventory-gated edges, cycle break  
- Provenance on concepts (doc/chunk/page/passage)  
- v4 dataset rebuild quality leap  

### Broken / weak
- **No grammar-constrained decoding** (free-form JSON scrape)  
- **v3 model unfit for relations** (NEXT_SESSION: ~13/1024 relation records)  
- Live `okf_results.json` ~**0 prereqs**  
- Gold F1 ~**3–10%** name match  
- Orphans ~**54%**, relation_consistency **16.6%**  
- Pilot edges largely **bak2 graft**, not live SLM  

### Metrics (accuracy.json era)
| Metric | Value |
|--------|------:|
| overall_score | 64.8% |
| schema_completeness_core | 39.9% |
| relation_consistency | 16.6% |
| connectivity | 45.7% |
| concepts / edges (rebuild) | 138 / 125 |
| orphans | 75 (54%) |

---

## G2 — Graph topology / exports (~35% of vision)

### Artifact table (P0)

| Artifact | Concepts | Edges | Notes |
|----------|---------:|------:|-------|
| `okf_graph.json` (**inference DATA_FILE**) | **135** | **0** | Chat loads this |
| `graph_ui/okf_graph.json` | **138** | **125** | Graph UI |
| `accuracy.json` stats | 138 | 125 | Last rebuild metrics |
| `_graph_edges.json` | — | **0** | Empty |
| `_graph_nodes.json` | 135 | — | Nodes only |

### Edge mix (`graph_ui` snapshot)
- REQUIRES: **50**  
- UNLOCKS: **6** (very few “unlocks”)  
- RELATED: **69**  
- Provenance on edges: `source` like `papers/….pdf:chunk_012`  

### Cross-document bridges
- Heuristic: **only ~7 / 125 edges** truly cross distinct doc source sets  
- **~118 / 125** share at least one doc (mostly **within-doc** topology)  
- Vision claim “bridges across islands” is **mostly not met**  

### Schema (real)
- Nodes: Document, Chunk, Concept  
- Rels: HAS_CHUNK, MENTIONS, REQUIRES, UNLOCKS, RELATED  

### Works
- MERGE ingest, inventory gating, export_graph code path  
- Structural audit (0 self-loops / 0 cycles in pilot eval snapshot)  
- Graph UI can show books/concepts when fed rich JSON  

### Flaws
- **P0 dual source of truth** (root vs graph_ui JSON)  
- **P0 inference uses edge-empty export** → prereq/unlock traversal empty in chat  
- **P1** unlocks almost absent (6)  
- **P1** cross-doc bridges rare  
- **P1** Kùzu lock blocks concurrent tests/services  

---

## G3 — Corpus / pilot ops (~78% ops scaffolding)

### Works
- Curated pilot corpus + gold + expected stats  
- Worker + jobs + atomic swap  
- Privacy policy  
- Readiness script + latency E2E gates  
- Session 8 package dense and real  
- Session 3 citation tests explicit  

### Flaws
- READY claim is **structural/latency**, not gold quality  
- Graph for pilot from **bak5 rebuild**, not proven full `./ingest_pilot_corpus.sh`  
- `bulk_ingest.sh` ≠ pilot doc set (operator footgun)  
- No auth; no delete-document product path  
- Sessions 1–5,6,7 **not labeled** in tests  
- TODO.md / implementation_plan.md **stale** (HF cloud, inference “not started”)  
- Sample synthetic job: **0 edges**  

### Session matrix (inferred)

| Theme | Status in repo |
|-------|----------------|
| S1 Chunking | Implicit (code+tests) |
| S2 OKF/cleanup | Implicit |
| S3 Citations | **Explicit tests** |
| S4 Graph/gold eval | Implicit |
| S5 Inference RAG | Implicit |
| S6 Ingestion worker | Strong artifacts |
| S7 Live E2E | Strong artifacts |
| S8 Pilot readiness | **Explicit** |

---

## I1 — Inference retrieval / synthesis (~40–45% of two-pass vision)

### Pipeline (actual)
1. Load concepts from **`BASE_DIR/okf_graph.json`** (not graph_ui copy)  
2. Embed query (Snowflake Arctic Embed; CPU forced if CUDA dead)  
3. Rank concepts; soft/strong/chitchat/offtopic routing  
4. Cypher neighborhood REQUIRES/UNLOCKS k≈2  
5. Citations via MENTIONS/chunks  
6. Natural template and/or Ollama wording  

### Live behavior (from earlier smoke, while services were up)

| Query | Route | Anchor (example) | Prereq/Unlock | Style |
|-------|-------|------------------|---------------|-------|
| hi | general_chat | none | 0/0 | Free chat / fallback |
| weather | general_chat | none | 0/0 | Offtopic free chat |
| AI agents | graph_soft | nearby AIML node | often 0/0 | Summary + related |
| fine-tuning | graph_soft | Full Fine-Tuning etc. | often 0/0 | Summary + related + cite |
| What is LoRA? | soft/strong family | often *LoRA variant node* not core | often 0/0 | Summary + related + page cite |

### Works
- Embedder ranking + domain soft gate  
- Citation payload: `evidence_id`, `doc_id`, `page_number`, `url` (`#page=N`)  
- Natural fallback (no raw notes if path works)  
- Latency budgets previously measured  

### Cannot / weak
- **Reliable multi-hop curriculum** when export has 0 edges  
- Correct primary anchor (family confusion)  
- Aura as default generator (off)  
- Agentic tool-use Cypher loop  
- Stable GPU path  

---

## I2 — UI/UX vs vision (~40% productization; ~55–65% pilot UI)

### Can do in browser
- Chat stream, modes UI, upload stages, concept chips  
- Citation **rail** with clickable PDF page links  
- Graph Books / Concepts / Cross-Book  
- Diagnostic inference page  

### Cannot
- Pedagogical bubble: “learn X → need Y from Book p.N [link] → Z…” as default  
- Chat bookshelf / catalog / shelf location  
- Working temperature slider (not sent to API)  
- True conversational-agent split  
- Single-port unified app  
- Auth / multi-user / recommend / spatial / multimodal  

### Vision checklist

| Feature | Status |
|---------|--------|
| OKF + extract | Shipped (quality issues) |
| Ingest CLI + live PDF job | Shipped pilot |
| Two-pass query RAG | Shipped partial |
| Relation second pass | Code yes, model no |
| Learning DAG | Partial / sparse |
| Fine-tune tooling | Yes; live model debt |
| Agentic | No |
| Multimodal product | No |
| Enterprise library | No |
| Spatial | No |
| Graph explorer | Yes |
| Chat + upload + cites | Yes |
| Pedagogy hyperlinks in prose | Partial |

---

# PART C — Master flaw list (merged, de-duplicated)

### P0 — breaks the product thesis
1. **`okf_graph.json` (inference) has 0 edges; `graph_ui/okf_graph.json` has 125** — dual truth.  
2. **Chat often 0 prereqs/unlocks** — “topology RAG” collapses to embed neighbors.  
3. **Cross-doc bridges rare (~7/125)** — islands not bridged.  
4. **Extractor model unfit for relations** (v3); edges largely **grafted**.  
5. **v4 dataset not proven as live deployed model**.  
6. **Wrong soft anchors** (LoRA family, odd AIML neighbors for “agents”).  
7. **Unauthenticated services** on open ports (pilot risk).  

### P1 — serious quality / UX
8. Gold concept F1 ~3–10%; curriculum recall ~57%.  
9. Orphan rate ~54%; UNLOCKS only 6.  
10. Answer format not multi-book curriculum with in-bubble links.  
11. Temperature / mode toggles partly cosmetic.  
12. Chat↔graph handoff fragile (hardcoded ports / relative graph link).  
13. Pilot READY claim vs weak gold / rebuild-from-cache.  
14. `bulk_ingest.sh` ≠ pilot corpus set.  
15. Kùzu exclusive lock vs tests + multi-process.  
16. CUDA flakiness → CPU embeddings.  
17. No single-document delete product path.  

### P2 — debt / consistency
18. Schema completeness core 39.9% (empty prereq lists).  
19. Type enum drift (framework/principle/etc.).  
20. Stale TODO / implementation_plan (HF cloud).  
21. Sessions 1–7 poorly labeled.  
22. Dual gold locations.  
23. `_graph_edges.json` empty.  
24. PDF-only upload UI while MD in corpus.  
25. Diagnostic page overclaims Aura generator.  

### P3 — polish
26. Domain-hardcoded suggested prompts.  
27. Visual language split chat vs graph.  
28. Scratch plans (recommend, shelf, agentic) not productized.  
29. Root bak/thin artifact clutter.  
30. Mobile layout stacks panels.  

---

# PART D — What it CAN do now

1. Structure-aware PDF/MD chunking with pages/sections/kinds.  
2. Offline OKF extraction + cleanup + Kùzu MERGE (when run).  
3. Serve chat UI + graph UI + inference API (when processes up).  
4. Embed queries (Snowflake) and rank concepts.  
5. Soft-route AIML learning vs chitchat/offtopic.  
6. Return summaries + related concepts + **page citations with `#page=` URLs** (rail).  
7. Optional Ollama natural wording; template fallback.  
8. Live PDF upload jobs with staged progress + atomic graph swap (code).  
9. Substantial automated tests (unit/integration/e2e/latency).  
10. Pilot corpus layout, privacy doc, readiness script/report.  
11. Training data pipeline including **v4**.  
12. Graph explorer Books/Concepts/Cross-Book (when fed rich JSON).  

---

# PART E — What it CANNOT do now

1. Reliable **cross-document curriculum GPS** (“normalize before B-tree”).  
2. Consistent multi-hop prereq/unlock paths in chat (export/edge drift).  
3. High gold-fidelity concept/edge inventory.  
4. Grammar-constrained industrial extraction.  
5. Default pedagogical reply: books + topics + pages + hyperlinks in prose + “need more?”.  
6. Agentic LangGraph tool-use traversal.  
7. Multimodal diagram/video OKF.  
8. University catalog (shelf, call number, availability, recommend).  
9. Auth, multi-tenant, enterprise deploy.  
10. Spatial Graph RAG.  
11. Trust root `okf_graph.json` as edge source of truth.  
12. Live-proof full pilot re-ingest as the only path to READY graph.  
13. Aura as default production generator.  
14. Working temperature control in chat.  

---

# PART F — Completeness scores (consensus)

| Layer | % |
|-------|--:|
| Chunking + provenance fields | 85 |
| Extract + cleanup plumbing | 70 |
| Live model relation quality | 20–30 |
| Kùzu merge code | 75 |
| Consistent graph artifacts | 30 |
| Cross-doc bridges | 25–35 |
| Embed + rank | 70 |
| Graph traversal in answers | 30–40 |
| Citations API + rail links | 55–65 |
| Pedagogical answer product | 35–45 |
| Fine-tune data (v4) | 60 (data) / 25 (deployed model) |
| Pilot ops (scripts/tests/privacy) | 75–78 |
| Chat + graph UI pilot | 55–65 |
| Enterprise / agentic / multimodal / spatial | 5–15 |
| **Overall vs full architecture doc** | **50–55** |
| **Supervised departmental pilot** | **65–75** |

---

# PART G — Top 15 actions (priority, no implementation yet)

1. **Unify exports:** root `okf_graph.json` ≡ `graph_ui/okf_graph.json` ≡ Kùzu after every rebuild.  
2. **Prove chat prereq/unlock counts > 0** on LoRA/RAG after fix.  
3. **v4 fine-tune + re-extract relations** (retire bak2 as sole edge source).  
4. **Reply template:** multi-hop “learn X → need Y (book, p.N link) → Z… want more?”  
5. **Soft-anchor ranking** fix (prefer core concept over LoRA sub-nodes).  
6. **Auth / bind-to-localhost** for pilot.  
7. **Single document delete** path.  
8. Wire temperature + real conversational mode or remove UI lies.  
9. Align `bulk_ingest.sh` with pilot corpus.  
10. Raise gold F1 via aliases / evaluation normalization.  
11. Increase UNLOCKS / reduce orphans.  
12. Grammar-constrained or schema-validated extraction.  
13. Single-port or documented process manager for 5050–5052.  
14. Refresh stale TODO / implementation_plan.  
15. Full live matrix of 20 queries after services up + edges fixed.  

---

# PART H — Bottom line

**Archipelago is a real local pilot stack**, not vaporware: chunking, cleanup, Kùzu, chat, graph UI, citations, tests, and pilot ops exist.

**It is not yet the full Concept-Topology Graph RAG product** in the vision brief: bridges are thin, exports disagree, edges are partly historical grafts, and answers are not full multi-book roadmaps with in-message hyperlinks.

**Use this file as the single checklist.** Next implementation wave should start at **G2 P0 (export/edge consistency)** then **I1 pedagogy + anchors**, then **G1 v4 model**.

---

*End of full audit. No product code was modified for this report (report file only).*

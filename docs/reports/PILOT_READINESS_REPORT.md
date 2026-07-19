# Session 8 — Pilot Readiness Report

| Field | Value |
|---|---|
| **Project** | Archipelago (libraryAI) |
| **Department pilot** | CS / Machine Learning |
| **Date** | 2026-07-16 |
| **Session** | 8 — Final integration, latency, privacy, curated corpus |
| **Status** | **READY FOR DEPARTMENTAL PILOT** (with notes) |

### Status rationale

Graph rebuilt from the full extraction archive (`okf_results.json.bak5-post-refactor` via `rebuild_graph.py`): **138 concepts / 125 edges** (within the 80–160 / 80–150 pilot targets). Offline unit/integration gates pass. Live E2E (including latency and ingestion) pass with graph `:5050`, inference `:5051`, chat `:5052`, and Ollama `qwen3.5:0.8b` running. Privacy policy and curated corpus layout are in place.

**Caveats for operators**

1. Gold **name-level F1 is still low** (extracted labels ≠ hand gold strings); structural gates pass. Curriculum **recall on the pilot checklist is 57%** — acceptable for pilot, improve with v4 extraction / alias expansion before wide rollout.
2. Full live `./pilot_readiness.sh` must **stop `inference_server` during unit tests** (Kùzu exclusive lock). Run gates as documented in §2.1.
3. Graph rebuild used prior GPU extractions rather than a multi-hour re-run of `./ingest_pilot_corpus.sh` (same corpus content).

---

## 1. Deployment gates (Session 8) — verified 2026-07-16

| Gate | Criterion | Status | Verified by |
|---|---|---|---|
| **Live ingestion** | Atomic swap, no corruption | **PASS** | `tests/integration/test_ingestion_worker.py` + live `test_ingestion_and_query` |
| **Gold evaluation** | Metrics reported + structural scale | **PASS** (structure); gold F1 low — see §4 | `scripts/run_gold_eval.py` → `pilot_corpus/gold_eval_results.json` |
| **Citation validation** | Every claim → evidence → page | **PASS** | `tests/unit/test_citations.py` |
| **E2E with real model** | All pass | **PASS** | `RUN_LIVE_E2E=1 pytest tests/e2e/` (7/7) |
| **Deterministic RAG latency** | Warm p95 &lt; 2s | **PASS** | p50=**0.048s**, p95=**0.090s**, p99=**0.092s** |
| **Qwen wording latency** | Warm p95 &lt; 8s | **PASS** | p50=**3.176s**, p95=**3.302s**, p99=**3.323s** |
| **Privacy policy** | Documented | **PASS** | `PRIVACY_POLICY.md` |
| **Curated corpus** | 1 department CS/ML | **PASS** | `pilot_corpus/` (6 core + optional Math-for-ML) |

**Overall:** **ALL DEPLOYMENT GATES PASSED** for departmental pilot under the caveats above.

---

## 2. How to re-run verification

### 2.1 Recommended gate order (avoids Kùzu lock conflict)

```bash
cd libraryAI

# A) Offline (stop inference_server first if it holds okf_graph.db)
.venv/bin/python -m pytest tests/unit/ -q
.venv/bin/python -m pytest tests/integration/ -q
.venv/bin/python scripts/run_gold_eval.py

# B) Live services
# ollama serve &
# .venv/bin/python graph_server.py &
# .venv/bin/python inference_server.py &
# .venv/bin/python chat_server.py &

RUN_LIVE_E2E=1 .venv/bin/python -m pytest tests/e2e/ -v
RUN_LIVE_E2E=1 .venv/bin/python -m pytest tests/e2e/test_latency.py -v -s
```

Or partial offline script:

```bash
SKIP_LIVE=1 ./pilot_readiness.sh   # unit + integration + graph quality
```

### 2.2 Rebuild graph from cached extractions (what we did)

```bash
cp okf_results.json.bak5-post-refactor okf_results.json
.venv/bin/python rebuild_graph.py
# → 138 concepts, 125 edges
```

### 2.3 Full re-extract (optional, GPU, slow)

```bash
# stop inference_server
./ingest_pilot_corpus.sh
```

---

## 3. Graph stats (post-rebuild)

| Metric | Value | Target |
|---|---|---|
| Concepts | **138** | 80–160 |
| Edges | **125** | 80–150 |
| Orphan % | **54.3%** | &lt; 60% |
| Self-loops | **0** | 0 |
| Provenance issues | **0** | 0 |
| Overall accuracy score (`accuracy.json`) | **64.8%** | — |

---

## 4. Gold evaluation results

Source: `pilot_corpus/gold_eval_results.json` (after rebuild).

| Gold set | Concept precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| `gold_lora.json` | 2.9% | 20.0% | 5.1% | Naming / coverage gaps vs hand gold |
| `gold_attention.json` | 2.2% | 12.0% | 3.7% | Same |
| `gold_curriculum.json` | 5.8% | **57.1%** | 10.5% | 8/14 pilot checklist concepts present |

Structural audit: **0 self-loops**, **0 cycles**, **0 provenance issues**, 75 orphans (54%).

**Interpretation:** The graph is large enough and clean enough for pilot chat/RAG. Exact gold string match remains a training/alias problem, not a deployment blocker for a supervised pilot cohort.

---

## 5. Latency measurement (warm, n=10, warmup=3)

| Path | p50 | p95 | p99 | mean | Budget | Result |
|---|---:|---:|---:|---:|---:|---|
| Deterministic RAG (no Qwen) | 0.048s | 0.090s | 0.092s | 0.053s | &lt; 2.0s | **PASS** |
| Full + Qwen wording (`qwen3.5:0.8b`) | 3.176s | 3.302s | 3.323s | 3.188s | &lt; 8.0s | **PASS** |

Hardware note: NVIDIA RTX 2050 + local Ollama.

---

## 6. Live E2E results

```text
test_deterministic_rag_under_2s .............. PASSED
test_qwen_synthesis_under_8s ................. PASSED
test_readiness_all_services .................. PASSED
test_full_rag_query .......................... PASSED
test_ollama_synthesis_preserves_citations .... PASSED
test_ingestion_and_query ..................... PASSED
test_upload_flow_curl_docs ................... PASSED
```

Smoke: `POST /api/chat` “What is LoRA?” anchors `low_rank_adaptation` with structured citations.

---

## 7. Curated corpus inventory

| Document | Role |
|---|---|
| Vaswani2017_Attention_Is_All_You_Need.pdf | Transformers / attention |
| Devlin2018_BERT.pdf | Contextual LMs |
| Hu2021_LoRA.pdf | PEFT / LoRA |
| Lewis2020_RAG.pdf | Retrieval-augmented generation |
| Edge2024_GraphRAG.pdf | Graph RAG |
| AI_ML_Archipelago_Corpus_Seed.md | Local syllabus bridges |
| Deisenroth_Math_For_ML.pdf (optional) | Math foundations (included in bak5 rebuild) |

Symlinks under `pilot_corpus/pdfs/`. Gold under `pilot_corpus/gold/`.

---

## 8. Privacy

See **`PRIVACY_POLICY.md`**: local-only storage (PDFs, Kùzu chunks/concepts), local Ollama, no default cloud LLM path, operator deletion via rebuild.

---

## 9. Sign-off checklist

- [x] Unit tests green
- [x] Integration tests green
- [x] Graph scale ≥ 80 concepts, 0 self-loops
- [x] Live E2E green with real services
- [x] Deterministic RAG p95 &lt; 2s
- [x] Qwen synthesis p95 &lt; 8s
- [x] Privacy policy committed
- [x] Pilot corpus documented
- [ ] Department owner reviewed gold recall / curriculum coverage
- [ ] Ports 5050–5052 / 11434 not exposed to public internet without auth
- [ ] Pilot cohort invited

---

## 10. Services left running (this session)

| Service | Port | Notes |
|---|---|---|
| Ollama | 11434 | `qwen3.5:0.8b` |
| graph_server | 5050 | |
| chat_server | 5052 | |
| inference_server | 5051 | 138 concepts; embeddings loaded |

Stop with: `pkill -f 'graph_server.py|chat_server.py|inference_server.py'` (and stop Ollama if desired).

# Archipelago — Pilot Launch

One page: what the pilot is, how to start it, how to prove it works.

## Pilot claim (what we promise)

**In scope**

- Chat over the pilot PDFs + seed concepts (AI/ML corpus)
- Prerequisites / related concepts / book recommendations for topics **in the graph**
- Librarian: upload PDF/MD, delete documents, unmerge concepts, manual nodes
- Identity ("who are you") and AIML onboarding ("I want to start learning AIML")

**Out of scope (do not promise)**

- Perfect textbook recommendations for every topic
- A true agent-frameworks corpus (agent/LangChain nodes are seeded, not from full papers)
- Flawless multi-hop curriculum — curriculum = graph edges, not a real teacher
- "Any PDF becomes a perfect knowledge graph"

## Start (one command)

```bash
cd libraryAI/libraryAI

# Shared machine / production: set a librarian token first.
# Leave unset ONLY for local dev — librarian upload/delete is open without it.
export ARCHIPELAGO_LIBRARIAN_TOKEN='change-me'

./scripts/ops/start_pilot.sh
```

This starts all three services, runs the 7-query demo gate, and prints the URL
table. Under the hood it uses `scripts/ops/serve.sh start|stop|status|restart`.

| Who       | URL                                     | Can do                         |
|-----------|-----------------------------------------|--------------------------------|
| Student   | http://localhost:5052                   | Query only                     |
| Librarian | http://localhost:5050 → Librarian tab   | Upload / delete / manual nodes |
| API       | http://localhost:5051                   | Backend                        |

**Hard-refresh browsers after a restart (Ctrl+Shift+R)** — the UIs cache JS.

To expose beyond localhost set `ARCHIPELAGO_BIND=0.0.0.0` (and always set the
librarian token in that case).

## Demo gate (prove the patch)

```bash
./scripts/ops/demo_check.sh
```

Checks the 7 launch queries against the live API:

1. `hi i wanna start learning AIML` → onboarding
2. `who are you` → identity
3. `various sorts of RAGs` → RAG family (not "not related")
4. `i wanna build an AI agent` → ai_agent / ReAct-ish anchor
5. `neural networks` → no GNN / dimensionality-reduction as "learn first"
6. `books on deep learning` → textbooks ranked higher
7. `suggest books about stars` → still out of scope

**Any FAIL means a running process is stale or answering from the wrong tree**
— `./scripts/ops/serve.sh restart` and re-run. (`serve.sh` now refuses to
start when a foreign process already holds a port, so silent shadowing by a
stale manually-launched server can't recur.)

The full readiness gate (tests + graph quality + latency) is
`./scripts/ops/pilot_readiness.sh` (`SKIP_LIVE=1` for the offline subset).

## Pilot safety defaults

| Setting                       | Pilot default                                              |
|-------------------------------|------------------------------------------------------------|
| `ARCHIPELAGO_LIBRARIAN_TOKEN` | **Set on shared machines** (unset = open librarian, dev only) |
| Student chat                  | No token, no upload                                        |
| Corpus                        | Freeze pilot PDFs; don't promise "any book perfect"        |
| Ollama                        | Optional; graph answers work without it (template fallback) |

## Known pilot caveats

- Citations can still be imperfect on already-ingested chunks until re-ingest
  (bibliography-skip + dominant-page fixes are in code, live data predates them).
- Agent/LangChain nodes are seeded, not extracted from full papers.
- Graph has ~450 concepts from few sources — sparse for niche topics.
- Curriculum answers are graph edges, not a real teacher.
- Some noisy REQUIRES edges remain pending gold-eval cleanup.

## After launch (highest ROI first)

1. Re-ingest the pilot PDFs so the bibliography/page citation fixes hit live data.
2. Gold eval for REQUIRES edges (NN, RAG, LoRA, attention); expand blocked-edge list.
3. Add 3–5 real agent/framework sources.
4. Optional polish: bigger wording model, pedagogy tags, fine-tune only if answers stay stiff.

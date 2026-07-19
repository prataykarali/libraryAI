# Archipelago architecture (feature-based)

**Goal:** one feature per package, files **≤500 lines**, thin root shims for
backwards compatibility.

## Layout

```
libraryAI/
├── archipelago/                 # app features
│   ├── apps/                    # entrypoints (run servers)
│   │   └── inference_app.py
│   ├── inference/               # chat RAG (split modules)
│   │   ├── state.py             # globals, paths, Flask app
│   │   ├── aliases.py           # acronyms / aliases / PDF links
│   │   ├── embeddings.py
│   │   ├── ranking.py
│   │   ├── routing.py
│   │   ├── neighborhood.py
│   │   ├── curriculum.py        # multi-hop paths
│   │   ├── citations.py
│   │   ├── synthesis.py
│   │   ├── routes_misc.py       # readiness + ingest + PDFs
│   │   ├── routes_chat.py       # /api/chat
│   │   └── bootstrap.py
│   └── ingestion/               # PDF/MD chunking
│       ├── pdf_utils.py
│       ├── pdf_chunk.py
│       ├── pdf_formats.py
│       └── pdf_io.py
├── okf/                         # extraction → graph pipeline
│   ├── graph/                   # Kùzu (was graph_db.py monolith)
│   ├── cleanup_parts/           # grounding / dedupe / cycles
│   ├── eval/                    # gold + structural metrics
│   ├── config.py, extraction.py, relations.py, pipeline.py, ...
│   └── graph_db.py / cleanup.py / evaluate.py   # thin shims
├── ui/
│   ├── chat/                    # chat_ui → symlink
│   └── graph/                   # graph_ui → symlink
├── docs/{guides,reports}/
├── training/                    # finetune + dataset builders
├── scripts/ops/                 # shell + rebuild helpers
├── tests/{unit,integration,e2e}/
├── training_data/               # train/test jsonl only
├── pdfs/                        # corpus binaries
├── inference_server.py          # COMPAT shim → archipelago.inference
└── pdf_ingestion.py             # COMPAT shim → archipelago.ingestion
```

## Dependency rules

1. **Features import downward only:** `apps` → `inference` / `okf`; never the reverse.
2. **Shared state** for inference lives in `archipelago.inference.state` only.
3. **Root `*.py` shims** re-export public APIs so existing tests keep working:
   `import inference_server`, `from okf.graph_db import …`, `import pdf_ingestion`.
4. **New code** should import from feature packages, not expand shims.

## How to run

```bash
# Inference API :5051
python -m archipelago.apps.inference_app
# or (compat)
python inference_server.py

# Graph UI :5050
python graph_server.py

# Chat static :5052
python chat_server.py
```

## Line budget

| Package | Module target |
|---------|----------------|
| `archipelago/inference/*` | ≤500 LOC each |
| `okf/graph/*` | ≤500 |
| `okf/cleanup_parts/*` | ≤500 |
| `okf/eval/*` | ≤500 |
| `archipelago/ingestion/*` | ≤500 |
| Root shims | ≤150 |

## Notes

- Subagents were used for the restructure; free-tier rate limits often abort
  parallel agents — parent agent finishes verification.
- Graph data stays at repo root (`okf_graph.json`, `okf_graph.db`) so paths
  and workers do not break.

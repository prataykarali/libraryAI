# Archipelago Privacy & Document Retention Policy

**Version:** 1.0  
**Audience:** Departmental pilot (local deployment)  
**Scope:** Archipelago knowledge graph + chat + PDF ingestion stack

---

## 1. Summary

Archipelago is designed as a **local-only** library assistant. During the pilot:

- Uploaded teaching materials stay on the **host machine**.
- Concept graphs and chunk text live in a **local Kùzu database**.
- Generation and extraction use **local models** (Ollama / on-disk weights).
- There is **no cloud transmission** of document content for inference by default.

---

## 2. What data is stored

| Data class | Where it lives | Contents |
|---|---|---|
| Source PDFs / Markdown | `pdfs/` (and pilot uploads under job dirs) | Full original document bytes |
| Text chunks | Kùzu `Chunk` nodes | Passage text, page numbers, section titles, text offsets, optional block geometry |
| Documents | Kùzu `Document` nodes | Document IDs (typically filename-derived) |
| Concept graph | Kùzu `Concept` nodes + `REQUIRES` / `UNLOCKS` / `RELATED` edges | Canonical concept names, summaries, types, difficulty, relation provenance (`doc:chunk`) |
| Mentions / evidence | `HAS_CHUNK`, `MENTIONS` relationships | Links chunks → concepts for citations |
| Export artifacts | `okf_graph.json`, `okf_results.json`, `accuracy.json`, `graph_audit.json` | JSON snapshots for UI and evaluation |
| Ingestion jobs | `jobs/` | Job metadata, progress, errors; uploads may be staged as `upload.pdf` under each job |
| Chat queries | Not persisted by default | Chat requests are handled in-memory for the request lifecycle unless an operator enables logging |

**Not stored by default:** student account profiles, authentication tokens from external IdPs, or remote analytics.

---

## 3. Retention

| Artifact | Default retention | Notes |
|---|---|---|
| Curated corpus under `pdfs/` / `pilot_corpus/` | Indefinite for pilot | Operator-owned teaching materials |
| Live graph (`okf_graph.db`) | Indefinite until rebuild / delete | Replaced atomically on successful re-ingest |
| Staged job uploads (`jobs/<id>/`) | Until operator cleans `jobs/` | Safe to delete completed/failed job dirs after audit |
| JSON exports | Indefinite | May contain concept summaries derived from PDFs |
| Application logs (`ingest_logs/`, `*.log`) | Operator-managed | May include file paths and error text; avoid logging full passages in production |

There is **no automatic cloud backup**. Any backup is an operator choice (local disk only recommended for the pilot).

---

## 4. Access control

- Services bind to the local host network by default (`localhost` / LAN as configured).
- **No external API keys** are required for the default RAG path.
- Ollama and embedding models run **on the same machine** (or operator-controlled LAN).
- Do not expose ports `5050` / `5051` / `5052` / `11434` to the public internet without reverse-proxy auth and TLS.

### Default network posture

| Port | Service | Recommended exposure |
|---|---|---|
| 5050 | Graph UI / API | Local or authenticated LAN |
| 5051 | Inference + ingest API | Local or authenticated LAN |
| 5052 | Chat UI static server | Local or authenticated LAN |
| 11434 | Ollama | Localhost only |

---

## 5. Model & network transmission policy

| Component | Default behavior |
|---|---|
| Retrieval / deterministic learning path | Local Kùzu + in-process embeddings |
| Optional wording pass | Local Ollama (`ARCHIPELAGO_OLLAMA_MODEL`, default `qwen3.5:0.8b`) |
| Extraction during ingest | Local aura-qwen / configured local model |
| External LLM APIs (OpenAI, Anthropic, HF Inference, etc.) | **Not used** in the default pilot path |

**Operator obligation:** Do not point environment variables at third-party inference endpoints for departmental content unless a separate data-processing agreement is in place.

Hugging Face may be contacted only if an operator **explicitly downloads** models/weights; that is an admin action, not a runtime path for student documents.

---

## 6. Deletion — remove a document and its graph artifacts

There is no single “delete document” UI button in the pilot. Use the following operator procedure:

### 6.1 Remove the source file

```bash
# Example: remove a pilot PDF from the curated tree
rm -f pdfs/papers/SomePaper.pdf
# or from pilot_corpus copies/symlinks
rm -f pilot_corpus/pdfs/SomePaper.pdf
```

### 6.2 Remove staged upload jobs (if any)

```bash
# Inspect then remove job directories that reference the document
ls jobs/
rm -rf jobs/<job_id>
```

### 6.3 Rebuild the live graph without the document

Ingestion uses **MERGE into a quarantine DB + atomic swap**. The safe way to drop a document’s contribution is to **rebuild** from the remaining corpus:

```bash
# 1. Stop inference_server (releases Kùzu lock on okf_graph.db)
# 2. Backup current graph
cp -a okf_graph.db okf_graph.db.bak.$(date +%Y%m%d)
cp -a okf_graph.json okf_graph.json.bak.$(date +%Y%m%d)

# 3. Clear extraction cache if you need a clean re-extract
mv okf_results.json okf_results.json.bak.$(date +%Y%m%d) 2>/dev/null || true

# 4. Re-ingest remaining corpus only
./ingest_pilot_corpus.sh
# or: .venv/bin/python okf_pipeline.py --add <remaining.pdf> --local ...
```

After a successful rebuild, UI exports (`okf_graph.json`, visualization JSON) refresh with the new inventory. Concepts that only existed in the deleted document disappear; shared concepts that remain in other documents are retained with remaining provenance.

### 6.4 Verify deletion

```bash
.venv/bin/python - <<'PY'
import kuzu
conn = kuzu.Connection(kuzu.Database("okf_graph.db"))
doc_id = "SomePaper.pdf"  # adjust
res = conn.execute(f"MATCH (d:Document {{id: '{doc_id}'}}) RETURN d.id")
print("still present" if res.has_next() else "document gone")
PY
```

### 6.5 Optional secure erase

For sensitive materials on shared disks, after logical delete use OS secure-delete tools on freed blocks according to institutional policy.

---

## 7. Student / faculty chat data

- Chat messages are processed for the HTTP request only unless the operator enables additional logging.
- Responses may include **page citations** and short **text spans** drawn from ingested chunks.
- Students should not paste personal data (student IDs, grades, private email) into the chat; the system is not a student-record system of record.

---

## 8. Operator responsibilities (pilot)

1. Host Archipelago only on department-approved machines.
2. Restrict network exposure of service ports.
3. Keep teaching PDFs under license / fair-use rules applicable to the institution.
4. Run `./pilot_readiness.sh` before inviting a broader cohort.
5. Document any deviation that introduces external API calls.

---

## 9. Contact

For pilot questions, contact the deploying instructor or library IT owner for this instance. This document is operational guidance for the Archipelago pilot stack, not a substitute for institutional legal counsel.

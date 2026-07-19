"""Flask routes: CORS, PDFs, readiness, ingestion API, diagnostics."""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

from flask import jsonify, send_from_directory, request, redirect

import kuzu

from ingestion_jobs import JobStatus
from ingestion_worker import job_store, get_worker, graph_lock
from archipelago.auth import require_librarian, librarian_token_expected
from archipelago.inference import state as st
import torch

@st.app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


@st.app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    """Serve local papers/textbooks from the pdfs folder.

    When the file is not on disk (slim deployments that don't ship the corpus
    PDFs), redirect to the canonical public source (arXiv / mml-book) so the
    reading pane can still stream the full document.
    """
    local = Path(st.PDF_DIR) / filename
    if local.is_file():
        return send_from_directory(str(st.PDF_DIR), filename)
    remote = REMOTE_PDF_SOURCES.get(Path(filename).name)
    if remote:
        return redirect(remote, code=302)
    return jsonify({"error": f"PDF not found: {filename}"}), 404


# Canonical public URLs for the corpus documents (all openly licensed/hosted:
# arXiv preprints and the officially-free MML book). Used as a streaming
# fallback so full texts never need to live on this machine.
REMOTE_PDF_SOURCES = {
    "Vaswani2017_Attention_Is_All_You_Need.pdf": "https://arxiv.org/pdf/1706.03762",
    "Hu2021_LoRA.pdf": "https://arxiv.org/pdf/2106.09685",
    "Lewis2020_RAG.pdf": "https://arxiv.org/pdf/2005.11401",
    "Devlin2018_BERT.pdf": "https://arxiv.org/pdf/1810.04805",
    "Edge2024_GraphRAG.pdf": "https://arxiv.org/pdf/2404.16130",
    "Deisenroth_Math_For_ML.pdf": "https://mml-book.github.io/book/mml-book.pdf",
}


# ── Shared chats: tiny file-backed store (no DB, no auth beyond size caps) ──
SHARES_DIR = Path(st.BASE_DIR) / "shares"
_SHARE_MAX_BYTES = 512 * 1024
_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")


@st.app.route("/api/share", methods=["POST"])
def create_share():
    """Persist a conversation snapshot and return a share id."""
    data = request.get_json(silent=True) or {}
    history = data.get("history")
    if not isinstance(history, list) or not history:
        return jsonify({"error": "history must be a non-empty list"}), 400
    clean = []
    for m in history[:200]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content[:20000]})
    if not clean:
        return jsonify({"error": "no valid messages in history"}), 400
    payload = {
        "title": str(data.get("title") or "")[:120],
        "history": clean,
        "created_at": time.time(),
    }
    raw = json.dumps(payload, ensure_ascii=False)
    if len(raw.encode("utf-8")) > _SHARE_MAX_BYTES:
        return jsonify({"error": "conversation too large to share"}), 413
    share_id = secrets.token_urlsafe(9)
    SHARES_DIR.mkdir(exist_ok=True)
    (SHARES_DIR / f"{share_id}.json").write_text(raw, encoding="utf-8")
    return jsonify({"share_id": share_id, "url": f"/?share={share_id}"})


@st.app.route("/api/share/<share_id>", methods=["GET"])
def get_share(share_id):
    """Fetch a shared conversation snapshot by id."""
    if not _SHARE_ID_RE.match(share_id or ""):
        return jsonify({"error": "invalid share id"}), 400
    path = SHARES_DIR / f"{share_id}.json"
    if not path.is_file():
        return jsonify({"error": "share not found"}), 404
    try:
        return jsonify(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return jsonify({"error": "share unreadable"}), 500



@st.app.route("/api/readiness", methods=["GET"])
def readiness():
    """Lightweight health contract for the UI and test harness.

    It reports what is available instead of claiming that a model is ready just
    because a background loader was started.  Ollama is intentionally reported
    as configured (not synchronously probed) so this endpoint remains fast.
    """
    graph_ok = False
    graph_error = None
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            res = conn.execute("MATCH (c:Concept) RETURN count(c)")
            graph_ok = res.has_next()
    except Exception as e:
        graph_error = str(e)

    payload = {
        "ready": bool(graph_ok and st.CONCEPTS_DATA),
        "graph": {"ready": graph_ok, "concept_count": len(st.CONCEPTS_DATA), "error": graph_error},
        "retrieval": {
            "embedding_ready": st.use_embeddings,
            "lexical_fallback": True,
            "semantic_threshold": st.SEMANTIC_ANCHOR_THRESHOLD,
            "lexical_threshold": st.LEXICAL_ANCHOR_THRESHOLD,
        },
        "synthesis": {
            "default_model": st.DEFAULT_OLLAMA_MODEL,
            "mode": "optional_after_retrieval",
            "aura_compatibility_loaded": st.aura_loaded,
        },
        "ingestion": {
            "upload_api_enabled": True,
            "delete_api_enabled": True,
            "librarian_only": True,
            "student_upload_enabled": False,
            "auth_required": bool(librarian_token_expected()),
            "worker_thread_alive": False,  # will be updated dynamically below
        },
        "roles": {
            "chat": "student",
            "graph_browse": "student",
            "graph_librarian": "librarian",
        },
    }
    try:
        from ingestion_worker import get_worker
        payload["ingestion"]["worker_thread_alive"] = get_worker().is_alive()
    except Exception:
        pass
    return jsonify(payload), (200 if payload["ready"] else 503)


@st.app.route("/api/ingest/capabilities", methods=["GET"])
def ingestion_capabilities():
    """Advertise upload API capabilities (librarian-only mutations)."""
    return jsonify({
        "upload_api_enabled": True,
        "delete_api_enabled": True,
        "librarian_only": True,
        "student_upload_enabled": False,
        "auth_required": bool(librarian_token_expected()),
        "list_documents_api": "/api/documents",
        "delete_document_api": "DELETE /api/documents/<doc_id>",
        "max_file_size_mb": 50,
        "supported_formats": ["pdf", "md", "markdown", "txt"],
        "note": (
            "Upload/delete is only available from the Graph UI Librarian "
            "console with a librarian token. Student chat cannot mutate the corpus."
        ),
    }), 200


@st.app.route("/api/ingest", methods=["POST"])
@require_librarian
def ingest_upload():
    """Librarian-only: accept a document upload (PDF/MD/TXT), create a job and enqueue it."""
    import time
    from werkzeug.utils import secure_filename

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(f.filename or "upload.pdf")
    lower = filename.lower()
    allowed = (".pdf", ".md", ".markdown", ".txt")
    if not any(lower.endswith(ext) for ext in allowed):
        return jsonify({
            "error": "Supported formats: PDF, Markdown (.md), plain text (.txt)",
        }), 400

    # Check file size (<= 50 MB)
    f.seek(0, 2)
    size_bytes = f.tell()
    f.seek(0)
    if size_bytes > 50 * 1024 * 1024:
        return jsonify({"error": "File exceeds 50 MB limit"}), 413

    # Create job and quarantine the upload (preserve extension for pipeline)
    job = job_store.create_job(filename)
    job_dir = job_store.job_dir(job.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower().replace(".markdown", ".md") or ".pdf"
    if ext not in (".pdf", ".md", ".txt"):
        ext = ".pdf"
    upload_path = job_dir / f"upload{ext}"
    f.save(str(upload_path))
    # Marker so the worker always finds the quarantined source
    (job_dir / "upload_source_name.txt").write_text(filename, encoding="utf-8")

    # Enqueue and start worker
    worker = get_worker()
    worker.enqueue(job.job_id)

    return jsonify({
        "job_id": job.job_id,
        "status": "queued",
        "filename": filename,
        "format": ext.lstrip("."),
    }), 202


@st.app.route("/api/ingest/<job_id>", methods=["GET"])
def ingest_status(job_id):
    """Return job status, progress per stage, elapsed time, and result."""
    import time
    from datetime import datetime, timezone

    record = job_store.get_job(job_id)
    if record is None:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    # Compute elapsed seconds
    try:
        created = datetime.fromisoformat(record.created_at)
        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    except Exception:
        elapsed = None

    return jsonify({
        "job_id":          record.job_id,
        "status":          record.status.value,
        "source_filename": record.source_filename,
        "created_at":      record.created_at,
        "updated_at":      record.updated_at,
        "elapsed_seconds": elapsed,
        "progress":        record.progress,
        "error":           record.error,
        "result":          record.result,
        "graph_version":   record.graph_version,
    }), 200


@st.app.route("/api/ingest/<job_id>/cancel", methods=["GET", "POST"])
@require_librarian
def ingest_cancel(job_id):
    """Librarian-only: request cancellation of a running job."""
    record = job_store.cancel_job(job_id)
    if record is None:
        return jsonify({"error": f"Job {job_id} not found"}), 404
    return jsonify({
        "job_id": record.job_id,
        "status": record.status.value,
        "cancelled": record.cancelled,
    }), 200


@st.app.route("/api/ingest", methods=["GET"])
def ingest_list():
    """List all ingestion jobs."""
    return jsonify({"jobs": job_store.list_jobs()}), 200


@st.app.route("/api/documents", methods=["GET"])
def list_graph_documents():
    """List documents currently in the live knowledge graph."""
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            from okf.graph.delete_document import list_documents
            docs = list_documents(conn)
        return jsonify({"documents": docs, "count": len(docs)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@st.app.route("/api/documents/<path:doc_id>", methods=["DELETE"])
@require_librarian
def delete_graph_document(doc_id):
    """Librarian-only: delete a document and unmerge its contribution from the live graph.

    Removes: Document, Chunks, MENTIONS, edges whose provenance is this doc,
    and orphan concepts that no longer have mentions or structural edges.

    Shared concepts used by other documents are retained.

    Query params:
      remove_pdf=1  — also delete the file under pdfs/ when present
    """
    remove_pdf = str(request.args.get("remove_pdf") or "").lower() in (
        "1", "true", "yes", "on",
    )
    try:
        from okf.graph.delete_document import delete_document_end_to_end

        with graph_lock.write_lock():
            stats = delete_document_end_to_end(
                st.db,
                doc_id,
                base_dir=Path(st.BASE_DIR),
                okf_results_path=Path(st.BASE_DIR) / "okf_results.json",
                remove_pdf=remove_pdf,
                pdf_dir=Path(st.PDF_DIR),
            )
            # Reload inference handles + concept cache
            try:
                st.reload_db()
            except Exception as e:
                stats["reload_warning"] = str(e)
            try:
                from archipelago.inference.embeddings import build_concept_embeddings
                build_concept_embeddings()
                stats["embeddings_rebuilt"] = True
            except Exception as e:
                stats["embeddings_warning"] = str(e)

        return jsonify({"status": "deleted", **stats}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Manual Librarian Endpoints ───────────────────────────────────────────────

@st.app.route("/api/manual/concept", methods=["POST", "OPTIONS"])
@require_librarian
def manual_add_concept():
    """Librarian-only: insert a new concept node into KuzuDB and regenerate artifacts.

    Expected JSON body::

        {
          "name":         "Agentic System",      # required
          "concept_type": "architecture",        # required
          "difficulty":   "advanced",            # required
          "summary":      "An AI system that…", # optional
          "tags":         ["agents", "llm"]      # optional list[str]
        }
    """
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True, silent=True) or {}
    name         = (data.get("name") or "").strip()
    concept_type = (data.get("concept_type") or "").strip()
    difficulty   = (data.get("difficulty") or "").strip()
    summary      = (data.get("summary") or "").strip()
    tags         = [t.strip() for t in (data.get("tags") or []) if t.strip()]

    if not name:
        return jsonify({"error": "Field 'name' is required."}), 400
    if not concept_type:
        return jsonify({"error": "Field 'concept_type' is required."}), 400
    if not difficulty:
        return jsonify({"error": "Field 'difficulty' is required."}), 400

    # Derive a slug-style ID
    concept_id = name.lower().replace(" ", "_").replace("-", "_")

    try:
        from okf.graph.ingest import ensure_concept
        from okf.graph.export import export_graph
        from okf.exports import write_all_artifacts as _waa

        with graph_lock.write_lock():
            conn = kuzu.Connection(st.db)
            ensure_concept(conn, concept_id, name, concept_type, difficulty, summary, tags)
            try:
                from okf.exports import build_visual_graph, build_graph_rag_index
                base_dir = Path(st.BASE_DIR)
                graph_export = export_graph(conn)
                graph_export["visualization"] = build_visual_graph([], graph_export)
                graph_export["graph_rag_index"] = build_graph_rag_index([], graph_export)
                _waa(graph_export, [], st.db, base_dir=base_dir)
                try:
                    st.reload_db()
                except Exception:
                    pass
            except Exception as export_err:
                return jsonify({
                    "concept_id": concept_id,
                    "name": name,
                    "warning": f"Concept stored but artifact export failed: {export_err}",
                }), 201

        return jsonify({"concept_id": concept_id, "name": name, "status": "created"}), 201

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@st.app.route("/api/manual/edge", methods=["POST", "OPTIONS"])
@require_librarian
def manual_add_edge():
    """Librarian-only: add a directed relationship between two existing concept nodes.

    Expected JSON body::

        {
          "from_concept": "Retrieval-Augmented Generation",  # required
          "to_concept":   "Agentic System",                  # required
          "relation":     "enables"                          # required
        }

    ``relation`` must be one of: requires | enables | uses | extends |
    part_of | contrasts_with | evaluated_by
    """
    if request.method == "OPTIONS":
        return "", 204

    VALID_RELATIONS = {
        "requires", "enables", "uses", "extends",
        "part_of", "contrasts_with", "evaluated_by",
    }

    data         = request.get_json(force=True, silent=True) or {}
    from_concept = (data.get("from_concept") or "").strip()
    to_concept   = (data.get("to_concept") or "").strip()
    relation     = (data.get("relation") or "").strip().lower()

    if not from_concept:
        return jsonify({"error": "Field 'from_concept' is required."}), 400
    if not to_concept:
        return jsonify({"error": "Field 'to_concept' is required."}), 400
    if relation not in VALID_RELATIONS:
        return jsonify({"error": f"'relation' must be one of: {sorted(VALID_RELATIONS)}"}), 400

    def _id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    from_id = _id(from_concept)
    to_id   = _id(to_concept)

    try:
        from okf.graph.ingest import create_edge
        from okf.util import create_concept_id
        from okf.graph.export import export_graph
        from okf.exports import write_all_artifacts as _waa

        # Prefer canonical IDs when names differ from slug form
        from_id = create_concept_id(from_concept) or from_id
        to_id = create_concept_id(to_concept) or to_id

        with graph_lock.write_lock():
            conn = kuzu.Connection(st.db)

            # Verify both concepts exist (by id, then fuzzy name)
            for cid, label in [(from_id, from_concept), (to_id, to_concept)]:
                safe = cid.replace("'", "\\'")
                res = conn.execute(
                    f"MATCH (c:Concept {{id: '{safe}'}}) RETURN c.id LIMIT 1"
                )
                if not res.has_next():
                    # try match by name
                    safe_name = label.replace("'", "\\'")
                    res2 = conn.execute(
                        f"MATCH (c:Concept) WHERE c.name = '{safe_name}' RETURN c.id LIMIT 1"
                    )
                    if not res2.has_next():
                        return jsonify({
                            "error": f"Concept not found: '{label}' (id: '{cid}'). Add it first."
                        }), 404
                    resolved = res2.get_next()[0]
                    if cid == from_id:
                        from_id = resolved
                    else:
                        to_id = resolved

            try:
                create_edge(conn, from_id, to_id, relation, source="manual:librarian")
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 409

            try:
                from okf.exports import build_visual_graph, build_graph_rag_index
                base_dir = Path(st.BASE_DIR)
                graph_export = export_graph(conn)
                graph_export["visualization"] = build_visual_graph([], graph_export)
                graph_export["graph_rag_index"] = build_graph_rag_index([], graph_export)
                _waa(graph_export, [], st.db, base_dir=base_dir)
                try:
                    st.reload_db()
                except Exception:
                    pass
            except Exception as export_err:
                return jsonify({
                    "from": from_concept, "to": to_concept, "relation": relation,
                    "warning": f"Edge stored but artifact export failed: {export_err}",
                }), 201

        return jsonify({
            "from": from_concept, "to": to_concept,
            "relation": relation, "status": "created",
        }), 201

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



@st.app.route("/")
def server_root():
    
    gpu_info = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_info = f"Active ({torch.cuda.get_device_name(0)}, Memory: {torch.cuda.memory_allocated(0)/(1024**2):.1f}MB allocated)"
        except Exception as e:
            gpu_info = f"Available but inactive: {e}"
            
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Archipelago Inference Server Diagnostic</title>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0f0a15;
                --card-bg: rgba(255, 255, 255, 0.03);
                --bd: rgba(255, 255, 255, 0.08);
                --tx: #f3f0f7;
                --tx-mu: #a59fb1;
                --accent: #8b5cf6;
                --success: #10b981;
                --warning: #f59e0b;
            }}
            body {{
                background: var(--bg);
                color: var(--tx);
                font-family: 'Outfit', sans-serif;
                margin: 0;
                padding: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                box-sizing: border-box;
            }}
            .container {{
                max-width: 800px;
                width: 100%;
            }}
            h1 {{
                font-size: 32px;
                font-weight: 800;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }}
            .subtitle {{
                color: var(--tx-mu);
                margin-bottom: 40px;
                font-size: 16px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .card {{
                background: var(--card-bg);
                border: 1px solid var(--bd);
                border-radius: 16px;
                padding: 24px;
                backdrop-filter: blur(20px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .card-title {{
                font-size: 11px;
                font-weight: 600;
                color: var(--tx-mu);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
            }}
            .card-value {{
                font-size: 18px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .status-dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }}
            .status-dot.active {{
                background: var(--success);
                box-shadow: 0 0 10px var(--success);
            }}
            .status-dot.inactive {{
                background: var(--warning);
                box-shadow: 0 0 10px var(--warning);
            }}
            .explain-section {{
                background: rgba(139, 92, 246, 0.05);
                border: 1px solid rgba(139, 92, 246, 0.2);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 40px;
                line-height: 1.6;
            }}
            .explain-title {{
                font-weight: 600;
                margin-bottom: 8px;
                color: #c084fc;
            }}
            .test-form {{
                background: var(--card-bg);
                border: 1px solid var(--bd);
                border-radius: 16px;
                padding: 28px;
            }}
            input[type="text"] {{
                width: 100%;
                padding: 12px 16px;
                border-radius: 10px;
                border: 1px solid var(--bd);
                background: rgba(0,0,0,0.2);
                color: #fff;
                font-family: inherit;
                outline: none;
                margin-bottom: 16px;
                box-sizing: border-box;
            }}
            input[type="text"]:focus {{
                border-color: var(--accent);
            }}
            button {{
                background: var(--accent);
                color: #fff;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: 600;
                cursor: pointer;
                font-family: inherit;
                transition: opacity 0.15s;
            }}
            button:hover {{
                opacity: 0.9;
            }}
            pre {{
                background: rgba(0,0,0,0.4);
                padding: 16px;
                border-radius: 10px;
                overflow-x: auto;
                font-size: 12.5px;
                margin-top: 16px;
                border: 1px solid var(--bd);
                color: #86efac;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Archipelago Inference Server</h1>
            <div class="subtitle">Diagnostic & Local RAG Control Panel (Port 5051)</div>
            
            <div class="grid">
                <div class="card">
                    <div class="card-title">Embedding Model</div>
                    <div class="card-value">
                        <span class="status-dot {'active' if st.use_embeddings else 'inactive'}"></span>
                        <span>{'Active (Snowflake)' if st.use_embeddings else 'Loading / Standby'}</span>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Generator Model</div>
                    <div class="card-value">
                        <span class="status-dot active"></span>
                        <span>Ollama ({st.DEFAULT_OLLAMA_MODEL})</span>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">KuzuDB Status</div>
                    <div class="card-value">
                        <span class="status-dot active"></span>
                        <span>{len(st.CONCEPTS_DATA)} Concepts</span>
                    </div>
                </div>
            </div>
            
            <div class="explain-section">
                <div class="explain-title">💨 Why is my computer's fan spinning?</div>
                <div>
                    The local inference server runs two models directly on your hardware (GPU: {gpu_info}):
                    <ul>
                        <li><strong>Snowflake Arctic Embed (M)</strong>: Converts query text into a 768-dimensional dense vector to find concept anchors in KuzuDB.</li>
                        <li><strong>Ollama ({st.DEFAULT_OLLAMA_MODEL})</strong>: Natural language synthesis over retrieved graph notes — runs via local Ollama server.</li>
                        <li><strong>lib-qwen (1.5B SLM, extraction-only)</strong>: Used exclusively during ingestion to extract concepts from PDFs. Not loaded at inference time.</li>
                    </ul>
                    Because these models run locally, loading model weights into memory and compiling tensors creates a temporary CPU/GPU load, which spins the system fan to cool down the processor.
                </div>
            </div>
            
            <div class="test-form">
                <div class="card-title" style="margin-bottom:16px;">Test Inference Directly</div>
                <input type="text" id="query" placeholder="Enter a concept (e.g. What is LoRA?)..." value="What is LoRA?">
                <button onclick="runTest()">Run Inference</button>
                <div id="output-section" style="display:none;">
                    <div class="card-title" style="margin-top:20px; margin-bottom:8px;">Response Payload</div>
                    <pre id="output"></pre>
                </div>
            </div>
        </div>
        
        <script>
            function runTest() {{
                const query = document.getElementById('query').value;
                const output = document.getElementById('output');
                const outSec = document.getElementById('output-section');
                outSec.style.display = 'block';
                output.textContent = 'Initializing stream...';
                
                fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query: query, mode: 'rag_synthesis', history: [] }})
                }})
                .then(res => {{
                    if (!res.ok) throw new Error("HTTP error " + res.status);
                    const reader = res.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    let metadataParsed = false;
                    output.textContent = '';
                    
                    function read() {{
                        return reader.read().then(({{ done, value }}) => {{
                            if (done) {{
                                if (buffer) {{
                                    output.textContent += buffer;
                                }}
                                return;
                            }}
                            buffer += decoder.decode(value, {{ stream: true }});
                            
                            if (!metadataParsed) {{
                                const index = buffer.indexOf('\n[STREAM_START]\n');
                                if (index !== -1) {{
                                    const metaStr = buffer.substring(0, index);
                                    buffer = buffer.substring(index + 16);
                                    metadataParsed = true;
                                    output.textContent += "--- RETRIEVAL METADATA ---\n" + 
                                        JSON.stringify(JSON.parse(metaStr), null, 2) + 
                                        "\n\n--- ARCHIPELAGO GENERATION ---\n";
                                }}
                            }}
                            
                            if (metadataParsed) {{
                                output.textContent += buffer;
                                buffer = '';
                            }}
                            
                            return read();
                        }});
                    }}
                    return read();
                }})
                .catch(err => {{
                    output.textContent = 'Error: ' + err;
                }});
            }}
        </script>
    </body>
    </html>
    """
    return html



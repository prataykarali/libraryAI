"""pipeline_staged.py — Staged pipeline execution for the background live worker."""
import json
from okf import config, extraction
from okf.config import _local_path
from okf.extraction import load_local_model
from okf.graph_db import ingest_to_kuzu
from okf.pipeline import compute_doc_id, filter_extracted_relations, cleanup_and_canonicalize

class PipelineAborted(Exception):
    """Raised when a job is cancelled mid-flight."""

def run_pipeline_staged(
    source_path: str,
    temp_db_path: str,
    on_progress,
    check_cancelled=None,
):
    """
    Staged pipeline execution for the safe live ingestion worker.
    """
    from pdf_ingestion import ingest_document
    from okf.cleanup import clean_pipeline, cleanup_and_canonicalize
    from okf.evaluate import structural_audit

    def _check():
        if check_cancelled and check_cancelled():
            raise PipelineAborted("Job cancelled")

    # ── Stage 1: PARSING ────────────────────────────────────────────────
    on_progress("PARSING", 0, {"message": "Chunking document"})
    _check()

    raw_chunks = ingest_document(source_path, max_pages=config.MAX_PAGES_PER_DOC)
    doc_id = compute_doc_id(source_path)
    for chunk in raw_chunks:
        chunk.setdefault("doc_id", doc_id)

    prose_chunks = [c for c in raw_chunks if c.get("chunk_kind", "prose") == "prose"]
    if not prose_chunks:
        raise ValueError("No prose chunks to extract from!")

    on_progress("PARSING", 100, {
        "message": f"Chunked into {len(prose_chunks)} prose chunks",
        "total_chunks": len(raw_chunks),
        "prose_chunks": len(prose_chunks),
    })
    _check()

    # ── Stage 2: EXTRACTION ─────────────────────────────────────────────
    on_progress("EXTRACTION", 0, {"message": "Starting OKF extraction"})

    if extraction.LOCAL_MODEL is None and extraction.LOCAL_MODE:
        load_local_model()

    new_results = []
    new_successful = 0
    for i, chunk in enumerate(prose_chunks):
        _check()
        from okf.extraction import extract_okf_v15
        results = extract_okf_v15(
            text=chunk["text"],
            doc_id=chunk.get("doc_id", doc_id),
            chunk_id=chunk["chunk_id"],
            page_number=chunk.get("page_number", 0),
            section_title=chunk.get("section_title", ""),
            doc_hash=chunk.get("doc_hash"),
            page_count=chunk.get("page_count"),
            doc_title=chunk.get("doc_title"),
            edition=chunk.get("edition"),
            page_label_map=chunk.get("page_label_map"),
        )
        if results:
            passage = chunk["text"][:1600]
            for r in results:
                r["source_passage"] = passage
                r.setdefault("section_title", chunk.get("section_title", ""))
            new_results.extend(results)
            new_successful += 1
        pct = int((i + 1) / len(prose_chunks) * 100)
        on_progress("EXTRACTION", pct, {
            "message": f"Extracted {i+1}/{len(prose_chunks)} chunks",
            "concepts_so_far": len(new_results),
        })

    if not new_results:
        raise ValueError("Extraction produced 0 concepts — aborting.")
    _check()

    # ── Stage 3: CANONICALIZATION ────────────────────────────────────────
    on_progress("CANONICALIZATION", 0, {"message": "Merging with existing knowledge graph"})

    from okf.pipeline import BASE_DIR
    saved_file = BASE_DIR / "okf_results.json"
    existing = []
    if saved_file.exists():
        try:
            existing = json.loads(saved_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing = [r for r in existing if r.get("doc_id") != doc_id]

    new_results = clean_pipeline(new_results, raw_chunks)
    new_results = filter_extracted_relations(new_results, raw_chunks,
                                             inventory=existing + new_results)

    okf_results = existing + new_results
    okf_results = cleanup_and_canonicalize(okf_results)
    
    # Apply co-mention RELATED builder and UNLOCKS heuristics in staged pipeline
    from okf.co_mention import build_co_mention_edges
    from okf.unlocks_heuristics import add_heuristic_unlocks
    okf_results = build_co_mention_edges(okf_results, raw_chunks)
    okf_results = add_heuristic_unlocks(okf_results, raw_chunks)
    
    _check()

    on_progress("CANONICALIZATION", 100, {
        "message": "Canonicalization complete",
        "total_concepts": len(okf_results),
    })

    # ── Stage 4: GRAPH_BUILD ─────────────────────────────────────────────
    on_progress("GRAPH_BUILD", 0, {"message": "Building graph in temp DB"})
    _check()

    conn, db, graph_export = ingest_to_kuzu(okf_results, db_path=temp_db_path)

    on_progress("GRAPH_BUILD", 100, {
        "message": "Graph build complete",
        "nodes": graph_export.get("stats", {}).get("total_concepts", 0),
        "edges": graph_export.get("stats", {}).get("total_edges", 0),
    })
    _check()

    # ── Stage 5: GRAPH_VALIDATION ────────────────────────────────────────
    on_progress("GRAPH_VALIDATION", 0, {"message": "Validating structural integrity"})

    audit_res = structural_audit(db)
    if audit_res.get("self_edges") or audit_res.get("cycles"):
        raise ValueError(
            f"Structural audit failed — self-edges: {audit_res.get('self_edges')}, "
            f"cycles: {audit_res.get('cycles')}"
        )

    on_progress("GRAPH_VALIDATION", 100, {"message": "Validation passed", "audit": audit_res})

    return okf_results, db, graph_export

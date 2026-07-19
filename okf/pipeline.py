"""Pipeline orchestration: full runs, incremental --add, and finalization."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from okf import config, extraction
from okf.cleanup import cleanup_and_canonicalize
from okf.config import BASE_DIR, MODEL_NAME, _local_path, infer_source_category
from okf.extraction import extract_chunks_with_model, load_local_model
from okf.graph_db import ingest_to_kuzu


def filter_extracted_relations(okf_results: list, chunks: list, inventory: list = None) -> list:
    """Helper to convert okf_results relations to filter_relations format,
    filter them, and rebuild the relationships in okf_results.

    inventory: optional record list used for the concept-name gate inside
    filter_relations. Defaults to okf_results itself; incremental paths pass
    the full merged corpus so relations targeting concepts from other
    documents survive, while co-occurrence is still validated against the
    chunks given (the new document's own text).
    """
    from okf.relations import filter_relations

    if not chunks:
        return okf_results

    if inventory is None:
        inventory = okf_results
    concepts = [r.get("concept_name") for r in inventory if r.get("concept_name")]
    
    for r in okf_results:
        concept_name = r.get("concept_name", "")
        if not concept_name:
            continue
            
        # 1. Prerequisites
        prereqs = []
        for p in r.get("prerequisites", []):
            if isinstance(p, str):
                prereqs.append({"source": concept_name, "target": p, "type": "prereq"})
        filtered_prereqs = filter_relations(prereqs, chunks, concepts)
        r["prerequisites"] = [x["target"] for x in filtered_prereqs]
        
        # 2. Unlocks
        unlocks = []
        for u in r.get("unlocks", []):
            if isinstance(u, str):
                unlocks.append({"source": concept_name, "target": u, "type": "unlock"})
        filtered_unlocks = filter_relations(unlocks, chunks, concepts)
        r["unlocks"] = [x["target"] for x in filtered_unlocks]
        
        # 3. Related
        related = []
        for rel in r.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                related.append({
                    "source": concept_name, 
                    "concept": rel["concept"], 
                    "relation": rel.get("relation", "related"), 
                    "type": "related"
                })
        filtered_related = filter_relations(related, chunks, concepts)
        r["related_to"] = [
            {"concept": x["concept"], "relation": x["relation"]} 
            for x in filtered_related
        ]
        
    return okf_results


def finalize_and_build(okf_results: list, total_chunks: int,
                       successful_chunk_count: int, chunks: list = None,
                       evaluate_gold_path: str = None):
    """Run every stage after extraction: save raw results, cleanup,
    canonicalization, KuzuDB graph build, exports and evaluation.

    Shared by run_pipeline (full / --resume runs) and add_document
    (incremental --add ingestion) so downstream behavior stays identical.
    """
    saved_file = BASE_DIR / "okf_results.json"

    # Save raw extraction results (anchored to BASE_DIR so resume-read and this
    # write always target the same file regardless of the current directory).
    with open(saved_file, "w", encoding="utf-8") as f:
        json.dump(okf_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to okf_results.json")

    # Integrate cleanup.clean_pipeline and filter_extracted_relations after extraction
    from okf.cleanup import clean_pipeline
    if chunks:
        okf_results = clean_pipeline(okf_results, chunks)
        okf_results = filter_extracted_relations(okf_results, chunks)

    okf_results = cleanup_and_canonicalize(okf_results)

    # Apply co-mention RELATED builder and UNLOCKS heuristics
    from okf.co_mention import build_co_mention_edges
    from okf.unlocks_heuristics import add_heuristic_unlocks
    if not chunks:
        pdf_chunks_path = BASE_DIR / "pdf_chunks.json"
        if pdf_chunks_path.exists():
            try:
                with open(pdf_chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load pdf_chunks.json: {e}")
    if chunks:
        okf_results = build_co_mention_edges(okf_results, chunks)
        okf_results = add_heuristic_unlocks(okf_results, chunks)

    # ── Stage 4: KùzuDB Graph Ingestion ──
    print(f"\n[4] STAGE 4: KuzuDB Graph Ingestion (MERGE)")
    print("-" * 50)

    conn, db, graph_export = ingest_to_kuzu(okf_results, db_path=str(BASE_DIR / "okf_graph.db"))

    # Call evaluate.structural_audit(db) before finalizing, and raise exception if cycles or self-edges are detected.
    from okf.evaluate import structural_audit
    audit_res = structural_audit(db)
    if audit_res.get("self_edges") or audit_res.get("cycles"):
        raise ValueError(
            f"Structural audit failed: self-edges or cycles detected in KuzuDB! "
            f"Self-edges: {audit_res.get('self_edges')}, Cycles: {audit_res.get('cycles')}"
        )

    # Save every downstream artifact (graph_audit.json, root + graph_ui
    # okf_graph.json, _graph_nodes/_graph_edges, accuracy.json) via the shared
    # writer so this path and the live ingestion worker can never drift.
    # All artifacts are anchored to BASE_DIR so outputs land next to the
    # resume-read inputs no matter which directory the pipeline runs from.
    from okf.exports import write_all_artifacts
    graph_audit, accuracy = write_all_artifacts(
        graph_export, okf_results, db,
        total_chunks=total_chunks,
        successful_chunk_count=successful_chunk_count,
    )

    # ── Stage 5: Evaluation ──
    print(f"\n[5] STAGE 5: Accuracy Evaluation")
    print("-" * 50)

    # If evaluate_gold_path is provided, run the comparison against gold-standard graphs
    if evaluate_gold_path:
        print(f"\n[5a] Comparing against gold standard: {evaluate_gold_path}")
        from okf.evaluate import evaluate_pipeline, print_report
        report = evaluate_pipeline(db, evaluate_gold_path)
        print_report(report)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  >> Overall Accuracy Score: {accuracy['overall_score']}%")
    print(f"\n  Breakdown:")
    for metric, value in accuracy["breakdown"].items():
        bar = "#" * int(value / 5) + "." * (20 - int(value / 5))
        print(f"    {metric:30s} {bar} {value}%")

    print(f"\n  Concept Type Distribution:")
    for t, count in accuracy["distributions"]["concept_types"].items():
        print(f"    {t:20s}: {count}")

    print(f"\n  Difficulty Distribution:")
    for d, count in accuracy["distributions"]["difficulty_levels"].items():
        print(f"    {d:20s}: {count}")

    print(f"\n  Graph Stats:")
    for k, v in accuracy["stats"].items():
        print(f"    {k:30s}: {v}")

    # Print some sample concept mappings
    print(f"\n{'=' * 70}")
    print(f"SAMPLE EXTRACTED CONCEPTS")
    print(f"{'=' * 70}")
    for result in okf_results[:10]:
        print(f"\n  [{result.get('concept_type', '?'):10s}] {result['concept_name']}")
        print(f"    Difficulty: {result.get('difficulty', '?')}")
        print(f"    Summary: {result.get('summary', '')[:80]}...")
        if result.get("prerequisites"):
            print(f"    Requires: {', '.join(result['prerequisites'][:5])}")
        if result.get("unlocks"):
            print(f"    Unlocks: {', '.join(result['unlocks'][:5])}")
        if result.get("related_to"):
            for rel in result["related_to"][:3]:
                if isinstance(rel, dict):
                    print(f"    {rel.get('relation', '?'):15s} -> {rel.get('concept', '?')}")
        if result.get("tags"):
            print(f"    Tags: {', '.join(result['tags'][:5])}")

    print(f"\n{'=' * 70}")
    print(f"GRAPH EDGES (sample)")
    print(f"{'=' * 70}")
    for edge in graph_export["edges"][:20]:
        arrow = "--requires-->" if edge["edge_type"] == "REQUIRES" else \
                "--unlocks--->" if edge["edge_type"] == "UNLOCKS" else \
                f"--{edge['relation']:10s}->"
        print(f"  {edge['from_name'][:30]:30s} {arrow} {edge['to_name'][:30]}")

    print(f"\n{'=' * 70}")
    print(f"[OK] Pipeline complete! Files: okf_results.json, okf_graph.json, graph_audit.json, accuracy.json")
    print(f"{'=' * 70}")

    return okf_results, graph_export, accuracy


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path: str = None, resume: bool = False, local: bool = False, evaluate_gold_path: str = None):
    """Run the full Archipelago pipeline."""
    from pdf_ingestion import ingest_folder, ingest_document

    print("=" * 70)
    print("ARCHIPELAGO PIPELINE - PDF -> OKF v1.5 -> KuzuDB Graph RAG")
    print("=" * 70)

    # ── Stage 1: Chunking ──
    print("\n[1] STAGE 1: Section-Aware Document Chunking")
    print("-" * 50)

    if input_path is None:
        input_path = str(BASE_DIR / "pdfs")

    # MAX_PAGES_PER_DOC is read from okf.config at call time so the CLI shim's
    # --max-pages override (which mutates okf.config) is honored, matching the
    # old single-module global behavior.
    if os.path.isdir(input_path):
        chunks = ingest_folder(input_path, max_pages=config.MAX_PAGES_PER_DOC)
    else:
        chunks = ingest_document(input_path, max_pages=config.MAX_PAGES_PER_DOC)

    raw_chunks = list(chunks)
    all_chunk_count = len(chunks)

    # Only prose chunks reach the SLM. Tables, references, front-matter and bare
    # math blocks are dropped here so they can't produce hallucinated nodes.
    kind_counts = Counter(c.get("chunk_kind", "prose") for c in chunks)
    prose_chunks = [c for c in chunks if c.get("chunk_kind", "prose") == "prose"]
    total_chunks = len(prose_chunks)

    print(f"\n  Total chunks: {all_chunk_count}  ->  {total_chunks} prose sent to SLM")
    dropped = {k: v for k, v in kind_counts.items() if k != "prose"}
    if dropped:
        print(f"  Dropped non-prose: {dropped}")

    if not prose_chunks:
        print("ERROR: No prose chunks to extract from!")
        return
    chunks = prose_chunks

    # ── Stage 2: OKF v1.5 Extraction ──
    if local:
        print(f"\n[2] STAGE 2: OKF v1.5 Extraction via local model: {_local_path}")
    else:
        print(f"\n[2] STAGE 2: OKF v1.5 Extraction via {MODEL_NAME}")
    print("-" * 50)

    saved_file = BASE_DIR / "okf_results.json"
    successful_chunk_count = 0
    if resume and saved_file.exists():
        print("  RESUMING from saved okf_results.json...")
        with open(saved_file, "r", encoding="utf-8") as f:
            okf_results = json.load(f)
        # Build map of chunks to restore page_number and section_title
        chunk_map = {(c["doc_id"], c["chunk_id"]): c for c in raw_chunks}
        # Fix any list-type concept_names from previous runs and restore chunk properties
        fixed = []
        for r in okf_results:
            cn = r.get("concept_name", "")
            if isinstance(cn, list):
                names = [n for n in cn if isinstance(n, str)]
                r["concept_name"] = names[0] if names else ""
            if r.get("doc_id") and not r.get("source_category"):
                r["source_category"] = infer_source_category(r.get("doc_id", ""))

            # Restore page_number, section_title, and source_passage from the current chunking pass
            doc_id = r.get("doc_id")
            chunk_id = r.get("chunk_id")
            if doc_id and chunk_id and (doc_id, chunk_id) in chunk_map:
                chunk = chunk_map[(doc_id, chunk_id)]
                r["page_number"] = chunk.get("page_number", 0)
                r["section_title"] = chunk.get("section_title", "")
                if not r.get("source_passage"):
                    r["source_passage"] = chunk.get("text", "")[:1600]

            if r.get("concept_name"):
                fixed.append(r)
        okf_results = fixed
        successful_chunk_count = len({
            (r.get("doc_id", ""), r.get("chunk_id", ""))
            for r in okf_results if r.get("chunk_id")
        })
        print(f"  Loaded {len(okf_results)} results (fixed list-type names)")
    else:
        if local:
            load_local_model()
        okf_results, successful_chunk_count = extract_chunks_with_model(chunks)
        print(f"\n  Extracted: {len(okf_results)} concepts from {total_chunks} chunks")

    return finalize_and_build(okf_results, total_chunks, successful_chunk_count, chunks=raw_chunks if resume else chunks, evaluate_gold_path=evaluate_gold_path)


# ---------------------------------------------------------------------------
# Incremental Ingestion (--add)
# ---------------------------------------------------------------------------
def compute_doc_id(path: str) -> str:
    """Derive the same doc_id the folder ingest would produce for this file.

    ingest_folder tags chunks with the path relative to the pdfs/ root (e.g.
    "papers/Edge2024_GraphRAG.pdf"); files outside pdfs/ get their basename.
    """
    resolved = Path(path).resolve()
    pdfs_root = (BASE_DIR / "pdfs").resolve()
    try:
        return str(resolved.relative_to(pdfs_root)).replace(os.sep, "/")
    except ValueError:
        return resolved.name


def add_document(path: str, limit: int = None, evaluate_gold_path: str = None):
    """Incrementally ingest ONE document with the LOCAL model and rebuild the
    merged graph. Re-adding a doc replaces its previous entries (no duplicates).

    limit: optional cap on prose chunks processed (fast testing on CPU).
    """
    from pdf_ingestion import ingest_document

    print("=" * 70)
    print("ARCHIPELAGO PIPELINE - INCREMENTAL ADD (single document)")
    print("=" * 70)

    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}")
        return

    doc_id = compute_doc_id(path)
    print(f"\n  Document: {path}")
    print(f"  doc_id:   {doc_id}")

    # ── Stage 1: Chunking (this document only) ──
    print("\n[1] STAGE 1: Section-Aware Document Chunking")
    print("-" * 50)

    chunks = ingest_document(path, max_pages=config.MAX_PAGES_PER_DOC)
    for chunk in chunks:
        chunk["doc_id"] = doc_id

    kind_counts = Counter(c.get("chunk_kind", "prose") for c in chunks)
    prose_chunks = [c for c in chunks if c.get("chunk_kind", "prose") == "prose"]
    new_chunk_total = len(prose_chunks)

    print(f"\n  Total chunks: {len(chunks)}  ->  {new_chunk_total} prose sent to SLM")
    dropped = {k: v for k, v in kind_counts.items() if k != "prose"}
    if dropped:
        print(f"  Dropped non-prose: {dropped}")

    if not prose_chunks:
        print("ERROR: No prose chunks to extract from!")
        return

    if limit is not None and limit < len(prose_chunks):
        print(f"  --limit {limit}: processing only the first {limit} prose chunks")
        prose_chunks = prose_chunks[:limit]

    # ── Stage 2: OKF v1.5 Extraction (LOCAL model only — never Ollama) ──
    print(f"\n[2] STAGE 2: OKF v1.5 Extraction via local model: {_local_path}")
    print("-" * 50)

    if extraction.LOCAL_MODEL is None:
        load_local_model()
    if not extraction.LOCAL_MODE or extraction.LOCAL_MODEL is None:
        print("ERROR: local model unavailable — --add never falls back to Ollama. Aborting.")
        return

    new_results, new_successful = extract_chunks_with_model(prose_chunks)
    print(f"\n  Extracted: {len(new_results)} concepts from {len(prose_chunks)} chunks")

    if not new_results:
        # Every chunk failed. Proceeding would delete the doc's previous
        # entries and replace them with nothing — abort instead so a bad run
        # can never destroy existing data.
        print("ERROR: extraction produced 0 concepts — leaving existing results untouched.")
        return

    # ── Merge with existing results (replace prior entries for this doc) ──
    print("\n[2a] MERGE WITH EXISTING RESULTS")
    print("-" * 50)

    saved_file = BASE_DIR / "okf_results.json"
    existing = []
    if saved_file.exists():
        with open(saved_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    before = len(existing)
    # Existing entries already carry page_number/section_title/source_passage;
    # leave them untouched — only this doc's records are replaced.
    existing = [r for r in existing if r.get("doc_id") != doc_id]
    replaced = before - len(existing)
    if replaced > 0:
        print(f"  Replaced {replaced} prior entries for {doc_id}")
    else:
        print(f"  No prior entries for {doc_id}")
    okf_results = existing + new_results
    print(f"  Merged total: {len(okf_results)} concepts "
          f"({len(existing)} existing + {len(new_results)} new)")

    # Chunk counts for the merged evaluation: distinct extracted chunks from the
    # kept existing results plus every prose chunk processed in this run.
    existing_chunk_count = len({
        (r.get("doc_id", ""), r.get("chunk_id", ""))
        for r in existing if r.get("chunk_id")
    })
    total_chunks = existing_chunk_count + len(prose_chunks)
    successful_chunk_count = existing_chunk_count + new_successful

    return finalize_and_build(okf_results, total_chunks, successful_chunk_count, chunks=prose_chunks, evaluate_gold_path=evaluate_gold_path)


# Re-export staged pipeline from okf.pipeline_staged
from okf.pipeline_staged import PipelineAborted, run_pipeline_staged

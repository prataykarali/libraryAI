"""Delete a document and unmerge its contribution from the live graph.

Policy
------
1. Remove the Document node, its Chunks, HAS_CHUNK, and MENTIONS edges.
2. Remove REQUIRES / UNLOCKS / RELATED edges whose ``source`` provenance
   belongs only to this document (``source`` starts with ``doc_id`` or
   ``doc_id:`` / ``doc_id/``).
3. Remove Concept nodes that have **no remaining MENTIONS** and **no remaining
   structural edges** after step 2 (true orphans introduced by that book).
4. Shared concepts used by other documents are kept; only this doc's
   mention/provenance edges are dropped.
5. Caller is responsible for ``graph_lock.write_lock()``, artifact rewrite,
   and inference reload/embeddings.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from okf.graph.common import _kuzu_escape
from okf.graph.export import export_graph
from okf.exports import build_visual_graph, build_graph_rag_index, write_all_artifacts

logger = logging.getLogger(__name__)


def _source_belongs_to_doc(source: str | None, doc_id: str) -> bool:
    """True when edge provenance points at this document."""
    if not source or not doc_id:
        return False
    s = str(source).replace("\\", "/")
    d = str(doc_id).replace("\\", "/")
    if s == d:
        return True
    if s.startswith(d + ":") or s.startswith(d + "/"):
        return True
    # bare filename match (upload jobs sometimes store basename only)
    base = d.rsplit("/", 1)[-1]
    if base and (s == base or s.startswith(base + ":") or s.startswith(base + "/")):
        return True
    return False


def list_documents(conn) -> list[dict]:
    """Return [{id, title, chunk_count}] for all Document nodes."""
    docs: dict[str, dict] = {}
    try:
        res = conn.execute("MATCH (d:Document) RETURN d.id, d.title")
        while res.has_next():
            did, title = res.get_next()
            docs[did] = {"id": did, "title": title or did, "chunk_count": 0}
        res = conn.execute(
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d.id, count(c)"
        )
        while res.has_next():
            did, n = res.get_next()
            if did in docs:
                docs[did]["chunk_count"] = int(n)
    except Exception as e:
        logger.warning("list_documents failed: %s", e)
    return sorted(docs.values(), key=lambda x: x["id"])


def delete_document_from_graph(conn, doc_id: str) -> dict[str, Any]:
    """Unmerge ``doc_id`` from an open Kuzu connection.

    Returns a stats dict. Raises ValueError if the document is not found.
    """
    safe = _kuzu_escape(doc_id)
    # Resolve actual document id (exact or basename / suffix match)
    resolved = None
    res = conn.execute("MATCH (d:Document) RETURN d.id")
    candidates = []
    while res.has_next():
        candidates.append(res.get_next()[0])
    for c in candidates:
        if c == doc_id or c.replace("\\", "/") == doc_id.replace("\\", "/"):
            resolved = c
            break
    if resolved is None:
        base = doc_id.replace("\\", "/").rsplit("/", 1)[-1]
        for c in candidates:
            if c.replace("\\", "/").rsplit("/", 1)[-1] == base:
                resolved = c
                break
    if resolved is None:
        raise ValueError(f"Document not found: {doc_id}")

    safe = _kuzu_escape(resolved)
    stats = {
        "doc_id": resolved,
        "chunks_deleted": 0,
        "mentions_deleted": 0,
        "edges_deleted": 0,
        "concepts_deleted": 0,
        "concepts_retained": 0,
    }

    # Concepts currently mentioned by this document (before delete)
    mentioned_ids: set[str] = set()
    res = conn.execute(
        f"""
        MATCH (d:Document {{id: '{safe}'}})-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->(c:Concept)
        RETURN DISTINCT c.id
        """
    )
    while res.has_next():
        mentioned_ids.add(res.get_next()[0])

    # 1) Structural edges whose provenance is this document
    for rel in ("REQUIRES", "UNLOCKS", "RELATED"):
        try:
            res = conn.execute(
                f"MATCH (a:Concept)-[r:{rel}]->(b:Concept) RETURN a.id, b.id, r.source"
            )
            to_drop = []
            while res.has_next():
                a_id, b_id, src = res.get_next()
                if _source_belongs_to_doc(src, resolved):
                    to_drop.append((a_id, b_id))
            for a_id, b_id in to_drop:
                a_e, b_e = _kuzu_escape(a_id), _kuzu_escape(b_id)
                try:
                    conn.execute(
                        f"MATCH (a:Concept {{id: '{a_e}'}})-[r:{rel}]->"
                        f"(b:Concept {{id: '{b_e}'}}) DELETE r"
                    )
                    stats["edges_deleted"] += 1
                except Exception as e:
                    logger.warning("edge delete %s %s->%s: %s", rel, a_id, b_id, e)
        except Exception as e:
            logger.warning("edge scan %s failed: %s", rel, e)

    # 2) MENTIONS from this document's chunks
    try:
        res = conn.execute(
            f"""
            MATCH (d:Document {{id: '{safe}'}})-[:HAS_CHUNK]->(chk:Chunk)-[m:MENTIONS]->(c:Concept)
            RETURN count(m)
            """
        )
        if res.has_next():
            stats["mentions_deleted"] = int(res.get_next()[0] or 0)
        conn.execute(
            f"""
            MATCH (d:Document {{id: '{safe}'}})-[:HAS_CHUNK]->(chk:Chunk)-[m:MENTIONS]->(:Concept)
            DELETE m
            """
        )
    except Exception as e:
        logger.warning("mentions delete failed: %s", e)

    # 3) Chunks + HAS_CHUNK
    try:
        res = conn.execute(
            f"MATCH (d:Document {{id: '{safe}'}})-[:HAS_CHUNK]->(chk:Chunk) RETURN count(chk)"
        )
        if res.has_next():
            stats["chunks_deleted"] = int(res.get_next()[0] or 0)
        # Detach-style: delete rel then chunk
        conn.execute(
            f"""
            MATCH (d:Document {{id: '{safe}'}})-[h:HAS_CHUNK]->(chk:Chunk)
            DELETE h
            """
        )
        # Delete orphaned chunks that were only linked to this doc
        # (chunk ids are typically unique per doc)
        res = conn.execute(
            f"""
            MATCH (chk:Chunk)
            WHERE NOT (:Document)-[:HAS_CHUNK]->(chk)
            RETURN chk.id
            """
        )
        orphan_chunks = []
        while res.has_next():
            orphan_chunks.append(res.get_next()[0])
        for cid in orphan_chunks:
            try:
                conn.execute(
                    f"MATCH (chk:Chunk {{id: '{_kuzu_escape(cid)}'}}) DELETE chk"
                )
            except Exception as e:
                logger.warning("chunk delete %s: %s", cid, e)
    except Exception as e:
        logger.warning("chunk delete failed: %s", e)

    # 4) Document node
    try:
        conn.execute(f"MATCH (d:Document {{id: '{safe}'}}) DELETE d")
    except Exception as e:
        raise RuntimeError(f"Failed to delete Document node: {e}") from e

    # 5) Orphan concepts: were mentioned by this doc, now have zero MENTIONS
    #    and zero structural edges
    for cid in mentioned_ids:
        c_e = _kuzu_escape(cid)
        try:
            res = conn.execute(
                f"MATCH (chk:Chunk)-[:MENTIONS]->(c:Concept {{id: '{c_e}'}}) RETURN count(chk)"
            )
            mentions_left = int(res.get_next()[0]) if res.has_next() else 0
            if mentions_left > 0:
                stats["concepts_retained"] += 1
                continue
            res = conn.execute(
                f"""
                MATCH (c:Concept {{id: '{c_e}'}})-[r:REQUIRES|UNLOCKS|RELATED]-(:Concept)
                RETURN count(r)
                """
            )
            edges_left = int(res.get_next()[0]) if res.has_next() else 0
            if edges_left > 0:
                stats["concepts_retained"] += 1
                continue
            conn.execute(f"MATCH (c:Concept {{id: '{c_e}'}}) DELETE c")
            stats["concepts_deleted"] += 1
        except Exception as e:
            logger.warning("orphan concept check %s: %s", cid, e)
            stats["concepts_retained"] += 1

    return stats


def filter_okf_results(okf_results: list, doc_id: str) -> list:
    """Drop extraction records belonging to ``doc_id``."""
    out = []
    for r in okf_results or []:
        rid = (r.get("doc_id") or "").replace("\\", "/")
        d = doc_id.replace("\\", "/")
        base = d.rsplit("/", 1)[-1]
        if rid == d or rid.endswith("/" + base) or rid == base:
            continue
        # Also drop edge provenance-only records if present
        out.append(r)
    return out


def rebuild_artifacts_after_delete(db, base_dir: Path, okf_results: list | None = None) -> dict:
    """Re-export graph JSON artifacts from the live DB after a delete."""
    import kuzu

    if hasattr(db, "execute"):
        conn = db
    else:
        conn = kuzu.Connection(db)

    graph_export = export_graph(conn)
    if okf_results is None:
        # Reconstruct minimal results from graph sources for accuracy/viz richness
        okf_results = []
        for cid, c in (graph_export.get("concepts") or {}).items():
            for src in c.get("sources") or []:
                okf_results.append({
                    "concept_name": c.get("name") or cid,
                    "concept_type": c.get("concept_type") or "definition",
                    "difficulty": c.get("difficulty") or "intermediate",
                    "summary": c.get("summary") or "",
                    "doc_id": src.get("doc_id") or "",
                    "chunk_id": src.get("chunk_id") or "",
                    "page_number": src.get("page_number") or 0,
                    "section_title": src.get("section_title") or "",
                    "prerequisites": [],
                    "unlocks": [],
                    "related_to": [],
                    "tags": [],
                })

    graph_export["visualization"] = build_visual_graph(okf_results, graph_export)
    graph_export["graph_rag_index"] = build_graph_rag_index(okf_results, graph_export)

    # Persist okf_results filtered
    results_path = Path(base_dir) / "okf_results.json"
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(okf_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("could not write okf_results.json: %s", e)

    write_all_artifacts(graph_export, okf_results, db, base_dir=Path(base_dir))
    return graph_export


def delete_document_end_to_end(
    db,
    doc_id: str,
    *,
    base_dir: Path,
    okf_results_path: Path | None = None,
    remove_pdf: bool = False,
    pdf_dir: Path | None = None,
) -> dict[str, Any]:
    """Full unmerge: Kuzu delete + artifact rewrite. Caller holds write lock."""
    import kuzu

    if hasattr(db, "execute"):
        conn = db
    else:
        conn = kuzu.Connection(db)

    stats = delete_document_from_graph(conn, doc_id)
    resolved = stats["doc_id"]

    # Filter okf_results
    results_path = okf_results_path or (Path(base_dir) / "okf_results.json")
    okf_results = []
    if results_path.exists():
        try:
            okf_results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            okf_results = []
    okf_results = filter_okf_results(okf_results, resolved)

    graph_export = rebuild_artifacts_after_delete(db, Path(base_dir), okf_results)
    stats["concepts_remaining"] = len(graph_export.get("concepts") or {})
    stats["edges_remaining"] = len(graph_export.get("edges") or [])

    if remove_pdf and pdf_dir:
        for candidate in (
            Path(pdf_dir) / resolved,
            Path(pdf_dir) / Path(resolved).name,
        ):
            try:
                if candidate.is_file():
                    os.remove(candidate)
                    stats["pdf_removed"] = str(candidate)
                    break
            except Exception as e:
                stats["pdf_remove_error"] = str(e)

    return stats

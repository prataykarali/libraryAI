"""Shared low-level helpers: concept IDs and provenance record utilities.

Kept in a leaf module (imports config only) because they are needed by
cleanup, graph_db and exports alike — placing them any higher would create
import cycles (e.g. ingest_to_kuzu -> build_visual_graph -> create_concept_id).
"""

import re

from okf.config import infer_source_category


def create_concept_id(name: str) -> str:
    """Generate a stable, deterministic ID from a concept name."""
    cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
    cid = re.sub(r'_+', '_', cid).strip('_')
    return cid or 'concept'


def _source_record(result: dict) -> dict:
    return {
        "doc_id": result.get("doc_id", ""),
        "source_category": result.get("source_category") or infer_source_category(result.get("doc_id", "")),
        "chunk_id": result.get("chunk_id", ""),
        "page_number": result.get("page_number", 0),
        "section_title": result.get("section_title", ""),
        "text_passage": result.get("source_passage", ""),
    }


def _dedupe_dicts(items: list) -> list:
    seen = set()
    deduped = []
    for item in items:
        key = tuple(sorted(item.items()))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _record_sources(result: dict) -> list:
    """All provenance records for a single result.

    Once records have been through merge_duplicate_results they carry an
    accumulated ``sources`` list (the union of every duplicate's provenance);
    return that when present so unioned evidence is never collapsed back to a
    single record. Otherwise fall back to the record's own source.
    """
    existing = result.get("sources")
    if isinstance(existing, list) and existing:
        return [s for s in existing if isinstance(s, dict)]
    return [_source_record(result)]

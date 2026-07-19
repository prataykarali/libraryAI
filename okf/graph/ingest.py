"""ingest.py — Kùzu graph submodule."""
from okf.graph.common import (
    _DEFAULT_DB_PATH, _kuzu_escape, _kuzu_literal, _SCHEMA_DDL,
    _create_schema, _migrate_schema, logger, build_visual_graph,
    build_graph_rag_index,
)
from okf.cleanup import is_valid_concept_name
from okf.util import create_concept_id, _source_record
from okf.config import infer_source_category
from okf.graph.export import enforce_dag, export_graph
import os
import logging
# logger may come from common
try:
    logger
except NameError:
    logger = logging.getLogger(__name__)


# ── Module-level helpers for librarian manual API (conn-first) ──────────────

_RELATION_TO_TABLE = {
    "requires": ("REQUIRES", "requires"),
    "enables": ("UNLOCKS", "enables"),
    "uses": ("RELATED", "uses"),
    "extends": ("RELATED", "extends"),
    "part_of": ("RELATED", "part_of"),
    "contrasts_with": ("RELATED", "contrasts_with"),
    "evaluated_by": ("RELATED", "evaluated_by"),
}


def ensure_concept(
    conn,
    concept_id: str,
    name: str,
    concept_type: str = "definition",
    difficulty: str = "intermediate",
    summary: str = "",
    tags=None,
) -> str:
    """Upsert a Concept node. Used by librarian manual API and tests.

    Signature: ``ensure_concept(conn, concept_id, name, ...)``.
    Returns the concept id.
    """
    if not concept_id:
        concept_id = create_concept_id(name)
    safe_id = _kuzu_escape(concept_id)
    safe_name = _kuzu_escape(name or concept_id)
    safe_type = _kuzu_escape(concept_type or "definition")
    safe_diff = _kuzu_escape(difficulty or "intermediate")
    safe_summary = _kuzu_escape((summary or "")[:500])
    try:
        conn.execute(
            f"""
            MERGE (c:Concept {{id: '{safe_id}'}})
            ON CREATE SET c.name = '{safe_name}',
                          c.concept_type = '{safe_type}',
                          c.difficulty = '{safe_diff}',
                          c.summary = '{safe_summary}'
            ON MATCH SET c.name = '{safe_name}',
                         c.concept_type = '{safe_type}',
                         c.difficulty = '{safe_diff}',
                         c.summary = '{safe_summary}'
            """
        )
    except Exception as e:
        logger.warning("ensure_concept failed for %s: %s", concept_id, e)
        raise
    return concept_id


def create_edge(conn, from_id: str, to_id: str, relation: str, source: str = "manual:librarian") -> bool:
    """Create a structural edge between two concepts.

    ``relation`` is a librarian-facing name (requires/enables/uses/…).
    Maps to REQUIRES / UNLOCKS / RELATED tables.
    """
    if not from_id or not to_id or from_id == to_id:
        return False
    key = (relation or "").strip().lower()
    mapping = _RELATION_TO_TABLE.get(key)
    if not mapping:
        # Allow direct table names
        if key.upper() in ("REQUIRES", "UNLOCKS", "RELATED"):
            rel_table, rel_type = key.upper(), key.lower()
        else:
            raise ValueError(f"Unknown relation: {relation}")
    else:
        rel_table, rel_type = mapping

    if rel_table == "REQUIRES":
        # from requires to  => to is prereq of from; learn-order: to -> from
        enforce_dag(conn, to_id, from_id)
    elif rel_table == "UNLOCKS":
        enforce_dag(conn, from_id, to_id)

    safe_from = _kuzu_escape(from_id)
    safe_to = _kuzu_escape(to_id)
    safe_source = _kuzu_escape(source or "manual")
    safe_rel = _kuzu_escape(rel_type)
    try:
        conn.execute(
            f"""
            MATCH (a:Concept {{id: '{safe_from}'}}),
                  (b:Concept {{id: '{safe_to}'}})
            MERGE (a)-[r:{rel_table}]->(b)
            ON CREATE SET r.relation_type = '{safe_rel}', r.source = '{safe_source}'
            """
        )
        return True
    except Exception as e:
        logger.warning("create_edge failed %s %s->%s: %s", rel_table, from_id, to_id, e)
        raise


def ingest_to_kuzu(okf_results: list, db_path: str = _DEFAULT_DB_PATH):
    """Ingest OKF results into KùzuDB with normalized schema and MERGE semantics."""
    import kuzu
    import shutil

    # Clean existing DB
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            shutil.rmtree(db_path, ignore_errors=True)

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Create schema (fresh DB — CREATEs carry the full column set) and run the
    # idempotent migration so pre-existing DBs pick up new columns too.
    print("  Creating graph schema...")
    _create_schema(conn)
    _migrate_schema(conn)

    # Best-effort summary lookup for placeholder nodes created from edge refs.
    _summary_lookup = {}
    for _r in okf_results:
        _name = _r.get("concept_name", "")
        _s = _r.get("summary", "")
        if _s and len(_s) > len(_summary_lookup.get(_name, "")):
            _summary_lookup[_name] = _s

    def ensure_document(doc_id):
        if not doc_id:
            return
        safe_doc_id = _kuzu_escape(doc_id)
        try:
            conn.execute(f"MERGE (d:Document {{id: '{safe_doc_id}'}})")
        except Exception as e:
            logger.warning("ensure_document failed for %s: %s", doc_id, e)

    def ensure_chunk(doc_id, chunk_id, page_number, section_title, text_passage,
                     text_offset_start=None, text_offset_end=None,
                     block_x=None, block_y=None, block_w=None, block_h=None):
        if not doc_id or not chunk_id:
            return ""
        cid = f"{doc_id}_{chunk_id}"
        safe_cid = _kuzu_escape(cid)
        safe_doc_id = _kuzu_escape(doc_id)
        safe_chunk_id = _kuzu_escape(chunk_id)
        safe_section = _kuzu_escape(section_title)
        safe_passage = _kuzu_escape(text_passage)

        try:
            conn.execute(f"""
                MERGE (c:Chunk {{id: '{safe_cid}'}})
                ON CREATE SET c.chunk_id = '{safe_chunk_id}',
                              c.page_number = {page_number},
                              c.section_title = '{safe_section}',
                              c.text_passage = '{safe_passage}',
                              c.text_offset_start = {_kuzu_literal(text_offset_start)},
                              c.text_offset_end = {_kuzu_literal(text_offset_end)},
                              c.block_x = {_kuzu_literal(block_x)},
                              c.block_y = {_kuzu_literal(block_y)},
                              c.block_w = {_kuzu_literal(block_w)},
                              c.block_h = {_kuzu_literal(block_h)}
            """)
            conn.execute(f"""
                MATCH (d:Document {{id: '{safe_doc_id}'}}),
                      (c:Chunk {{id: '{safe_cid}'}})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """)
        except Exception as e:
            logger.warning("ensure_chunk failed for %s -> %s: %s", doc_id, cid, e)
        return cid

    def ensure_concept(name, concept_type="definition", difficulty="intermediate",
                       summary="", is_placeholder=False):
        if not is_valid_concept_name(name):
            return ""
        cid = create_concept_id(name)

        safe_name = _kuzu_escape(name)
        safe_summary = _kuzu_escape((summary or _summary_lookup.get(name, ""))[:500])

        try:
            if is_placeholder:
                conn.execute(f"""
                    MERGE (c:Concept {{id: '{cid}'}})
                    ON CREATE SET c.name = '{safe_name}',
                                  c.concept_type = '{concept_type}',
                                  c.difficulty = '{difficulty}',
                                  c.summary = '{safe_summary}'
                """)
            else:
                conn.execute(f"""
                    MERGE (c:Concept {{id: '{cid}'}})
                    ON CREATE SET c.name = '{safe_name}',
                                  c.concept_type = '{concept_type}',
                                  c.difficulty = '{difficulty}',
                                  c.summary = '{safe_summary}'
                    ON MATCH SET c.name = '{safe_name}',
                                 c.concept_type = '{concept_type}',
                                 c.difficulty = '{difficulty}',
                                 c.summary = '{safe_summary}'
                """)
        except Exception as e:
            logger.warning("ensure_concept failed for %s (%s): %s", name, cid, e)
        return cid

    def link_chunk_concept(chunk_db_id, concept_id):
        if not chunk_db_id or not concept_id:
            return
        safe_chunk_db_id = _kuzu_escape(chunk_db_id)
        try:
            conn.execute(f"""
                MATCH (c:Chunk {{id: '{safe_chunk_db_id}'}}),
                      (con:Concept {{id: '{concept_id}'}})
                MERGE (c)-[:MENTIONS]->(con)
            """)
        except Exception as e:
            logger.warning(
                "link_chunk_concept failed: %s -> %s: %s",
                chunk_db_id, concept_id, e,
            )

    def create_edge(from_id, to_id, rel_table, rel_type, source):
        """Create a relationship edge using MERGE to prevent duplicate edges."""
        if from_id == to_id:
            logger.warning(f"Self-loop detected and skipped: {from_id} -> {to_id}")
            return False
        
        try:
            if rel_table == "REQUIRES":
                enforce_dag(conn, to_id, from_id)
            elif rel_table == "UNLOCKS":
                enforce_dag(conn, from_id, to_id)
        except ValueError as e:
            logger.warning(f"Cycle detected and edge skipped: {e}")
            return False

        safe_source = _kuzu_escape(source)
        try:
            conn.execute(f"""
                MATCH (a:Concept {{id: '{from_id}'}}),
                      (b:Concept {{id: '{to_id}'}})
                MERGE (a)-[r:{rel_table}]->(b)
                ON CREATE SET r.relation_type = '{rel_type}', r.source = '{safe_source}'
            """)
            return True
        except Exception as e:
            logger.warning(
                "create_edge failed: %s %s -> %s (%s): %s",
                rel_table, from_id, to_id, rel_type, e,
            )
            return False

    # Ingest all concepts, documents, chunks, and links
    print("  Ingesting nodes and chunk associations...")
    unique_concept_ids = set()
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue

        # 3. Ensure Concept
        concept_id = ensure_concept(
            name,
            result.get("concept_type", "definition"),
            result.get("difficulty", "intermediate"),
            result.get("summary", "")
        )
        if concept_id:
            unique_concept_ids.add(concept_id)

        # Get all sources (provenance records) accumulated during merging
        sources = result.get("sources")
        if not (isinstance(sources, list) and sources):
            sources = [_source_record(result)]

        for src in sources:
            doc_id = src.get("doc_id", "")
            chunk_id = src.get("chunk_id", "")
            page_number = int(src.get("page_number", 0))
            section_title = src.get("section_title", "")
            text_passage = src.get("text_passage", "")

            if not doc_id or not chunk_id:
                continue

            # 1. Ensure Document
            ensure_document(doc_id)

            # 2. Ensure Chunk & Link Document -> Chunk
            chunk_db_id = ensure_chunk(
                doc_id, chunk_id, page_number, section_title, text_passage,
                text_offset_start=src.get("text_offset_start"),
                text_offset_end=src.get("text_offset_end"),
                block_x=src.get("block_x"), block_y=src.get("block_y"),
                block_w=src.get("block_w"), block_h=src.get("block_h"),
            )

            # 4. Link Chunk -> Concept
            if chunk_db_id and concept_id:
                link_chunk_concept(chunk_db_id, concept_id)

        # Relation targets are NOT minted as nodes here. A target only gets an
        # edge if some record actually extracted it (checked in the edge loop
        # against the canonical inventory), so hallucinated placeholder nodes
        # (e.g. 'Grande Jura Distribution') can never enter the graph.

    print(f"    -> {len(unique_concept_ids)} unique concept nodes")

    # Create edges
    print("  Creating relationship edges...")
    # Canonical concept inventory: only names some record actually extracted
    # may be edge targets. Anything else is skipped and counted.
    extracted_names = {
        r.get("concept_name", "").strip().lower()
        for r in okf_results if r.get("concept_name")
    }
    skipped_unknown_targets = 0
    edge_count = 0
    seen_related_pairs = set()
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        concept_id = create_concept_id(name)
        # Default provenance: this record's own chunk. Individual relations may
        # override via relation_provenance (set by merges/relation_pass), which
        # records the (doc_id, chunk_id) of the record that ASSERTED them.
        source_info = f"{result.get('doc_id', '')}:{result.get('chunk_id', '')}"
        rel_prov = result.get("relation_provenance") or {}

        for prereq in result.get("prerequisites", []):
            if isinstance(prereq, str) and prereq.strip() and is_valid_concept_name(prereq):
                if prereq.strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                prereq_id = create_concept_id(prereq)
                src = rel_prov.get(f"prereq:{prereq.lower()}", source_info)
                if create_edge(concept_id, prereq_id, "REQUIRES", "requires", src):
                    edge_count += 1

        for unlock in result.get("unlocks", []):
            if isinstance(unlock, str) and unlock.strip() and is_valid_concept_name(unlock):
                if unlock.strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                unlock_id = create_concept_id(unlock)
                src = rel_prov.get(f"unlock:{unlock.lower()}", source_info)
                if create_edge(concept_id, unlock_id, "UNLOCKS", "enables", src):
                    edge_count += 1

        for rel in result.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept") and is_valid_concept_name(rel["concept"]):
                if rel["concept"].strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                rel_id = create_concept_id(rel["concept"])
                rel_type = rel.get("relation", "related")
                # Symmetric-ish relations (uses/contrasts_with/variant_of) get
                # extracted from both endpoints; keep a single direction so the
                # UI doesn't draw the same association twice.
                sym_key = (min(concept_id, rel_id), max(concept_id, rel_id), rel_type)
                if sym_key in seen_related_pairs:
                    continue
                seen_related_pairs.add(sym_key)
                src = rel_prov.get(f"related:{rel['concept'].lower()}", source_info)
                if create_edge(concept_id, rel_id, "RELATED", rel_type, src):
                    edge_count += 1

    print(f"    -> {edge_count} edges created")
    if skipped_unknown_targets:
        print(f"    -> {skipped_unknown_targets} edges skipped "
              f"(target not in extracted concept inventory)")

    # Export graph to JSON
    print("  Exporting graph...")
    export = export_graph(conn)
    export["visualization"] = build_visual_graph(okf_results, export)
    export["graph_rag_index"] = build_graph_rag_index(okf_results, export)

    return conn, db, export

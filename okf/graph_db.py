"""KùzuDB graph ingestion with MERGE semantics."""

import os

from okf.cleanup import is_valid_concept_name
from okf.config import infer_source_category
from okf.exports import build_graph_rag_index, build_visual_graph
from okf.util import _source_record, create_concept_id


def _kuzu_escape(value: str) -> str:
    """Escape a string literal for inline Kuzu Cypher queries.

    Kuzu uses backslash escaping (\\') — NOT SQL-style doubled quotes ('').
    Doubled quotes raise a parser exception, which ensure_concept/ensure_chunk
    silently swallowed, dropping any node/chunk whose text contained an
    apostrophe (root cause of phantom viz nodes missing from the concepts dict).
    """
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def ingest_to_kuzu(okf_results: list, db_path: str = "okf_graph.db"):
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

    # Create schema
    print("  Creating graph schema...")
    try:
        conn.execute("""
            CREATE NODE TABLE Document (
                id STRING PRIMARY KEY
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE NODE TABLE Chunk (
                id STRING PRIMARY KEY,
                chunk_id STRING,
                page_number INT64,
                section_title STRING,
                text_passage STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE NODE TABLE Concept (
                id STRING PRIMARY KEY,
                name STRING,
                concept_type STRING,
                difficulty STRING,
                summary STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE HAS_CHUNK (
                FROM Document TO Chunk
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE MENTIONS (
                FROM Chunk TO Concept
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE REQUIRES (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE UNLOCKS (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE RELATED (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

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
        except Exception:
            pass

    def ensure_chunk(doc_id, chunk_id, page_number, section_title, text_passage):
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
                              c.text_passage = '{safe_passage}'
            """)
            conn.execute(f"""
                MATCH (d:Document {{id: '{safe_doc_id}'}}),
                      (c:Chunk {{id: '{safe_cid}'}})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """)
        except Exception:
            pass
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
        except Exception:
            pass
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
        except Exception:
            pass

    def create_edge(from_id, to_id, rel_table, rel_type, source):
        """Create a relationship edge using MERGE to prevent duplicate edges."""
        if from_id == to_id:
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
        except Exception:
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
            chunk_db_id = ensure_chunk(doc_id, chunk_id, page_number, section_title, text_passage)

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


def export_graph(conn) -> dict:
    """Export the full graph structure to a dict, reconstructing concept sources from chunk relationships."""
    # Get concepts, documents, chunks, and their mentions
    result = conn.execute("""
        MATCH (c:Concept)
        OPTIONAL MATCH (chk:Chunk)-[:MENTIONS]->(c)
        OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(chk)
        RETURN c.id, c.name, c.concept_type, c.difficulty, c.summary, doc.id, chk.chunk_id, chk.page_number, chk.section_title, chk.text_passage
        ORDER BY c.name
    """)

    concepts = {}
    while result.has_next():
        row = result.get_next()
        cid = row[0]
        name = row[1]
        concept_type = row[2]
        difficulty = row[3]
        summary = row[4]

        doc_id = row[5]
        chunk_id = row[6]
        page_number = row[7]
        section_title = row[8]
        text_passage = row[9]

        if cid not in concepts:
            concepts[cid] = {
                "name": name,
                "concept_type": concept_type,
                "difficulty": difficulty,
                "summary": summary,
                "sources": []
            }

        if doc_id:
            source_rec = {
                "doc_id": doc_id,
                "source_category": infer_source_category(doc_id),
                "chunk_id": chunk_id,
                "page_number": int(page_number) if page_number is not None else 0,
                "section_title": section_title,
                "text_passage": text_passage
            }
            if source_rec not in concepts[cid]["sources"]:
                concepts[cid]["sources"].append(source_rec)

    # Get all edges
    edges = []
    for rel_table in ["REQUIRES", "UNLOCKS", "RELATED"]:
        try:
            result = conn.execute(f"""
                MATCH (a:Concept)-[r:{rel_table}]->(b:Concept)
                RETURN a.id, a.name, r.relation_type, b.id, b.name, r.source
            """)
            while result.has_next():
                row = result.get_next()
                edges.append({
                    "from_id": row[0],
                    "from_name": row[1],
                    "relation": row[2],
                    "to_id": row[3],
                    "to_name": row[4],
                    "source": row[5],
                    "edge_type": rel_table
                })
        except Exception:
            pass

    return {
        "concepts": concepts,
        "edges": edges,
        "stats": {
            "total_concepts": len(concepts),
            "total_edges": len(edges),
            "requires_edges": sum(1 for e in edges if e["edge_type"] == "REQUIRES"),
            "unlocks_edges": sum(1 for e in edges if e["edge_type"] == "UNLOCKS"),
            "related_edges": sum(1 for e in edges if e["edge_type"] == "RELATED"),
        }
    }

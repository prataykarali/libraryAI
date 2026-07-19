"""evidence.py — Kùzu graph submodule."""
import json
from okf.graph.common import (
    _DEFAULT_DB_PATH, _kuzu_escape, _kuzu_literal, _SCHEMA_DDL,
    _create_schema, _migrate_schema, logger, BASE_DIR,
)
from okf.util import create_concept_id
import os

_EVIDENCE_RETURN = """
RETURN chk.chunk_id, doc.id, chk.page_number, chk.section_title, chk.text_passage,
       chk.text_offset_start, chk.text_offset_end, chk.block_x, chk.block_y,
       chk.block_w, chk.block_h, doc.title, doc.page_label_map
"""



def _row_to_evidence(row) -> dict:
    """Map an _EVIDENCE_RETURN result row to the shared evidence contract."""
    block_vals = row[7:11]
    block_bbox = [float(v) for v in block_vals] if all(
        v is not None for v in block_vals) else None
    page_label_map = None
    if row[12]:
        try:
            parsed = json.loads(row[12])
            if isinstance(parsed, dict) and parsed:
                page_label_map = parsed
        except (ValueError, TypeError):
            pass
    return {
        "chunk_id": row[0],
        "doc_id": row[1],
        "page_number": int(row[2]) if row[2] is not None else 0,
        "section_title": row[3],
        "text": row[4],
        "text_offset_start": int(row[5]) if row[5] is not None else None,
        "text_offset_end": int(row[6]) if row[6] is not None else None,
        "block_bbox": block_bbox,
        "doc_title": row[11],
        "page_label_map": page_label_map,
    }

# Test-injected / process-wide defaults (see set_default_connection).
_DEFAULT_CONN = None
_DEFAULT_GRAPH_DB = None


class GraphDB:
    """Open (or create) a KùzuDB graph without wiping existing data."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        import kuzu

        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        _create_schema(self.conn)   # fresh DB: full column set
        _migrate_schema(self.conn)  # pre-existing DB: add new columns in place

    # -- writers ------------------------------------------------------------

    def add_document(self, doc_id: str, doc_hash: str = None,
                     page_count: int = None, title: str = None,
                     edition: str = None, page_label_map=None):
        """MERGE a Document node, storing citation metadata.

        page_label_map may be a dict (JSON-encoded here) or a pre-encoded
        JSON string; it is stored as a JSON string either way.
        """
        if not doc_id:
            return
        if isinstance(page_label_map, dict):
            page_label_map = json.dumps(page_label_map, ensure_ascii=False)
        props = {
            "doc_hash": doc_hash,
            "page_count": page_count,
            "title": title,
            "edition": edition,
            "page_label_map": page_label_map,
        }
        assignments = ", ".join(
            f"d.{col} = {_kuzu_literal(val)}" for col, val in props.items()
        )
        safe_doc_id = _kuzu_escape(doc_id)
        self.conn.execute(f"""
            MERGE (d:Document {{id: '{safe_doc_id}'}})
            ON CREATE SET {assignments}
            ON MATCH SET {assignments}
        """)

    def add_chunk(self, doc_id: str, chunk_id: str, page_number: int = 0,
                  section_title: str = "", text: str = "",
                  text_offset_start: int = None, text_offset_end: int = None,
                  block_x: float = None, block_y: float = None,
                  block_w: float = None, block_h: float = None) -> str:
        """MERGE a Chunk node (id = '<doc_id>_<chunk_id>', matching
        ingest_to_kuzu) with offsets/bbox and link it to its Document."""
        if not doc_id or not chunk_id:
            return ""
        cid = f"{doc_id}_{chunk_id}"
        props = {
            "chunk_id": chunk_id,
            "page_number": int(page_number or 0),
            "section_title": section_title,
            "text_passage": text,
            "text_offset_start": text_offset_start,
            "text_offset_end": text_offset_end,
            "block_x": block_x,
            "block_y": block_y,
            "block_w": block_w,
            "block_h": block_h,
        }
        assignments = ", ".join(
            f"c.{col} = {_kuzu_literal(val)}" for col, val in props.items()
        )
        safe_cid = _kuzu_escape(cid)
        safe_doc_id = _kuzu_escape(doc_id)
        self.conn.execute(f"""
            MERGE (c:Chunk {{id: '{safe_cid}'}})
            ON CREATE SET {assignments}
            ON MATCH SET {assignments}
        """)
        self.conn.execute(f"MERGE (d:Document {{id: '{safe_doc_id}'}})")
        self.conn.execute(f"""
            MATCH (d:Document {{id: '{safe_doc_id}'}}),
                  (c:Chunk {{id: '{safe_cid}'}})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """)
        return cid

    def add_concept(self, name: str, concept_type: str = "definition",
                    difficulty: str = "intermediate", summary: str = "") -> str:
        """MERGE a Concept node (helper so evidence links can be built)."""
        cid = create_concept_id(name)
        safe_name = _kuzu_escape(name)
        self.conn.execute(f"""
            MERGE (c:Concept {{id: '{cid}'}})
            ON CREATE SET c.name = '{safe_name}',
                          c.concept_type = '{_kuzu_escape(concept_type)}',
                          c.difficulty = '{_kuzu_escape(difficulty)}',
                          c.summary = '{_kuzu_escape((summary or "")[:500])}'
        """)
        return cid

    def link_mention(self, doc_id: str, chunk_id: str, concept_name: str):
        """MERGE a (Chunk)-[:MENTIONS]->(Concept) edge."""
        safe_cid = _kuzu_escape(f"{doc_id}_{chunk_id}")
        concept_id = create_concept_id(concept_name)
        self.conn.execute(f"""
            MATCH (c:Chunk {{id: '{safe_cid}'}}),
                  (con:Concept {{id: '{concept_id}'}})
            MERGE (c)-[:MENTIONS]->(con)
        """)

    # -- evidence getters (shared contract with the inference server) --------

    def get_evidence_for_concept(self, concept_name: str) -> list:
        """All evidence chunks mentioning a concept (matched by id or name).

        Each dict has exactly: chunk_id, doc_id, page_number, section_title,
        text, text_offset_start, text_offset_end, block_bbox, doc_title,
        page_label_map.
        """
        concept_id = create_concept_id(concept_name or "")
        safe_name = _kuzu_escape((concept_name or "").lower())
        evidence = []
        try:
            res = self.conn.execute(f"""
                MATCH (doc:Document)-[:HAS_CHUNK]->(chk:Chunk)
                      -[:MENTIONS]->(c:Concept)
                WHERE c.id = '{concept_id}' OR lower(c.name) = '{safe_name}'
                {_EVIDENCE_RETURN}
                ORDER BY doc.id, chk.page_number
            """)
            while res.has_next():
                evidence.append(_row_to_evidence(res.get_next()))
        except Exception:
            pass
        return evidence

    def get_evidence_for_edge(self, source: str, target: str) -> dict:
        """Evidence for a REQUIRES prerequisite edge source -> target.

        Resolution order:
          1. the chunk recorded in the edge's provenance ('doc_id:chunk_id'),
          2. any chunk mentioning both concepts,
          3. any chunk mentioning the source concept.
        Returns the evidence dict (plus 'source'/'target' keys) or None.
        """
        source_id = create_concept_id(source or "")
        target_id = create_concept_id(target or "")

        evidence = None

        # 1. Edge provenance: REQUIRES stores 'doc_id:chunk_id' in r.source.
        prov = ""
        for a, b in ((source_id, target_id), (target_id, source_id)):
            try:
                res = self.conn.execute(f"""
                    MATCH (a:Concept {{id: '{a}'}})
                          -[r:REQUIRES]->(b:Concept {{id: '{b}'}})
                    RETURN r.source
                """)
                if res.has_next():
                    prov = str(res.get_next()[0] or "")
                    break
            except Exception:
                pass
        if prov and ":" in prov:
            doc_id, chunk_id = prov.rsplit(":", 1)
            if doc_id and chunk_id:
                safe_cid = _kuzu_escape(f"{doc_id}_{chunk_id}")
                try:
                    res = self.conn.execute(f"""
                        MATCH (doc:Document)-[:HAS_CHUNK]->
                              (chk:Chunk {{id: '{safe_cid}'}})
                        {_EVIDENCE_RETURN}
                    """)
                    if res.has_next():
                        evidence = _row_to_evidence(res.get_next())
                except Exception:
                    pass

        # 2. Fall back to a chunk mentioning BOTH concepts.
        if evidence is None:
            try:
                res = self.conn.execute(f"""
                    MATCH (doc:Document)-[:HAS_CHUNK]->(chk:Chunk),
                          (chk)-[:MENTIONS]->(a:Concept {{id: '{source_id}'}}),
                          (chk)-[:MENTIONS]->(b:Concept {{id: '{target_id}'}})
                    {_EVIDENCE_RETURN}
                    LIMIT 1
                """)
                if res.has_next():
                    evidence = _row_to_evidence(res.get_next())
            except Exception:
                pass

        # 3. Last resort: any chunk mentioning the source concept.
        if evidence is None:
            for_source = self.get_evidence_for_concept(source)
            if for_source:
                evidence = for_source[0]

        if evidence is None:
            return None
        evidence["source"] = source
        evidence["target"] = target
        return evidence

    def close(self):
        self.conn = None
        self.db = None

def set_default_connection(conn):
    """Set a default connection for module-level functions.
    
    This allows tests to inject a specific database connection.
    """
    global _DEFAULT_CONN
    _DEFAULT_CONN = conn

def _get_default_graph_db():
    """Get a default GraphDB instance (creates one if needed)."""
    global _DEFAULT_GRAPH_DB
    if _DEFAULT_GRAPH_DB is None:
        _DEFAULT_GRAPH_DB = GraphDB()
    return _DEFAULT_GRAPH_DB

def _get_conn(conn):
    """Get a kuzu Connection - either the provided one, a test default, or a new default."""
    if conn is not None and hasattr(conn, 'execute'):
        return conn
    # Use test-injected default if available
    if _DEFAULT_CONN is not None:
        return _DEFAULT_CONN
    db = _get_default_graph_db()
    return db.conn

def _is_connection(obj):
    """Check if object is a kuzu Connection (has execute method)."""
    return obj is not None and hasattr(obj, 'execute')

def add_document(conn=None, doc_id: str = None, doc_hash: str = None,
                 page_count: int = None, title: str = None,
                 edition: str = None, page_label_map=None):
    """Module-level wrapper for GraphDB.add_document.
    
    Can be called as:
      add_document(conn, doc_id, doc_hash, page_count, title, edition, page_label_map)
      add_document(doc_id, doc_hash, page_count, title, edition, page_label_map)  # uses default DB
    """
    actual_conn = _get_conn(conn)
    # If first arg was actually doc_id (no conn passed), shift args
    if conn is not None and not _is_connection(conn):
        doc_id = conn
    if isinstance(page_label_map, dict):
        page_label_map = json.dumps(page_label_map, ensure_ascii=False)
    props = {
        "doc_hash": doc_hash,
        "page_count": page_count,
        "title": title,
        "edition": edition,
        "page_label_map": page_label_map,
    }
    assignments = ", ".join(
        f"d.{col} = {_kuzu_literal(val)}" for col, val in props.items()
    )
    safe_doc_id = _kuzu_escape(doc_id)
    actual_conn.execute(f"""
        MERGE (d:Document {{id: '{safe_doc_id}'}})
        ON CREATE SET {assignments}
        ON MATCH SET {assignments}
    """)

def add_chunk(conn=None, doc_id: str = None, chunk_id: str = None, page_number: int = 0,
              section_title: str = "", text: str = "",
              text_passage: str = None,
              text_offset_start: int = None, text_offset_end: int = None,
              block_x: float = None, block_y: float = None,
              block_w: float = None, block_h: float = None) -> str:
    """Module-level wrapper for GraphDB.add_chunk.
    
    Can be called as:
      add_chunk(conn, doc_id, chunk_id, page_number, section_title, text, ...)
      add_chunk(doc_id, chunk_id, page_number, section_title, text, ...)  # uses default DB
    """
    actual_conn = _get_conn(conn)
    if conn is not None and not _is_connection(conn):
        # First arg was actually doc_id
        doc_id = conn
    if not doc_id or not chunk_id:
        return ""
    cid = f"{doc_id}_{chunk_id}"
    actual_text = text_passage if text_passage is not None else text
    props = {
        "chunk_id": chunk_id,
        "page_number": int(page_number or 0),
        "section_title": section_title,
        "text_passage": actual_text,
        "text_offset_start": text_offset_start,
        "text_offset_end": text_offset_end,
        "block_x": block_x,
        "block_y": block_y,
        "block_w": block_w,
        "block_h": block_h,
    }
    assignments = ", ".join(
        f"c.{col} = {_kuzu_literal(val)}" for col, val in props.items()
    )
    safe_cid = _kuzu_escape(cid)
    safe_doc_id = _kuzu_escape(doc_id)
    actual_conn.execute(f"""
        MERGE (c:Chunk {{id: '{safe_cid}'}})
        ON CREATE SET {assignments}
        ON MATCH SET {assignments}
    """)
    actual_conn.execute(f"MERGE (d:Document {{id: '{safe_doc_id}'}})")
    actual_conn.execute(f"""
        MATCH (d:Document {{id: '{safe_doc_id}'}}),
              (c:Chunk {{id: '{safe_cid}'}})
        MERGE (d)-[:HAS_CHUNK]->(c)
    """)
    return cid

def get_evidence_for_concept(conn=None, concept_name: str = None) -> list:
    """Module-level wrapper for GraphDB.get_evidence_for_concept.
    
    Can be called as:
      get_evidence_for_concept(conn, concept_name)
      get_evidence_for_concept(concept_name)  # uses default DB
    """
    actual_conn = _get_conn(conn)
    if conn is not None and not _is_connection(conn):
        concept_name = conn
    concept_id = create_concept_id(concept_name or "")
    safe_name = _kuzu_escape((concept_name or "").lower())
    evidence = []
    try:
        res = actual_conn.execute(f"""
            MATCH (doc:Document)-[:HAS_CHUNK]->(chk:Chunk)
                  -[:MENTIONS]->(c:Concept)
            WHERE c.id = '{concept_id}' OR lower(c.name) = '{safe_name}'
            {_EVIDENCE_RETURN}
            ORDER BY doc.id, chk.page_number
        """)
        while res.has_next():
            evidence.append(_row_to_evidence(res.get_next()))
    except Exception:
        pass
    return evidence

def get_evidence_for_edge(conn=None, source: str = None, target: str = None):
    """Module-level wrapper for GraphDB.get_evidence_for_edge.
    
    Can be called as:
      get_evidence_for_edge(conn, source, target)
      get_evidence_for_edge(source, target)  # uses default DB
    """
    if conn is not None and not _is_connection(conn):
        target = source
        source = conn
        conn = None
    actual_conn = _get_conn(conn)
    
    source_id = create_concept_id(source or "")
    target_id = create_concept_id(target or "")

    evidence = None

    # 1. Edge provenance: REQUIRES stores 'doc_id:chunk_id' in r.source.
    prov = ""
    for a, b in ((source_id, target_id), (target_id, source_id)):
        try:
            res = actual_conn.execute(f"""
                MATCH (a:Concept {{id: '{a}'}})
                      -[r:REQUIRES]->(b:Concept {{id: '{b}'}})
                RETURN r.source
            """)
            if res.has_next():
                prov = str(res.get_next()[0] or "")
                break
        except Exception:
            pass
    if prov and ":" in prov:
        doc_id, chunk_id = prov.rsplit(":", 1)
        if doc_id and chunk_id:
            safe_cid = _kuzu_escape(f"{doc_id}_{chunk_id}")
            try:
                res = actual_conn.execute(f"""
                    MATCH (doc:Document)-[:HAS_CHUNK]->
                          (chk:Chunk {{id: '{safe_cid}'}})
                    {_EVIDENCE_RETURN}
                """)
                if res.has_next():
                    evidence = _row_to_evidence(res.get_next())
            except Exception:
                pass

    # 2. Fall back to a chunk mentioning BOTH concepts.
    if evidence is None:
        try:
            res = actual_conn.execute(f"""
                MATCH (doc:Document)-[:HAS_CHUNK]->(chk:Chunk),
                      (chk)-[:MENTIONS]->(a:Concept {{id: '{source_id}'}}),
                      (chk)-[:MENTIONS]->(b:Concept {{id: '{target_id}'}})
                {_EVIDENCE_RETURN}
                LIMIT 1
            """)
            if res.has_next():
                evidence = _row_to_evidence(res.get_next())
        except Exception:
            pass

    # 3. Last resort: any chunk mentioning the source concept.
    if evidence is None:
        for_source = get_evidence_for_concept(actual_conn, source)
        if for_source:
            evidence = for_source[0]

    if evidence is None:
        return None
    evidence["source"] = source
    evidence["target"] = target
    return evidence

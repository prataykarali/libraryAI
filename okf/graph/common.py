"""KùzuDB graph ingestion with MERGE semantics."""

import json
import logging
import os

from okf.config import BASE_DIR, infer_source_category
from okf.exports import build_graph_rag_index, build_visual_graph
from okf.util import _source_record, create_concept_id

logger = logging.getLogger(__name__)

# Anchor the default DB to the repo root so behaviour never depends on the
# process CWD (a CWD=parent process once minted an empty schema-only DB there).
_DEFAULT_DB_PATH = str(BASE_DIR / "okf_graph.db")


def _kuzu_escape(value: str) -> str:
    """Escape a string literal for inline Kuzu Cypher queries.

    Kuzu uses backslash escaping (\\') — NOT SQL-style doubled quotes ('').
    Doubled quotes raise a parser exception, which ensure_concept/ensure_chunk
    silently swallowed, dropping any node/chunk whose text contained an
    apostrophe (root cause of phantom viz nodes missing from the concepts dict).
    """
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def _kuzu_literal(value) -> str:
    """Render a Python value as an inline Kuzu Cypher literal (None -> NULL)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return f"'{_kuzu_escape(str(value))}'"


# ---------------------------------------------------------------------------
# Schema DDL — single source of truth. Fresh databases (ingest_to_kuzu and
# GraphDB on a new path) get the full column set via these CREATEs; databases
# built before the citation-correctness columns existed are upgraded in place
# by _migrate_schema().
# ---------------------------------------------------------------------------
_SCHEMA_DDL = [
    """
    CREATE NODE TABLE Document (
        id STRING PRIMARY KEY,
        doc_hash STRING,
        page_count INT64,
        title STRING,
        edition STRING,
        page_label_map STRING
    )
    """,
    """
    CREATE NODE TABLE Chunk (
        id STRING PRIMARY KEY,
        chunk_id STRING,
        page_number INT64,
        section_title STRING,
        text_passage STRING,
        text_offset_start INT64,
        text_offset_end INT64,
        block_x DOUBLE,
        block_y DOUBLE,
        block_w DOUBLE,
        block_h DOUBLE
    )
    """,
    """
    CREATE NODE TABLE Concept (
        id STRING PRIMARY KEY,
        name STRING,
        concept_type STRING,
        difficulty STRING,
        summary STRING
    )
    """,
    """
    CREATE REL TABLE HAS_CHUNK (
        FROM Document TO Chunk
    )
    """,
    """
    CREATE REL TABLE MENTIONS (
        FROM Chunk TO Concept
    )
    """,
    """
    CREATE REL TABLE REQUIRES (
        FROM Concept TO Concept,
        relation_type STRING,
        source STRING
    )
    """,
    """
    CREATE REL TABLE UNLOCKS (
        FROM Concept TO Concept,
        relation_type STRING,
        source STRING
    )
    """,
    """
    CREATE REL TABLE RELATED (
        FROM Concept TO Concept,
        relation_type STRING,
        source STRING
    )
    """,
]

# Columns added after the original schema shipped (citation correctness).
# Kuzu 0.11.x raises a Binder exception ("... already exists in table ...")
# when ALTER TABLE ADD targets an existing property, so every ALTER is wrapped
# in try/except — running the migration repeatedly is a no-op (idempotent).
_MIGRATION_COLUMNS = [
    ("Chunk", "text_offset_start", "INT64"),
    ("Chunk", "text_offset_end", "INT64"),
    ("Chunk", "block_x", "DOUBLE"),
    ("Chunk", "block_y", "DOUBLE"),
    ("Chunk", "block_w", "DOUBLE"),
    ("Chunk", "block_h", "DOUBLE"),
    ("Document", "doc_hash", "STRING"),
    ("Document", "page_count", "INT64"),
    ("Document", "title", "STRING"),
    ("Document", "edition", "STRING"),
    ("Document", "page_label_map", "STRING"),
]


def _create_schema(conn):
    """CREATE all node/rel tables, ignoring 'already exists' errors."""
    for ddl in _SCHEMA_DDL:
        try:
            conn.execute(ddl)
        except Exception:
            pass


def _migrate_schema(conn):
    """ALTER pre-existing tables to add the citation-correctness columns."""
    for table, col, col_type in _MIGRATION_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD {col} {col_type}")
        except Exception:
            pass



__all__ = [
    "BASE_DIR",
    "_DEFAULT_DB_PATH",
    "_MIGRATION_COLUMNS",
    "_SCHEMA_DDL",
    "_create_schema",
    "_kuzu_escape",
    "_kuzu_literal",
    "_migrate_schema",
    "logger",
    "create_concept_id",
    "infer_source_category",
    "_source_record",
    "build_graph_rag_index",
    "build_visual_graph",
]

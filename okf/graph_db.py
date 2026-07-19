"""Shim: okf.graph_db → okf.graph.* (feature split, ≤500 LOC modules)."""
from okf.graph.common import (
    _DEFAULT_DB_PATH,
    _kuzu_escape,
    _kuzu_literal,
    _SCHEMA_DDL,
    _create_schema,
    _migrate_schema,
)
from okf.graph.ingest import ingest_to_kuzu
from okf.graph.export import export_graph, enforce_dag, get_orphan_ratio, get_edge_provenance
from okf.graph.evidence import (
    GraphDB,
    set_default_connection,
    add_document,
    add_chunk,
    get_evidence_for_concept,
    get_evidence_for_edge,
    _row_to_evidence,
    _get_default_graph_db,
    _get_conn,
    _is_connection,
    _DEFAULT_CONN,
    _DEFAULT_GRAPH_DB,
)

# Re-bind module-level names so tests can read/write okf.graph_db._DEFAULT_CONN
# and stay in sync with okf.graph.evidence (single source of truth via setattr).
import okf.graph.evidence as _evidence


def __getattr__(name):
    if name in ("_DEFAULT_CONN", "_DEFAULT_GRAPH_DB"):
        return getattr(_evidence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __setattr__(name, value):
    if name in ("_DEFAULT_CONN", "_DEFAULT_GRAPH_DB"):
        setattr(_evidence, name, value)
        return
    globals()[name] = value


__all__ = [
    "ingest_to_kuzu", "export_graph", "enforce_dag", "get_orphan_ratio",
    "get_edge_provenance", "GraphDB", "set_default_connection",
    "add_document", "add_chunk", "get_evidence_for_concept", "get_evidence_for_edge",
    "_kuzu_escape", "_kuzu_literal", "_DEFAULT_DB_PATH",
    "_DEFAULT_CONN", "_DEFAULT_GRAPH_DB",
]

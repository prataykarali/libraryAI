"""Lazy okf.graph_db access for citations."""
from archipelago.inference import state as st

def _default_graph_db():
    if st._GRAPH_DB is None and not st._GRAPH_DB_IMPORT_FAILED:
        try:
            from okf import graph_db as _okf_graph_db
            st._GRAPH_DB = _okf_graph_db
        except Exception as e:
            print(f"Warning: okf.graph_db unavailable ({e}); citation evidence will use the legacy Kuzu lookup.")
            st._GRAPH_DB_IMPORT_FAILED = True
    return st._GRAPH_DB

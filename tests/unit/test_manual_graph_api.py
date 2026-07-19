"""Tests for librarian manual ensure_concept / create_edge helpers."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_ensure_concept_and_create_edge_module_level(tmp_path):
    kuzu = pytest.importorskip("kuzu")
    from okf.graph.common import _SCHEMA_DDL
    from okf.graph.ingest import ensure_concept, create_edge

    db = kuzu.Database(str(tmp_path / "manual.db"))
    conn = kuzu.Connection(db)
    for ddl in _SCHEMA_DDL:
        try:
            conn.execute(ddl)
        except Exception:
            pass

    ensure_concept(conn, "agentic_system", "Agentic System", "architecture", "advanced", "summary")
    ensure_concept(conn, "tool_use", "Tool Use", "technique", "advanced", "tools")
    assert create_edge(conn, "agentic_system", "tool_use", "requires") is True

    res = conn.execute(
        "MATCH (a:Concept {id:'agentic_system'})-[:REQUIRES]->(b:Concept {id:'tool_use'}) RETURN count(*)"
    )
    assert res.get_next()[0] == 1


def test_create_edge_unknown_relation_raises(tmp_path):
    kuzu = pytest.importorskip("kuzu")
    from okf.graph.common import _SCHEMA_DDL
    from okf.graph.ingest import ensure_concept, create_edge

    db = kuzu.Database(str(tmp_path / "manual2.db"))
    conn = kuzu.Connection(db)
    for ddl in _SCHEMA_DDL:
        try:
            conn.execute(ddl)
        except Exception:
            pass
    ensure_concept(conn, "a", "A", "definition", "intermediate", "")
    ensure_concept(conn, "b", "B", "definition", "intermediate", "")
    with pytest.raises(ValueError):
        create_edge(conn, "a", "b", "not_a_real_relation")

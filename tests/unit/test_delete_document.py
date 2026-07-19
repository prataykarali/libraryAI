"""Unit tests for document delete / unmerge helpers and directionality."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_source_belongs_to_doc():
    from okf.graph.delete_document import _source_belongs_to_doc

    assert _source_belongs_to_doc("papers/Hu2021_LoRA.pdf:chunk_036", "papers/Hu2021_LoRA.pdf")
    assert _source_belongs_to_doc("Hu2021_LoRA.pdf:chunk_1", "papers/Hu2021_LoRA.pdf")
    assert not _source_belongs_to_doc("papers/other.pdf:chunk_1", "papers/Hu2021_LoRA.pdf")


def test_filter_okf_results():
    from okf.graph.delete_document import filter_okf_results

    rows = [
        {"doc_id": "papers/A.pdf", "concept_name": "X"},
        {"doc_id": "papers/B.pdf", "concept_name": "Y"},
        {"doc_id": "papers/A.pdf", "concept_name": "Z"},
    ]
    kept = filter_okf_results(rows, "papers/A.pdf")
    assert len(kept) == 1
    assert kept[0]["concept_name"] == "Y"


def test_bibliography_and_directionality():
    from okf.cleanup_parts.directionality import (
        is_bibliography_section,
        is_inverted_prerequisite,
        prune_inverted_prerequisites,
    )

    assert is_bibliography_section("References")
    assert is_bibliography_section("Appendix A")
    assert not is_bibliography_section("3 Methods")

    assert is_inverted_prerequisite(
        "Neural Network", "Graph Neural Network", "intermediate", "advanced"
    )
    assert is_inverted_prerequisite(
        "Linear Regression", "Neural Network", "intermediate", "intermediate"
    )
    assert not is_inverted_prerequisite(
        "Deep Learning", "Neural Network", "advanced", "intermediate"
    )

    recs = [
        {
            "concept_name": "Neural Network",
            "difficulty": "intermediate",
            "prerequisites": ["Graph Neural Network", "Linear Algebra"],
        },
        {"concept_name": "Linear Algebra", "difficulty": "foundational", "prerequisites": []},
        {"concept_name": "Graph Neural Network", "difficulty": "advanced", "prerequisites": []},
    ]
    n = prune_inverted_prerequisites(recs)
    assert n >= 1
    assert recs[0]["prerequisites"] == ["Linear Algebra"]


def test_delete_document_from_temp_kuzu(tmp_path):
    """End-to-end unmerge on an isolated Kuzu DB (no live lock)."""
    kuzu = pytest.importorskip("kuzu")
    from okf.graph.common import _SCHEMA_DDL
    from okf.graph.delete_document import delete_document_from_graph, list_documents

    db_path = str(tmp_path / "test_delete.db")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    for ddl in _SCHEMA_DDL:
        try:
            conn.execute(ddl)
        except Exception:
            pass

    # Two docs; shared concept C_shared; doc-only concept C_only
    conn.execute(
        "CREATE (:Document {id: 'papers/A.pdf', doc_hash: '', page_count: 1, "
        "title: 'A', edition: '', page_label_map: ''})"
    )
    conn.execute(
        "CREATE (:Document {id: 'papers/B.pdf', doc_hash: '', page_count: 1, "
        "title: 'B', edition: '', page_label_map: ''})"
    )
    conn.execute(
        "CREATE (:Chunk {id: 'A_c0', chunk_id: 'chunk_0', page_number: 1, "
        "section_title: 'Intro', text_passage: 'shared concept', "
        "text_offset_start: 0, text_offset_end: 10, "
        "block_x: 0.0, block_y: 0.0, block_w: 0.0, block_h: 0.0})"
    )
    conn.execute(
        "CREATE (:Chunk {id: 'A_c1', chunk_id: 'chunk_1', page_number: 2, "
        "section_title: 'Only', text_passage: 'only A', "
        "text_offset_start: 0, text_offset_end: 10, "
        "block_x: 0.0, block_y: 0.0, block_w: 0.0, block_h: 0.0})"
    )
    conn.execute(
        "CREATE (:Chunk {id: 'B_c0', chunk_id: 'chunk_0', page_number: 1, "
        "section_title: 'Intro', text_passage: 'shared from B', "
        "text_offset_start: 0, text_offset_end: 10, "
        "block_x: 0.0, block_y: 0.0, block_w: 0.0, block_h: 0.0})"
    )
    conn.execute(
        "CREATE (:Concept {id: 'shared_concept', name: 'Shared Concept', "
        "concept_type: 'definition', difficulty: 'intermediate', summary: 'shared'})"
    )
    conn.execute(
        "CREATE (:Concept {id: 'only_a', name: 'Only A', "
        "concept_type: 'definition', difficulty: 'intermediate', summary: 'only'})"
    )
    conn.execute(
        "CREATE (:Concept {id: 'only_a_prereq', name: 'Only A Prereq', "
        "concept_type: 'definition', difficulty: 'foundational', summary: 'prereq'})"
    )

    conn.execute(
        "MATCH (d:Document {id: 'papers/A.pdf'}), (c:Chunk {id: 'A_c0'}) "
        "CREATE (d)-[:HAS_CHUNK]->(c)"
    )
    conn.execute(
        "MATCH (d:Document {id: 'papers/A.pdf'}), (c:Chunk {id: 'A_c1'}) "
        "CREATE (d)-[:HAS_CHUNK]->(c)"
    )
    conn.execute(
        "MATCH (d:Document {id: 'papers/B.pdf'}), (c:Chunk {id: 'B_c0'}) "
        "CREATE (d)-[:HAS_CHUNK]->(c)"
    )
    conn.execute(
        "MATCH (chk:Chunk {id: 'A_c0'}), (c:Concept {id: 'shared_concept'}) "
        "CREATE (chk)-[:MENTIONS]->(c)"
    )
    conn.execute(
        "MATCH (chk:Chunk {id: 'B_c0'}), (c:Concept {id: 'shared_concept'}) "
        "CREATE (chk)-[:MENTIONS]->(c)"
    )
    conn.execute(
        "MATCH (chk:Chunk {id: 'A_c1'}), (c:Concept {id: 'only_a'}) "
        "CREATE (chk)-[:MENTIONS]->(c)"
    )
    conn.execute(
        "MATCH (chk:Chunk {id: 'A_c1'}), (c:Concept {id: 'only_a_prereq'}) "
        "CREATE (chk)-[:MENTIONS]->(c)"
    )
    # Edge only from A
    conn.execute(
        "MATCH (a:Concept {id: 'only_a'}), (b:Concept {id: 'only_a_prereq'}) "
        "CREATE (a)-[:REQUIRES {relation_type: 'requires', source: 'papers/A.pdf:chunk_1'}]->(b)"
    )

    docs = list_documents(conn)
    assert any(d["id"] == "papers/A.pdf" for d in docs)

    stats = delete_document_from_graph(conn, "papers/A.pdf")
    assert stats["doc_id"] == "papers/A.pdf"
    assert stats["chunks_deleted"] >= 2
    assert stats["concepts_deleted"] >= 1  # only_a and only_a_prereq orphans

    # Shared concept retained via B
    res = conn.execute("MATCH (c:Concept {id: 'shared_concept'}) RETURN c.id")
    assert res.has_next()

    # Doc A gone
    res = conn.execute("MATCH (d:Document {id: 'papers/A.pdf'}) RETURN d.id")
    assert not res.has_next()

    # Doc B still there
    res = conn.execute("MATCH (d:Document {id: 'papers/B.pdf'}) RETURN d.id")
    assert res.has_next()


def test_citation_quality_prefers_non_biblio():
    from archipelago.inference.neighborhood import _citation_quality_score

    body = _citation_quality_score("3. Method", "We define LoRA as low-rank...", 4)
    biblio = _citation_quality_score("References", "Kingma et al. arxiv ... et al ... et al", 14)
    assert body > biblio

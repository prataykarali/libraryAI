"""Unit tests for okf/evaluate.py."""

import pytest
import kuzu

from okf.evaluate import compare_concepts, compare_edges, structural_audit


# ---------------------------------------------------------------------------
# compare_concepts tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compare_concepts_perfect_match():
    """extracted == gold → precision=1.0, recall=1.0, f1=1.0."""
    concepts = ["algebra", "calculus", "statistics"]
    result = compare_concepts(concepts, concepts)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["true_positives"] == 3
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


@pytest.mark.unit
def test_compare_concepts_partial_match():
    """2/3 extracted are correct → check precision, recall, f1.

    extracted: ["algebra", "calculus", "wrong_concept"]
    gold:      ["algebra", "calculus", "statistics"]

    TP=2, FP=1, FN=1
    precision = 2/(2+1) = 2/3
    recall    = 2/(2+1) = 2/3
    f1        = 2*(2/3)*(2/3) / (2/3+2/3) = 2/3
    """
    extracted = [
        {"concept_name": "Algebra"},
        {"concept_name": "Calculus"},
        {"concept_name": "Wrong Concept"},
    ]
    gold = ["algebra", "calculus", "statistics"]

    result = compare_concepts(extracted, gold)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compare_edges tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compare_edges_direction_accuracy():
    """One correctly directed edge + one reversed → direction_accuracy = 0.5.

    Gold:      A→B  (source=a, target=b, type=UNLOCKS)
               C→D  (source=c, target=d, type=UNLOCKS)

    Extracted: A→B  correct direction
               D→C  reversed direction (D→C instead of C→D)

    Both pairs exist undirected, so matched_undir has 2 items.
    direction_accuracy = 1 / 2 = 0.5
    """
    gold = [
        {"source": "a", "target": "b", "type": "UNLOCKS"},
        {"source": "c", "target": "d", "type": "UNLOCKS"},
    ]
    extracted = [
        {"source": "a", "target": "b", "type": "UNLOCKS"},   # correct
        {"source": "d", "target": "c", "type": "UNLOCKS"},   # reversed
    ]

    result = compare_edges(extracted, gold)

    assert "direction_accuracy" in result
    assert result["direction_accuracy"] == pytest.approx(0.5)
    assert "directed" in result
    assert "undirected" in result


# ---------------------------------------------------------------------------
# structural_audit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_structural_audit_clean_graph(sample_graph):
    """The sample_graph fixture has no self-loops and no cycles."""
    conn = sample_graph
    audit = structural_audit(conn)

    assert audit["self_edges"] == [], (
        f"Expected no self-edges, got: {audit['self_edges']}"
    )
    assert audit["cycles"] == [], (
        f"Expected no cycles, got: {audit['cycles']}"
    )
    assert "orphan_count" in audit
    assert "orphan_percentage" in audit


@pytest.mark.unit
def test_structural_audit_self_edge(tmp_kuzu_db):
    """A concept with a REQUIRES self-loop must be detected by structural_audit."""
    conn = tmp_kuzu_db

    # Create a concept node
    conn.execute(
        "CREATE (c:Concept {id: 'loop_concept', name: 'Loop Concept', "
        "concept_type: 'definition', difficulty: 'foundational', summary: 'A self-referential concept'})"
    )

    # Create a self-loop edge: loop_concept REQUIRES loop_concept
    conn.execute(
        "MATCH (a:Concept {id: 'loop_concept'}), (b:Concept {id: 'loop_concept'}) "
        "CREATE (a)-[:REQUIRES {relation_type: 'requires', source: 'test:chunk_001'}]->(b)"
    )

    audit = structural_audit(conn)

    self_edge_names = [se["concept"] for se in audit["self_edges"]]
    assert "Loop Concept" in self_edge_names, (
        f"Expected 'Loop Concept' in self_edges, got: {audit['self_edges']}"
    )
    assert len(audit["self_edges"]) >= 1

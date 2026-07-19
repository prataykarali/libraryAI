"""Integration tests for graph quality checks.

Covers:
- Structural audit: no cycles, no self-edges in the sample graph fixture
- Orphan ratio below 0.20 for the sample graph
- compare_concepts recall/F1 against gold_attention.json
- compare_concepts recall against gold_lora.json
"""

import json
import os

import pytest

from okf.evaluate import compare_concepts, structural_audit
from okf.graph_db import get_orphan_ratio

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_GOLD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "okf", "gold")


def _gold_path(filename: str) -> str:
    return os.path.normpath(os.path.join(_GOLD_DIR, filename))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_cycles_in_sample_graph(sample_graph):
    """The sample graph fixture must contain zero dependency cycles."""
    conn = sample_graph
    result = structural_audit(conn)
    assert len(result["cycles"]) == 0, (
        f"Expected no cycles, but found {len(result['cycles'])}: {result['cycles']}"
    )


@pytest.mark.integration
def test_no_self_edges_in_sample_graph(sample_graph):
    """Every node in the sample graph must not have an edge pointing to itself."""
    conn = sample_graph
    result = structural_audit(conn)
    assert len(result["self_edges"]) == 0, (
        f"Expected no self-edges, but found {len(result['self_edges'])}: {result['self_edges']}"
    )


@pytest.mark.integration
def test_orphan_ratio_below_threshold(sample_graph):
    """Less than 20 % of concept nodes should be orphans (no edges)."""
    conn = sample_graph
    ratio = get_orphan_ratio(conn)
    assert ratio < 0.20, (
        f"Orphan ratio {ratio:.2%} is >= 0.20. Too many disconnected concept nodes."
    )


@pytest.mark.integration
def test_compare_against_gold_attention():
    """A subset of gold-attention concepts should yield recall > 0 and f1 >= 0."""
    gold_path = _gold_path("gold_attention.json")
    with open(gold_path, "r", encoding="utf-8") as fh:
        gold_data = json.load(fh)

    gold_concepts: list[str] = gold_data["concepts"]

    # Build a small extracted list using 5 of the gold concepts
    extracted = gold_concepts[:5]  # e.g. first 5 names from the gold set

    metrics = compare_concepts(extracted, gold_concepts)

    assert metrics["recall"] > 0.0, (
        f"Expected recall > 0.0, got {metrics['recall']}. "
        "Extracted concepts were drawn from the gold set so recall must be positive."
    )
    assert metrics["f1"] >= 0.0, (
        f"F1 must be >= 0, got {metrics['f1']}."
    )


@pytest.mark.integration
def test_compare_against_gold_lora():
    """A subset of gold-lora concepts should yield recall > 0."""
    gold_path = _gold_path("gold_lora.json")
    with open(gold_path, "r", encoding="utf-8") as fh:
        gold_data = json.load(fh)

    gold_concepts: list[str] = gold_data["concepts"]

    # Build a small extracted list using 4 of the gold concepts
    extracted = gold_concepts[:4]  # e.g. first 4 names from the gold set

    metrics = compare_concepts(extracted, gold_concepts)

    assert metrics["recall"] > 0.0, (
        f"Expected recall > 0.0, got {metrics['recall']}. "
        "Extracted concepts were drawn from the gold set so recall must be positive."
    )

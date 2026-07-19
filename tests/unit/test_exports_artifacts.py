"""
tests/unit/test_write_all_artifacts.py — Unit tests for okf.exports.write_all_artifacts,
the single shared artifact writer used by finalize_and_build and the ingestion
worker's live swap (regression for root/graph_ui artifact drift).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from okf.exports import write_all_artifacts


GRAPH_EXPORT = {
    "concepts": {
        "basic_algebra": {
            "name": "Basic Algebra",
            "concept_type": "definition",
            "difficulty": "foundational",
            "summary": "Solving equations with variables.",
            "sources": [],
        },
        "linear_algebra": {
            "name": "Linear Algebra",
            "concept_type": "definition",
            "difficulty": "intermediate",
            "summary": "Vectors and matrices.",
            "sources": [],
        },
    },
    "edges": [
        {
            "from_id": "linear_algebra",
            "from_name": "Linear Algebra",
            "to_id": "basic_algebra",
            "to_name": "Basic Algebra",
            "relation": "requires",
            "edge_type": "REQUIRES",
            "source": "doc.pdf:chunk_001",
        }
    ],
    "stats": {"total_concepts": 2, "total_edges": 1},
}

OKF_RESULTS = [
    {
        "concept_name": "Basic Algebra",
        "summary": "Solving equations with variables.",
        "prerequisites": [],
        "unlocks": ["Linear Algebra"],
        "related_to": [],
        "doc_id": "doc.pdf",
        "chunk_id": "chunk_001",
    },
    {
        "concept_name": "Linear Algebra",
        "summary": "Vectors and matrices.",
        "prerequisites": ["Basic Algebra"],
        "unlocks": [],
        "related_to": [],
        "doc_id": "doc.pdf",
        "chunk_id": "chunk_002",
    },
]


@pytest.mark.unit
def test_write_all_artifacts_root_and_graph_ui_identical(tmp_path):
    """
    Bug B oracle: root okf_graph.json and graph_ui/okf_graph.json must both be
    written, with identical concept and edge counts.
    """
    (tmp_path / "graph_ui").mkdir()

    with patch("okf.exports.export_vis_json") as mock_vis:
        write_all_artifacts(GRAPH_EXPORT, OKF_RESULTS, MagicMock(),
                            base_dir=tmp_path)

    root_graph = json.loads((tmp_path / "okf_graph.json").read_text(encoding="utf-8"))
    ui_graph = json.loads(
        (tmp_path / "graph_ui" / "okf_graph.json").read_text(encoding="utf-8"))

    assert len(root_graph["concepts"]) == len(ui_graph["concepts"]) == 2
    assert len(root_graph["edges"]) == len(ui_graph["edges"]) == 1
    assert root_graph == ui_graph

    # Every other downstream artifact must be written by the same call
    assert (tmp_path / "graph_audit.json").exists()
    assert (tmp_path / "accuracy.json").exists()
    mock_vis.assert_called_once()
    assert mock_vis.call_args[0][1] == str(tmp_path / "_graph_nodes.json")
    assert mock_vis.call_args[0][2] == str(tmp_path / "_graph_edges.json")

    accuracy = json.loads((tmp_path / "accuracy.json").read_text(encoding="utf-8"))
    assert accuracy["stats"]["total_concepts_in_graph"] == 2
    assert accuracy["stats"]["total_edges_in_graph"] == 1


@pytest.mark.unit
def test_write_all_artifacts_without_graph_ui_dir(tmp_path):
    """If graph_ui/ does not exist, the writer still produces every root
    artifact and simply skips the static copy (finalize_and_build behavior)."""
    with patch("okf.exports.export_vis_json"):
        graph_audit, accuracy = write_all_artifacts(
            GRAPH_EXPORT, OKF_RESULTS, MagicMock(), base_dir=tmp_path)

    assert (tmp_path / "okf_graph.json").exists()
    assert (tmp_path / "graph_audit.json").exists()
    assert (tmp_path / "accuracy.json").exists()
    assert not (tmp_path / "graph_ui" / "okf_graph.json").exists()
    assert "overall_score" in accuracy
    assert "stats" in graph_audit

"""Unit tests for okf/unlocks_heuristics.py."""
import pytest
from okf.unlocks_heuristics import add_heuristic_unlocks

def test_add_heuristic_unlocks_promotion():
    okf_results = [
        {
            "concept_name": "Linear Algebra",
            "unlocks": [],
            "related_to": [
                {"concept": "Matrix Decomposition", "relation": "enables"}
            ]
        },
        {
            "concept_name": "Matrix Decomposition",
            "unlocks": [],
            "related_to": []
        }
    ]
    results = add_heuristic_unlocks(okf_results, [])
    
    r1 = next(r for r in results if r["concept_name"] == "Linear Algebra")
    assert "Matrix Decomposition" in r1["unlocks"]
    # Should be promoted out of related_to
    assert len(r1["related_to"]) == 0

def test_add_heuristic_unlocks_text_triggers():
    okf_results = [
        {
            "concept_name": "Linear Algebra",
            "unlocks": [],
            "related_to": []
        },
        {
            "concept_name": "Matrix Decomposition",
            "unlocks": [],
            "related_to": []
        }
    ]
    chunks = [
        {
            "doc_id": "math.pdf",
            "chunk_id": "c1",
            "text": "Linear Algebra enables Matrix Decomposition methods in machine learning."
        }
    ]
    results = add_heuristic_unlocks(okf_results, chunks)
    r1 = next(r for r in results if r["concept_name"] == "Linear Algebra")
    assert "Matrix Decomposition" in r1["unlocks"]
    assert r1["relation_provenance"]["unlock:matrix decomposition"] == "math.pdf:c1"

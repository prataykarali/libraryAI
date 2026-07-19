"""Unit tests for okf/co_mention.py."""
import pytest
from okf.co_mention import build_co_mention_edges

def test_build_co_mention_edges():
    okf_results = [
        {
            "concept_name": "Low-Rank Adaptation",
            "doc_id": "paper1.pdf",
            "prerequisites": [],
            "unlocks": [],
            "related_to": []
        },
        {
            "concept_name": "Transformer",
            "doc_id": "paper2.pdf",
            "prerequisites": [],
            "unlocks": [],
            "related_to": []
        }
    ]
    chunks = [
        {
            "doc_id": "paper1.pdf",
            "chunk_id": "c1",
            "text": "Low-Rank Adaptation (LoRA) is applied to the self-attention weights of a Transformer model."
        }
    ]
    
    results = build_co_mention_edges(okf_results, chunks)
    
    # Check that Transformer was added to Low-Rank Adaptation's related_to
    r1 = next(r for r in results if r["concept_name"] == "Low-Rank Adaptation")
    r2 = next(r for r in results if r["concept_name"] == "Transformer")
    
    related1 = [x["concept"] for x in r1["related_to"] if x["relation"] == "co_mention"]
    related2 = [x["concept"] for x in r2["related_to"] if x["relation"] == "co_mention"]
    
    assert "Transformer" in related1
    assert "Low-Rank Adaptation" in related2
    
    # Check provenance
    assert r1["relation_provenance"]["related:transformer"] == "paper1.pdf:c1"

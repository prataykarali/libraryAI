"""Unit tests for okf/alias_index.py."""
import pytest
from okf.alias_index import (
    extract_acronym,
    generate_aliases_for_name,
    build_alias_index,
    resolve_concept_name,
    names_equivalent,
)

def test_extract_acronym():
    assert extract_acronym("Low-Rank Adaptation") == "LRA"
    assert extract_acronym("Low-Rank Adaptation (LoRA)") == "LORA"
    assert extract_acronym("Retrieval-Augmented Generation") == "RAG"
    assert extract_acronym("") == ""

def test_generate_aliases_for_name():
    aliases = generate_aliases_for_name("Low-Rank Adaptation (LoRA)")
    assert "lora" in aliases
    assert "low rank adaptation" in aliases
    assert "low-rank adaptation" in aliases

def test_build_alias_index():
    concepts = ["Low-Rank Adaptation", "Retrieval-Augmented Generation"]
    index = build_alias_index(concepts)
    
    assert index["lora"] == "Low-Rank Adaptation"
    assert index["lra"] == "Low-Rank Adaptation"
    assert index["rag"] == "Retrieval-Augmented Generation"
    assert index["low-rank adaptation"] == "Low-Rank Adaptation"

def test_resolve_concept_name():
    concepts = ["Low-Rank Adaptation", "Retrieval-Augmented Generation"]
    index = build_alias_index(concepts)
    
    assert resolve_concept_name("LoRA", index) == "Low-Rank Adaptation"
    assert resolve_concept_name("LRA", index) == "Low-Rank Adaptation"
    assert resolve_concept_name("RAG", index) == "Retrieval-Augmented Generation"
    assert resolve_concept_name("Random Name", index) == "Random Name"

def test_names_equivalent():
    concepts = ["Low-Rank Adaptation", "Retrieval-Augmented Generation"]
    index = build_alias_index(concepts)
    
    assert names_equivalent("LoRA", "low_rank_adaptation", index)
    assert names_equivalent("Low-Rank Adaptation", "LRA", index)
    assert not names_equivalent("LoRA", "RAG", index)

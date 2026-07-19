import pytest
from okf.relations import validate_relation, filter_relations, infer_prerequisite_direction

@pytest.mark.unit
def test_validate_relation_both_present():
    relation = {"source": "Concept A", "target": "Concept B"}
    chunks = [
        {"text": "Concept A is closely related to Concept B in this chunk."}
    ]
    assert validate_relation(relation, chunks) is True

@pytest.mark.unit
def test_validate_relation_missing_concept():
    relation = {"source": "Concept A", "target": "Concept B"}
    # Only Concept A is present
    chunks = [
        {"text": "Concept A is a great concept, but the other one is not here."}
    ]
    assert validate_relation(relation, chunks) is False

@pytest.mark.unit
def test_filter_relations():
    relations = [
        {"source": "Concept A", "target": "Concept B"},   # A and B both in chunk 0 → KEPT
        {"source": "Concept A", "target": "Concept C"},   # C not in any chunk → DROPPED
        {"source": "Concept B", "target": "Concept C"}    # C not in any chunk → DROPPED
    ]
    chunks = [
        {"text": "Concept A is linked to Concept B in this document"},
    ]
    concepts = [
        {"concept_name": "Concept A"},
        {"concept_name": "Concept B"},
        {"concept_name": "Concept C"}
    ]
    
    filtered = filter_relations(relations, chunks, concepts)
    assert len(filtered) == 1
    assert filtered[0]["source"] == "Concept A"
    assert filtered[0]["target"] == "Concept B"

@pytest.mark.unit
def test_prerequisite_direction():
    chunks = [
        {"text": "Here we introduce Concept A first, then later in this same chunk we introduce Concept B."}
    ]
    
    # 1. Test direction by text position: Concept A appears before Concept B in the same chunk
    direction = infer_prerequisite_direction("Concept A", "Concept B", chunks)
    assert direction == ("Concept A", "Concept B")
    
    # 2. Test direction by chunk index: Concept A appears in earlier chunk
    chunks_indices = [
        {"text": "Concept A is defined here."},
        {"text": "Concept B is defined here."}
    ]
    direction = infer_prerequisite_direction("Concept B", "Concept A", chunks_indices)
    assert direction == ("Concept A", "Concept B")
    
    # 3. Fallback to difficulty
    concept_a = {"concept_name": "Concept A", "difficulty": "intermediate"}
    concept_b = {"concept_name": "Concept B", "difficulty": "foundational"}
    # Concept B is foundational, Concept A is intermediate. B should be prerequisite of A.
    direction = infer_prerequisite_direction(concept_a, concept_b, [])
    assert direction == ("Concept B", "Concept A")

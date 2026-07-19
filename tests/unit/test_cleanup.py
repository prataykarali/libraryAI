import pytest
from okf.cleanup import reject_weak_concepts, deduplicate_concepts, prune_orphans

@pytest.mark.unit
def test_reject_weak_concepts():
    chunks = [
        {"doc_id": "doc1.pdf", "chunk_id": "chunk_1", "text": "This text contains the concept Low-Rank Adaptation and some details."}
    ]
    concepts = [
        # Grounded by exact match
        {"concept_name": "Low-Rank Adaptation", "summary": "A fine-tuning method", "doc_id": "doc1.pdf", "chunk_id": "chunk_1"},
        # Weak concept - name not in text, summary overlap low
        {"concept_name": "Rocket Science", "summary": "Building rockets", "doc_id": "doc1.pdf", "chunk_id": "chunk_1"},
        # Grounded by summary overlap (rescued)
        {"concept_name": "PEFT", "summary": "This is a concept details details Low-Rank Adaptation details", "doc_id": "doc1.pdf", "chunk_id": "chunk_1"}
    ]
    
    kept = reject_weak_concepts(concepts, chunks, min_overlap=0.3)
    concept_names = [c["concept_name"] for c in kept]
    assert "Low-Rank Adaptation" in concept_names
    assert "Rocket Science" not in concept_names
    assert "PEFT" in concept_names

@pytest.mark.unit
def test_deduplicate_concepts():
    concepts = [
        {"concept_name": "Low-Rank Adaptation", "summary": "Short desc", "prerequisites": ["A"], "unlocks": ["B"], "related_to": []},
        {"concept_name": "Low Rank Adaptation", "summary": "A much longer and richer description of LRA", "prerequisites": ["C"], "unlocks": ["D"], "related_to": []}
    ]
    
    deduped = deduplicate_concepts(concepts, threshold=85)
    assert len(deduped) == 1
    item = deduped[0]
    assert item["concept_name"] in ("Low-Rank Adaptation", "Low Rank Adaptation")
    # Kept longer summary
    assert item["summary"] == "A much longer and richer description of LRA"
    # Merged prerequisites and unlocks
    assert set(item["prerequisites"]) == {"A", "C"}
    assert set(item["unlocks"]) == {"B", "D"}

@pytest.mark.unit
def test_prune_orphans_under_threshold(tmp_kuzu_db):
    conn = tmp_kuzu_db
    
    # 5 concepts, 1 connected, 4 orphans (orphan ratio = 80%)
    # Let's test with threshold=0.90 (under threshold, so kept)
    conn.execute("CREATE (c1:Concept {id: 'c1', name: 'Concept A'})")
    conn.execute("CREATE (c2:Concept {id: 'c2', name: 'Concept B'})")
    conn.execute("CREATE (c3:Concept {id: 'c3', name: 'Concept C'})")
    conn.execute("CREATE (c4:Concept {id: 'c4', name: 'Concept D'})")
    conn.execute("CREATE (c5:Concept {id: 'c5', name: 'Concept E'})")
    
    conn.execute("MATCH (a:Concept {id: 'c1'}), (b:Concept {id: 'c2'}) CREATE (a)-[:REQUIRES {relation_type: 'requires', source: 'test'}]->(b)")
    
    # orphans are c3, c4, c5 (3 orphans out of 5 concepts = 60%)
    # threshold = 80% (0.80). Since 60% < 80%, orphans are kept!
    pruned = prune_orphans(conn, threshold=0.80)
    assert len(pruned) == 0
    
    # Verify nodes still exist in database
    res = conn.execute("MATCH (c:Concept) RETURN count(c)")
    assert res.get_next()[0] == 5

@pytest.mark.unit
def test_prune_orphans_over_threshold(tmp_kuzu_db):
    conn = tmp_kuzu_db
    
    conn.execute("CREATE (c1:Concept {id: 'c1', name: 'Concept A'})")
    conn.execute("CREATE (c2:Concept {id: 'c2', name: 'Concept B'})")
    conn.execute("CREATE (c3:Concept {id: 'c3', name: 'Concept C'})")
    conn.execute("CREATE (c4:Concept {id: 'c4', name: 'Concept D'})")
    conn.execute("CREATE (c5:Concept {id: 'c5', name: 'Concept E'})")
    
    conn.execute("MATCH (a:Concept {id: 'c1'}), (b:Concept {id: 'c2'}) CREATE (a)-[:REQUIRES {relation_type: 'requires', source: 'test'}]->(b)")
    
    # 3 orphans out of 5 = 60%.
    # threshold = 20% (0.20). Since 60% >= 20%, orphans are removed!
    pruned = prune_orphans(conn, threshold=0.20)
    assert len(pruned) == 3
    pruned_ids = {p["id"] for p in pruned}
    assert pruned_ids == {"c3", "c4", "c5"}
    
    # Verify only connected nodes remain
    res = conn.execute("MATCH (c:Concept) RETURN count(c)")
    assert res.get_next()[0] == 2

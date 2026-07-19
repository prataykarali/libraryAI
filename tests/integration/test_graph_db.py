import json
import os
import re
import pytest
from unittest.mock import patch

import okf_extraction
from mock_data import MOCK_TEXT_CHUNKS

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

@pytest.fixture
def gold_okf_data():
    """Loads gold standard relationship data from okf_relationships.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "okf_relationships.json")
    with open(path, "r") as f:
        return json.load(f)

def test_extract_and_find_relationships(gold_okf_data, tmp_path):
    """
    Ported validation from legacy test_okf_graph.py.
    Extract OKF (mocked) and verify relationship mapping and topological sort.
    """
    # Mock extract_batch to return the gold_okf_data without hitting external APIs
    with patch("okf_extraction.extract_batch", return_value=gold_okf_data) as mock_extract:
        text_chunks = [chunk["text"] for chunk in MOCK_TEXT_CHUNKS]
        results = okf_extraction.extract_batch(text_chunks)
        
    assert results == gold_okf_data
    
    # Step 2: Build a concept name mapping (lowercase for matching)
    concept_names = {r.get('concept_name', '').lower(): r for r in results}
    
    relationships = []
    
    for result in results:
        concept = result.get('concept_name', '')
        unlocks = result.get('unlocks', [])
        
        # Check each unlock to see if it matches a concept in our batch
        flat_unlocks = []
        for u in unlocks:
            if isinstance(u, list):
                flat_unlocks.extend([item for item in u if isinstance(item, str)])
            elif isinstance(u, str):
                flat_unlocks.append(u)
        
        for unlock in flat_unlocks:
            unlock_lower = unlock.lower()
            for existing_concept_lower, existing_result in concept_names.items():
                if (unlock_lower in existing_concept_lower or 
                    existing_concept_lower in unlock_lower or
                    unlock_lower.split()[0] in existing_concept_lower):
                    
                    existing_concept = existing_result.get('concept_name', '')
                    if existing_concept.lower() != concept.lower():
                        relationships.append({
                            "from": concept,
                            "to": existing_concept,
                            "relation": "unlocks"
                        })
                        
    # Verify we extracted relationships successfully
    assert len(relationships) > 0
    
    # Verify specifically that "Linear Algebra" triggers unlocking relation
    lora_relations = [r for r in relationships if "Linear Algebra" in r["from"]]
    assert len(lora_relations) > 0
    
    # Step 4: Topological sort to show learning path
    learned = set()
    remaining = {r.get('concept_name', '') for r in results}
    path = []
    
    for _ in range(len(results)):
        for result in results:
            concept = result.get('concept_name', '')
            if concept in remaining and concept not in learned:
                raw_prereqs = result.get('prerequisites', [])
                prereqs = []
                for p in raw_prereqs:
                    if isinstance(p, list):
                        prereqs.extend([item.lower() for item in p if isinstance(item, str)])
                    elif isinstance(p, str):
                        prereqs.append(p.lower())
                
                # Check if prerequisites are satisfied
                prereq_satisfied = True
                for prereq in prereqs:
                    if any(prereq in cn or cn in prereq for cn in learned):
                        continue
                    elif any(prereq in cn or cn in prereq for cn in concept_names.keys() if cn not in learned):
                        prereq_satisfied = False
                        break
                
                if prereq_satisfied:
                    path.append(concept)
                    learned.add(concept)
                    remaining.discard(concept)
    
    # Add any remaining concepts
    for result in results:
        concept = result.get('concept_name', '')
        if concept not in learned:
            path.append(concept)
            
    assert len(path) == len(results)
    
    # Step 5: Save full relationship data to a temporary file
    output = {
        "concepts": [{"name": r.get('concept_name', ''), "summary": r.get('summary', '')} for r in results],
        "relationships": relationships,
        "learning_path": path,
        "full_okf_data": results
    }
    
    temp_file = tmp_path / "okf_full_relationships.json"
    with open(temp_file, "w") as f:
        json.dump(output, f, indent=2)
        
    assert temp_file.exists()


def test_analyze_relationships(gold_okf_data, tmp_path):
    """
    Ported validation from legacy test_relationships.py.
    Verifies relationship analysis and cross-reference matching on the mock data.
    """
    results = gold_okf_data
    
    # Validate structure of the extraction results
    for result in results:
        assert "concept_name" in result
        assert "summary" in result
        assert "prerequisites" in result
        assert "unlocks" in result
        
    # Cross-references check (from test_relationships.py)
    assert len(results) >= 2
    concept1_name = results[0].get('concept_name', '').lower()
    concept2_name = results[1].get('concept_name', '').lower()
    
    concept1_unlocks = [u.lower() for u in results[0].get('unlocks', [])]
    concept2_prereqs = [p.lower() for p in results[1].get('prerequisites', [])]
    
    related = any(concept2_name in unlock or concept1_name in prereq 
                 for unlock in concept1_unlocks for prereq in concept2_prereqs)
                 
    assert related is True, f"Expected '{results[0]['concept_name']}' and '{results[1]['concept_name']}' to be related"
    
    # Write mock relationships JSON to a temporary file
    temp_file = tmp_path / "mock_relationships.json"
    with open(temp_file, "w") as f:
        json.dump(results, f, indent=2)
    assert temp_file.exists()


def test_kuzu_db_integration(tmp_kuzu_db, gold_okf_data):
    """
    Verifies that concepts and relationships can be correctly loaded and queried from KuzuDB.
    Uses the tmp_kuzu_db fixture.
    """
    conn = tmp_kuzu_db
    
    def escape(s):
        return str(s).replace("\\", "\\\\").replace("'", "\\'")
        
    def get_concept_id(name):
        cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
        return re.sub(r'_+', '_', cid).strip('_') or 'concept'
        
    # 1. Populate concepts
    for concept in gold_okf_data:
        name = concept.get('concept_name', '')
        summary = concept.get('summary', '')
        concept_type = "definition"
        difficulty = "intermediate"
        cid = get_concept_id(name)
        
        conn.execute(f"""
            CREATE (c:Concept {{
                id: '{cid}',
                name: '{escape(name)}',
                concept_type: '{concept_type}',
                difficulty: '{difficulty}',
                summary: '{escape(summary)}'
            }})
        """)
        
    # Verify concepts are in the DB
    res = conn.execute("MATCH (c:Concept) RETURN count(c)")
    assert res.has_next()
    assert res.get_next()[0] == len(gold_okf_data)
    
    # Build concept map for ID resolution
    concept_map = {
        c.get('concept_name', '').lower(): get_concept_id(c.get('concept_name', ''))
        for c in gold_okf_data
    }
    
    # 2. Add REQUIRES and UNLOCKS relationships
    rel_count = 0
    for concept in gold_okf_data:
        concept_name = concept.get('concept_name', '')
        concept_id = get_concept_id(concept_name)

        prerequisites = concept.get('prerequisites', [])
        if isinstance(prerequisites, str):
            prerequisites = [prerequisites]
        for prereq in prerequisites:
            if not isinstance(prereq, str):
                continue
            prereq_lower = prereq.lower()
            matched_concept = None
            for existing_name, existing_id in concept_map.items():
                if (prereq_lower in existing_name or existing_name in prereq_lower or
                    prereq_lower.split()[0] in existing_name):
                    matched_concept = existing_id
                    break
            if matched_concept and matched_concept != concept_id:
                try:
                    conn.execute(f"""
                        MATCH (from:Concept {{id: '{concept_id}'}}),
                              (to:Concept {{id: '{matched_concept}'}})
                        CREATE (from)-[:REQUIRES {{relation_type: 'requires', source: 'extracted'}}]->(to)
                    """)
                    rel_count += 1
                except Exception:
                    pass

        unlocks = concept.get('unlocks', [])
        if isinstance(unlocks, str):
            unlocks = [unlocks]
        for unlock in unlocks:
            if not isinstance(unlock, str):
                continue
            unlock_lower = unlock.lower()
            matched_concept = None
            for existing_name, existing_id in concept_map.items():
                if (unlock_lower in existing_name or existing_name in unlock_lower or
                    unlock_lower.split()[0] in existing_name):
                    matched_concept = existing_id
                    break
            if matched_concept and matched_concept != concept_id:
                try:
                    conn.execute(f"""
                        MATCH (from:Concept {{id: '{concept_id}'}}),
                              (to:Concept {{id: '{matched_concept}'}})
                        CREATE (from)-[:UNLOCKS {{relation_type: 'unlocks', source: 'extracted'}}]->(to)
                    """)
                    rel_count += 1
                except Exception:
                    pass
                    
    assert rel_count > 0
    
    # 3. Query relationships to verify database contents
    # Check that "machine_learning" requires "linear_algebra"
    res = conn.execute("""
        MATCH (ml:Concept {id: 'machine_learning'})-[:REQUIRES]->(la:Concept {id: 'linear_algebra'})
        RETURN ml.name, la.name
    """)
    assert res.has_next()
    row = res.get_next()
    assert "Machine Learning" in row[0]
    assert "Linear Algebra" in row[1]
    
    # Check that "linear_algebra" unlocks "machine_learning"
    res = conn.execute("""
        MATCH (la:Concept {id: 'linear_algebra'})-[:UNLOCKS]->(ml:Concept {id: 'machine_learning'})
        RETURN la.name, ml.name
    """)
    assert res.has_next()

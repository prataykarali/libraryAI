import pytest
import kuzu
from okf.graph_db import enforce_dag, get_orphan_ratio, get_edge_provenance
from okf.pipeline import finalize_and_build
from unittest.mock import patch, MagicMock

def test_enforce_dag_valid_and_invalid(tmp_kuzu_db):
    conn = tmp_kuzu_db
    
    # Insert Concept nodes
    conn.execute("CREATE (c1:Concept {id: 'a', name: 'A'})")
    conn.execute("CREATE (c2:Concept {id: 'b', name: 'B'})")
    conn.execute("CREATE (c3:Concept {id: 'c', name: 'C'})")
    
    # Add a valid dependency: a -> b (meaning b requires a)
    # Edge is b -REQUIRES-> a
    conn.execute("MATCH (a:Concept {id: 'a'}), (b:Concept {id: 'b'}) CREATE (b)-[:REQUIRES {relation_type: 'requires', source: 'test'}]->(a)")
    
    # 1. Test adding a valid dependency: b -> c (c requires b)
    # enforce_dag(conn, b, c) checks if adding dependency b -> c forms cycle (path c -> b exists)
    # There is no path from c to b, so this should not raise error.
    enforce_dag(conn, 'b', 'c')
    
    # 2. Test self-loop: a -> a should raise ValueError
    with pytest.raises(ValueError, match="Self-loop detected"):
        enforce_dag(conn, 'a', 'a')
        
    # 3. Test cycle creation:
    # Existing dependency: a -> b (b requires a, i.e., b -REQUIRES-> a)
    # Proposed dependency: b -> a (a requires b, i.e., a -REQUIRES-> b)
    # Path b -> a exists (due to a -> b dependency? Wait: path b -> a exists because of dependency a -> b).
    # Let's check: enforce_dag(conn, 'b', 'a') checks if adding dependency b -> a creates cycle.
    # Since dependency a -> b already exists (b requires a), adding dependency b -> a forms cycle a -> b -> a.
    # So there is a path from a to b, which is 'b'.
    with pytest.raises(ValueError, match="creates a cycle"):
        enforce_dag(conn, 'b', 'a')


def test_get_orphan_ratio(tmp_kuzu_db):
    conn = tmp_kuzu_db
    
    # Empty DB
    assert get_orphan_ratio(conn) == 0.0
    
    # Insert concepts
    conn.execute("CREATE (c1:Concept {id: 'a', name: 'A'})")
    conn.execute("CREATE (c2:Concept {id: 'b', name: 'B'})")
    
    # Both are orphans
    assert get_orphan_ratio(conn) == 1.0
    
    # Add edge
    conn.execute("MATCH (a:Concept {id: 'a'}), (b:Concept {id: 'b'}) CREATE (a)-[:REQUIRES {relation_type: 'requires', source: 'test'}]->(b)")
    
    # No orphans left
    assert get_orphan_ratio(conn) == 0.0


def test_get_edge_provenance(sample_graph):
    conn = sample_graph
    
    # Get provenance of linear_algebra -> basic_algebra REQUIRES edge
    prov = get_edge_provenance(conn, 'linear_algebra', 'basic_algebra')
    assert prov == "math_for_ml.pdf:chunk_002"
    
    # Non-existent edge
    assert get_edge_provenance(conn, 'basic_algebra', 'deep_learning') == ""


def test_pipeline_structural_audit_abort(tmp_kuzu_db):
    conn = tmp_kuzu_db
    
    # We want to call finalize_and_build.
    # To trigger abort, we bypass enforce_dag during ingestion so the cycle is actually written to KuzuDB,
    # and then structural_audit(db) will find the cycle and raise ValueError.
    
    okf_results = [
        {
            "concept_name": "Concept A",
            "concept_type": "definition",
            "difficulty": "foundational",
            "summary": "Summary A",
            "prerequisites": ["Concept B"],
            "unlocks": []
        },
        {
            "concept_name": "Concept B",
            "concept_type": "definition",
            "difficulty": "intermediate",
            "summary": "Summary B",
            "prerequisites": ["Concept A"],
            "unlocks": []
        }
    ]
    
    from pathlib import Path
    with patch("okf.pipeline.BASE_DIR", Path(tmp_kuzu_db.db_path).parent), \
         patch("okf.pipeline.cleanup_and_canonicalize", side_effect=lambda x: x), \
         patch("okf.graph.ingest.enforce_dag", side_effect=lambda *args: None):
        with pytest.raises(ValueError, match="Structural audit failed: self-edges or cycles detected in KuzuDB"):
            finalize_and_build(okf_results, 2, 2, chunks=[])


def test_enforce_dag_relations_math():
    from archipelago.inference.quality_and_eval import enforce_dag_relations, is_math_concept
    
    assert is_math_concept("Linear Regression") is True
    assert is_math_concept("Matrix Inverse") is True
    assert is_math_concept("LoRA") is False

    concepts_dict = {
        "LoRA": {"difficulty": "advanced"},
        "Linear Regression": {"difficulty": "foundational"},
        "Other Concept": {"difficulty": "intermediate"}
    }

    # Case 1: Advanced non-math concept unlocks math concept
    # e.g., LoRA unlocks Linear Regression -> should be reversed to Linear Regression unlocks LoRA (and rel is requires)
    relations = [
        {"source": "LoRA", "target": "Linear Regression", "relation": "unlocks"}
    ]
    fixed = enforce_dag_relations(relations, concepts_dict)
    assert fixed[0]["source"] == "Linear Regression"
    assert fixed[0]["target"] == "LoRA"
    assert fixed[0]["relation"] == "requires"

    # Case 2: Math concept is unlocked by Advanced concept (in requires form)
    # e.g., Linear Regression requires LoRA -> should be reversed to LoRA requires Linear Regression
    relations2 = [
        {"source": "Linear Regression", "target": "LoRA", "relation": "requires"}
    ]
    fixed2 = enforce_dag_relations(relations2, concepts_dict)
    assert fixed2[0]["source"] == "LoRA"
    assert fixed2[0]["target"] == "Linear Regression"


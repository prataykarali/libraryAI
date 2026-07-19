import pytest
import kuzu
import json

def test_tmp_kuzu_db(tmp_kuzu_db):
    """Verify that tmp_kuzu_db initializes a valid Connection and schema."""
    assert isinstance(tmp_kuzu_db, kuzu.Connection)
    
    # Verify node tables exist
    res = tmp_kuzu_db.execute("MATCH (c:Concept) RETURN count(c)")
    assert res.has_next()
    assert res.get_next()[0] == 0

def test_sample_chunks(sample_chunks):
    """Verify sample_chunks contents and types."""
    assert isinstance(sample_chunks, list)
    assert len(sample_chunks) > 0
    for chunk in sample_chunks:
        assert "doc_id" in chunk
        assert "chunk_id" in chunk
        assert "text" in chunk

def test_sample_concepts(sample_concepts):
    """Verify sample_concepts format."""
    assert isinstance(sample_concepts, list)
    assert len(sample_concepts) == 5
    for concept in sample_concepts:
        assert "concept_name" in concept
        assert "summary" in concept
        assert "sources" in concept

def test_sample_graph(sample_graph):
    """Verify that sample_graph populates concepts and relationships correctly."""
    # Count concepts
    res = sample_graph.execute("MATCH (c:Concept) RETURN count(c)")
    assert res.get_next()[0] == 5
    
    # Count REQUIRES edges
    res = sample_graph.execute("MATCH ()-[r:REQUIRES]->() RETURN count(r)")
    assert res.get_next()[0] == 4
    
    # Verify a specific relationship path: deep_learning requires machine_learning requires linear_algebra
    res = sample_graph.execute("""
        MATCH (dl:Concept {id: 'deep_learning'})-[:REQUIRES]->(ml:Concept)-[:REQUIRES]->(la:Concept)
        RETURN dl.name, ml.name, la.name
    """)
    assert res.has_next()
    row = res.get_next()
    assert row[0] == "Deep Learning"
    assert row[1] == "Machine Learning"
    assert row[2] == "Linear Algebra"

def test_mock_ollama(mock_ollama):
    """Verify mock_ollama patches ollama.Client."""
    import ollama
    client = ollama.Client()
    
    # Test list()
    models = client.list()
    assert models == {'models': [{'name': 'qwen3.5:0.8b'}]}
    
    # Test chat()
    response = client.chat(model='some-model', messages=[])
    assert response['message']['content'] == 'Mocked Ollama response content.'

def test_flask_test_client(flask_test_client):
    """Verify that flask_test_client communicates with the app and has mocked state."""
    # Test readiness API
    response = flask_test_client.get("/api/readiness")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ready"] is True
    assert payload["graph"]["concept_count"] == 5

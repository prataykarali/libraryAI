import json
import pytest
import kuzu
import inference_server

pytestmark = pytest.mark.integration

def _stream_parts(response):
    body = response.get_data(as_text=True)
    if "\n[STREAM_START]\n" in body:
        metadata, text = body.split("\n[STREAM_START]\n", 1)
        return json.loads(metadata), text
    return {}, body

@pytest.fixture(autouse=True)
def mock_ollama_for_library(monkeypatch):
    class MockOllamaClient:
        def __init__(self, host=None):
            pass
        def list(self):
            return {'models': [{'name': 'qwen3.5:0.8b'}]}
        def chat(self, model, messages, tools=None, options=None, think=None):
            user_msg = messages[-1]["content"] if messages else ""
            user_msg_lower = user_msg.lower()
            
            # If it's about astronomy or stars, return "no" (out of scope)
            if "astronomy" in user_msg_lower or "stars" in user_msg_lower:
                return {
                    'message': {
                        'role': 'assistant',
                        'content': 'no',
                        'tool_calls': []
                    }
                }
            # Otherwise return "yes" or default mock content
            return {
                'message': {
                    'role': 'assistant',
                    'content': 'yes',
                    'tool_calls': []
                }
            }
            
    import ollama
    monkeypatch.setattr(ollama, "Client", MockOllamaClient)

@pytest.fixture
def library_setup(tmp_kuzu_db):
    """
    Set up concepts, documents, chunks, and relationships in the temporary database.
    Also updates inference_server.CONCEPTS_DATA.
    """
    conn = tmp_kuzu_db
    
    import okf.graph_db as graph_db
    old_conn = graph_db._DEFAULT_CONN
    graph_db.set_default_connection(conn)
    
    try:
        conn.execute("DROP TABLE HAS_CHUNK")
        conn.execute("DROP TABLE MENTIONS")
        conn.execute("DROP TABLE Chunk")
        conn.execute("DROP TABLE Document")
    except Exception:
        pass
        
    conn.execute("""
        CREATE NODE TABLE Document (
            id STRING PRIMARY KEY,
            doc_hash STRING,
            page_count INT64,
            title STRING,
            edition STRING,
            page_label_map STRING
        )
    """)
    conn.execute("""
        CREATE NODE TABLE Chunk (
            id STRING PRIMARY KEY,
            chunk_id STRING,
            page_number INT64,
            section_title STRING,
            text_passage STRING,
            text_offset_start INT64,
            text_offset_end INT64,
            block_x DOUBLE,
            block_y DOUBLE,
            block_w DOUBLE,
            block_h DOUBLE
        )
    """)
    conn.execute("CREATE REL TABLE HAS_CHUNK (FROM Document TO Chunk)")
    conn.execute("CREATE REL TABLE MENTIONS (FROM Chunk TO Concept)")
    
    # 1. Create Documents
    conn.execute("MERGE (d:Document {id: 'lora_paper.pdf', title: 'Low-Rank Adaptation of Large Language Models'})")
    conn.execute("MERGE (d:Document {id: 'attention_paper.pdf', title: 'Attention Is All You Need'})")
    
    # 2. Create Chunks
    conn.execute("""
        CREATE (c1:Chunk {
            id: 'lora_paper_chunk_001',
            chunk_id: 'chunk_001',
            page_number: 1,
            section_title: 'Introduction to LoRA',
            text_passage: 'LoRA adapts large models by updating only a subset of parameters.',
            text_offset_start: 0,
            text_offset_end: 65,
            block_x: 10.0,
            block_y: 10.0,
            block_w: 100.0,
            block_h: 20.0
        })
    """)
    conn.execute("""
        CREATE (c2:Chunk {
            id: 'lora_paper_chunk_002',
            chunk_id: 'chunk_002',
            page_number: 3,
            section_title: 'Methodology & Parameter Efficiency',
            text_passage: 'Fine-tuning is parameterefficient by training low-rank matrices.',
            text_offset_start: 0,
            text_offset_end: 88,
            block_x: 10.0,
            block_y: 10.0,
            block_w: 100.0,
            block_h: 20.0
        })
    """)
    
    # 3. Create Concepts
    conn.execute("""
        CREATE (c3:Concept {
            id: 'low_rank_adaptation',
            name: 'Low-Rank Adaptation',
            concept_type: 'definition',
            difficulty: 'intermediate',
            summary: 'Parameter-efficient fine-tuning technique using low-rank updates.'
        })
    """)
    conn.execute("""
        CREATE (c4:Concept {
            id: 'fine_tuning',
            name: 'Fine-Tuning',
            concept_type: 'definition',
            difficulty: 'intermediate',
            summary: 'Adapting pretrained models on downstream tasks.'
        })
    """)
    
    # 4. Link Chunk -> Document
    conn.execute("MATCH (d:Document {id: 'lora_paper.pdf'}), (c:Chunk {id: 'lora_paper_chunk_001'}) CREATE (d)-[:HAS_CHUNK]->(c)")
    conn.execute("MATCH (d:Document {id: 'lora_paper.pdf'}), (c:Chunk {id: 'lora_paper_chunk_002'}) CREATE (d)-[:HAS_CHUNK]->(c)")
    
    # 5. Link Chunk -> Concept
    conn.execute("MATCH (c:Chunk {id: 'lora_paper_chunk_001'}), (co:Concept {id: 'low_rank_adaptation'}) CREATE (c)-[:MENTIONS]->(co)")
    conn.execute("MATCH (c:Chunk {id: 'lora_paper_chunk_002'}), (co:Concept {id: 'fine_tuning'}) CREATE (c)-[:MENTIONS]->(co)")
    
    # Update inference_server CONCEPTS_DATA
    old_concepts = dict(inference_server.CONCEPTS_DATA)
    inference_server.CONCEPTS_DATA["low_rank_adaptation"] = {
        "id": "low_rank_adaptation",
        "label": "Low-Rank Adaptation",
        "name": "Low-Rank Adaptation",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Parameter-efficient fine-tuning technique using low-rank updates."
    }
    inference_server.CONCEPTS_DATA["fine_tuning"] = {
        "id": "fine_tuning",
        "label": "Fine-Tuning",
        "name": "Fine-Tuning",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Adapting pretrained models on downstream tasks."
    }
    
    yield conn
    
    inference_server.CONCEPTS_DATA = old_concepts
    graph_db._DEFAULT_CONN = old_conn

def test_astronomy_out_of_scope(flask_test_client, library_setup):
    """
    Verify that out-of-scope technical queries (e.g. astronomy/stars) return
    the canonical out-of-scope message (scope_gate.OUT_OF_SCOPE_MESSAGE).
    """
    # 1. Direct astronomy question
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "tell me about astronomy", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)
    from archipelago.inference.scope_gate import OUT_OF_SCOPE_MESSAGE
    assert OUT_OF_SCOPE_MESSAGE.split(".")[0] in text.strip()
    assert metadata["routing"]["route"] == "out_of_scope"

    # 2. Suggest books about stars (bypassing normal suggest books parser)
    response2 = flask_test_client.post(
        "/api/chat",
        json={"query": "suggest books about stars", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response2.status_code == 200
    _, text2 = _stream_parts(response2)
    assert OUT_OF_SCOPE_MESSAGE.split(".")[0] in text2.strip()

def test_suggest_books_for_topic(flask_test_client, library_setup):
    """
    Verify book-recommendation phrasings trigger library_books and return lora_paper.pdf.
    """
    for query in (
        "suggest books about fine-tuning",
        "suggest me top 5 books for the topic: fine-tuning",
    ):
        response = flask_test_client.post(
            "/api/chat",
            json={"query": query, "mode": "rag_synthesis", "synthesis": False},
        )
        assert response.status_code == 200
        metadata, text = _stream_parts(response)
        assert metadata["routing"]["route"] == "library_books", query
        assert "lora_paper.pdf" in text
        assert "Low-Rank Adaptation of Large Language Models" in text

def test_book_chapters_list(flask_test_client, library_setup):
    """
    Verify that "what are the chapters of lora_paper" triggers
    library_chapters and lists the sections with starting page numbers.
    """
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "what are the chapters of lora_paper", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)
    assert metadata["routing"]["route"] == "library_chapters"
    assert "Introduction to LoRA" in text
    assert "Methodology & Parameter Efficiency" in text
    assert "page 1" in text.lower()
    assert "page 3" in text.lower()

def test_chapter_lookup_discussing_concept(flask_test_client, library_setup):
    """
    Verify that "which chapter of lora_paper discusses Fine-Tuning" triggers
    library_chapter_lookup and pinpoints page 3.
    """
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "which chapter of lora_paper discusses Fine-Tuning", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)
    assert metadata["routing"]["route"] == "library_chapter_lookup"
    assert "Methodology & Parameter Efficiency" in text
    assert "Page 3" in text

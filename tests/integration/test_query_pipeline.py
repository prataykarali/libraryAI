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

@pytest.fixture
def lora_setup(tmp_kuzu_db):
    """
    Set up concepts, documents, chunks, and relationships for LoRA in the temporary database.
    Also updates inference_server.CONCEPTS_DATA.
    """
    conn = tmp_kuzu_db
    
    import okf.graph_db as graph_db
    old_conn = graph_db._DEFAULT_CONN
    graph_db.set_default_connection(conn)
    
    # Drop existing chunk/rel tables to recreate with full schema
    try:
        conn.execute("DROP TABLE HAS_CHUNK")
        conn.execute("DROP TABLE MENTIONS")
        conn.execute("DROP TABLE Chunk")
    except Exception:
        pass
        
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
    conn.execute("MERGE (d:Document {id: 'lora_paper.pdf'})")
    
    # 2. Create Chunks
    conn.execute("""
        CREATE (c1:Chunk {
            id: 'lora_paper.pdf_chunk_001',
            chunk_id: 'chunk_001',
            page_number: 1,
            section_title: 'Introduction',
            text_passage: 'LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.',
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
            id: 'lora_paper.pdf_chunk_002',
            chunk_id: 'chunk_002',
            page_number: 2,
            section_title: 'Methodology',
            text_passage: 'Parameter-efficient fine-tuning adapts large models by updating only a subset of parameters.',
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
            name: 'Low-Rank Adaptation (LoRA)',
            concept_type: 'definition',
            difficulty: 'intermediate',
            summary: 'Low-Rank Adaptation adapts large models with low rank matrices.'
        })
    """)
    conn.execute("""
        CREATE (c4:Concept {
            id: 'fine_tuning',
            name: 'Fine-Tuning',
            concept_type: 'definition',
            difficulty: 'intermediate',
            summary: 'Fine-Tuning updates model weights on a downstream dataset.'
        })
    """)
    
    # 4. Link Chunk -> Document (HAS_CHUNK)
    conn.execute("MATCH (d:Document {id: 'lora_paper.pdf'}), (c:Chunk {id: 'lora_paper.pdf_chunk_001'}) CREATE (d)-[:HAS_CHUNK]->(c)")
    conn.execute("MATCH (d:Document {id: 'lora_paper.pdf'}), (c:Chunk {id: 'lora_paper.pdf_chunk_002'}) CREATE (d)-[:HAS_CHUNK]->(c)")
    
    # 5. Link Chunk -> Concept (MENTIONS)
    conn.execute("MATCH (c:Chunk {id: 'lora_paper.pdf_chunk_001'}), (co:Concept {id: 'low_rank_adaptation'}) CREATE (c)-[:MENTIONS]->(co)")
    conn.execute("MATCH (c:Chunk {id: 'lora_paper.pdf_chunk_002'}), (co:Concept {id: 'fine_tuning'}) CREATE (c)-[:MENTIONS]->(co)")
    
    # 6. Link REQUIRES relationship (Low-Rank Adaptation REQUIRES Fine-Tuning)
    conn.execute("""
        MATCH (from:Concept {id: 'low_rank_adaptation'}), (to:Concept {id: 'fine_tuning'})
        CREATE (from)-[:REQUIRES {relation_type: 'requires', source: 'lora_paper.pdf:chunk_001'}]->(to)
    """)
    
    # Update inference_server CONCEPTS_DATA
    old_concepts = dict(inference_server.CONCEPTS_DATA)
    inference_server.CONCEPTS_DATA["low_rank_adaptation"] = {
        "id": "low_rank_adaptation",
        "label": "Low-Rank Adaptation (LoRA)",
        "name": "Low-Rank Adaptation (LoRA)",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Low-Rank Adaptation adapts large models with low rank matrices."
    }
    inference_server.CONCEPTS_DATA["fine_tuning"] = {
        "id": "fine_tuning",
        "label": "Fine-Tuning",
        "name": "Fine-Tuning",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Fine-Tuning updates model weights on a downstream dataset."
    }
    
    yield conn
    
    # Restore original CONCEPTS_DATA and default conn
    inference_server.CONCEPTS_DATA = old_concepts
    graph_db._DEFAULT_CONN = old_conn

def test_weak_out_of_domain_rejected(flask_test_client):
    """
    Off-topic queries (weather, etc.) must not force a fake curriculum anchor.
    They route to general chat (no graph anchor).
    """
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "What is the weather?", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)

    routing = metadata.get("routing") or {}
    assert routing.get("route") == "general_chat" or metadata.get("anchor_concept") is None
    # Must not pretend weather is a graph concept with a learning path
    assert "Learning path:" not in (text or "")
    assert text and len(text.strip()) > 10

def test_known_question_retrieves_expected_path(flask_test_client, lora_setup):
    """
    Verify that querying "What is LoRA?" successfully matches low_rank_adaptation,
    returns the correct anchor, and lists fine_tuning in prerequisites.
    """
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "What is LoRA?", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)
    
    # Check anchor concept matched correctly
    assert metadata["anchor_concept"]["id"] == "low_rank_adaptation"
    
    # Check prerequisites includes fine_tuning
    prereqs = [p["id"] for p in metadata["prerequisites"]]
    assert "fine_tuning" in prereqs
    
    # Natural template (or synthesis) should mention LoRA / fine-tuning
    combined = f"{text} {json.dumps(metadata)}".lower()
    assert "lora" in combined or "low-rank" in combined or "low rank" in combined
    assert "fine" in combined

def test_every_citation_maps_to_kuzu_evidence(flask_test_client, lora_setup):
    """
    Parse citation IDs from response text and metadata; each must map to real
    Kuzu evidence (Document/Chunk with matching doc_id and page).
    """
    import re

    conn = lora_setup

    response = flask_test_client.post(
        "/api/chat", json={"query": "What is LoRA?", "mode": "rag_synthesis"}
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)

    citations = metadata.get("citations", [])
    assert len(citations) > 0

    # Parse S# evidence IDs from rendered response text (e.g. "[S1: ...").
    text_ids = set(re.findall(r"\[(S\d+):", text or ""))
    meta_ids = {
        c["evidence_id"]
        for c in citations
        if isinstance(c.get("evidence_id"), str) and c["evidence_id"]
    }
    assert meta_ids, "metadata citations must carry evidence_id values"
    assert text_ids, "response text must contain [S#: citation brackets"
    # Every ID cited in the body must appear in structured metadata.
    assert text_ids <= meta_ids, (
        f"text citation IDs not present in metadata: {sorted(text_ids - meta_ids)}"
    )

    # Union of IDs from text + metadata must each resolve to Kuzu evidence.
    all_ids = text_ids | meta_ids
    by_id = {c["evidence_id"]: c for c in citations if c.get("evidence_id")}
    for eid in sorted(all_ids, key=lambda s: int(s[1:])):
        citation = by_id[eid]
        doc_id = citation.get("doc_id")
        page_number = citation.get("page_number")

        assert doc_id, f"{eid} missing doc_id"
        assert page_number is not None, f"{eid} missing page_number"

        safe_doc = str(doc_id).replace("'", "\\'")
        res = conn.execute(
            f"MATCH (d:Document {{id: '{safe_doc}'}})-[:HAS_CHUNK]->(chk:Chunk) "
            f"WHERE chk.page_number = {int(page_number)} "
            f"RETURN chk.id, chk.chunk_id, chk.text_passage"
        )
        assert res.has_next(), (
            f"{eid} has no Kuzu chunk for doc_id={doc_id!r} page={page_number}"
        )
        row = res.get_next()
        assert row[0], f"{eid} returned empty chunk id"
        # If the payload includes a text span, it should appear in the stored passage.
        text_span = citation.get("text_span") or ""
        passage = row[2] or ""
        if text_span and passage:
            assert text_span[:40] in passage or passage[:40] in text_span

"""Session 2 integration: multi-hop curriculum + routing with graph edges."""

import json

import pytest

pytestmark = pytest.mark.integration


def _stream_parts(response):
    body = response.get_data(as_text=True)
    if "\n[STREAM_START]\n" in body:
        metadata, text = body.split("\n[STREAM_START]\n", 1)
        return json.loads(metadata), text
    return {}, body


@pytest.fixture
def curriculum_graph(flask_test_client, tmp_kuzu_db):
    """LoRA → Full Fine-Tuning, LoRA → Transformer; Fine-Tuning chain for multi-hop.

    Depends on flask_test_client so CONCEPTS_DATA / db overrides win during the test.
    """
    import inference_server
    import okf.graph_db as graph_db

    conn = tmp_kuzu_db
    old_conn = graph_db._DEFAULT_CONN
    graph_db.set_default_connection(conn)

    # Full chunk schema for citations
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

    concepts = [
        ("low_rank_adaptation", "Low-Rank Adaptation (LoRA)", "PEFT via low-rank matrices."),
        ("full_fine_tuning", "Full Fine-Tuning", "Update all model weights."),
        ("transformer", "Transformer", "Attention-based sequence model."),
        ("matrix_decomposition", "Matrix Decomposition", "Factor matrices into low-rank forms."),
    ]
    for cid, name, summary in concepts:
        conn.execute(f"""
            CREATE (c:Concept {{
                id: '{cid}',
                name: '{name}',
                concept_type: 'definition',
                difficulty: 'intermediate',
                summary: '{summary}'
            }})
        """)

    conn.execute("MERGE (d:Document {id: 'papers/Hu2021_LoRA.pdf'})")
    for i, (cid, page, passage) in enumerate([
        ("low_rank_adaptation", 4, "LoRA freezes weights and injects low-rank adapters."),
        ("full_fine_tuning", 2, "Full fine-tuning updates all parameters."),
        ("transformer", 1, "Transformers use multi-head attention."),
        ("matrix_decomposition", 5, "Low-rank matrix decomposition."),
    ], 1):
        chunk_id = f"papers/Hu2021_LoRA.pdf_chunk_{i:03d}"
        conn.execute(f"""
            CREATE (c:Chunk {{
                id: '{chunk_id}',
                chunk_id: 'chunk_{i:03d}',
                page_number: {page},
                section_title: 'Sec',
                text_passage: '{passage}',
                text_offset_start: 0,
                text_offset_end: 40,
                block_x: 0.0, block_y: 0.0, block_w: 1.0, block_h: 1.0
            }})
        """)
        conn.execute(
            f"MATCH (d:Document {{id: 'papers/Hu2021_LoRA.pdf'}}), (c:Chunk {{id: '{chunk_id}'}}) "
            f"CREATE (d)-[:HAS_CHUNK]->(c)"
        )
        conn.execute(
            f"MATCH (c:Chunk {{id: '{chunk_id}'}}), (co:Concept {{id: '{cid}'}}) "
            f"CREATE (c)-[:MENTIONS]->(co)"
        )

    # Multi-hop: LoRA REQUIRES full_fine_tuning; full_fine_tuning REQUIRES matrix_decomposition
    # LoRA REQUIRES transformer
    edges = [
        ("low_rank_adaptation", "full_fine_tuning"),
        ("full_fine_tuning", "matrix_decomposition"),
        ("low_rank_adaptation", "transformer"),
    ]
    for a, b in edges:
        conn.execute(f"""
            MATCH (from:Concept {{id: '{a}'}}), (to:Concept {{id: '{b}'}})
            CREATE (from)-[:REQUIRES {{relation_type: 'requires', source: 'papers/Hu2021_LoRA.pdf:chunk_001'}}]->(to)
        """)

    old_concepts = dict(inference_server.CONCEPTS_DATA)
    inference_server.CONCEPTS_DATA = {
        cid: {
            "id": cid,
            "label": name,
            "name": name,
            "summary": summary,
            "tags": ["lora"] if "lora" in cid or "Low-Rank" in name else [],
            "degree": 5 if cid == "low_rank_adaptation" else 2,
        }
        for cid, name, summary in concepts
    }
    for cid, node in inference_server.CONCEPTS_DATA.items():
        node["aliases"] = inference_server.generate_aliases(node)

    old_db = inference_server.db
    inference_server.db = conn.db
    old_emb = inference_server.use_embeddings
    inference_server.use_embeddings = False

    yield conn

    inference_server.CONCEPTS_DATA = old_concepts
    inference_server.db = old_db
    inference_server.use_embeddings = old_emb
    graph_db._DEFAULT_CONN = old_conn


def test_find_curriculum_chains_multihop(curriculum_graph):
    import inference_server as s
    paths = s.find_curriculum_chains("low_rank_adaptation", max_hops=3, max_paths=6)
    assert paths, "expected at least one curriculum path"
    # Some path should mention matrix_decomposition (2-hop) or transformer
    all_ids = set()
    for p in paths:
        for n in p.get("nodes") or []:
            all_ids.add(n.get("id"))
        for lab in p.get("labels") or []:
            all_ids.add(lab.lower())
    joined = " ".join(str(x) for x in all_ids).lower()
    assert "transformer" in joined or "matrix" in joined or "fine" in joined
    assert any(p.get("hops", 0) >= 1 for p in paths)


def test_api_chat_lora_curriculum_metadata(flask_test_client, curriculum_graph):
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "What is LoRA?", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)

    routing = metadata.get("routing") or {}
    assert routing.get("route") in ("graph_strong", "graph_soft")
    anchor = metadata.get("anchor_concept") or {}
    assert anchor.get("id") == "low_rank_adaptation", anchor

    prereq_ids = {p.get("id") for p in (metadata.get("prerequisites") or [])}
    assert "full_fine_tuning" in prereq_ids or "transformer" in prereq_ids

    paths = metadata.get("curriculum_paths") or []
    assert isinstance(paths, list)
    # Text should mention curriculum or prereqs and ideally page links
    lower = (text or "").lower()
    assert "lora" in lower or "low-rank" in lower
    citations = metadata.get("citations") or []
    if citations:
        assert any("#page=" in (c.get("url") or "") for c in citations)


def test_weather_still_general_chat(flask_test_client, curriculum_graph):
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "What is the weather?", "mode": "rag_synthesis", "synthesis": False},
    )
    assert response.status_code == 200
    metadata, text = _stream_parts(response)
    routing = metadata.get("routing") or {}
    assert routing.get("route") == "general_chat" or metadata.get("anchor_concept") is None
    assert "Learning path:" not in (text or "")

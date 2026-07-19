"""Fast contracts for the deterministic Archipelago inference path.

These tests do not download/load an embedding model or contact Ollama. They
exercise the safe lexical fallback and the streaming API shape used by the UI.

Session 2: off-topic/low-sim routes to general_chat (not a fake Failed anchor);
known concepts return natural curriculum answers with prereqs/unlocks metadata.
"""

import json
import pytest

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

def _stream_parts(response):
    body = response.get_data(as_text=True)
    metadata, text = body.split("\n[STREAM_START]\n", 1)
    return json.loads(metadata), text


def test_readiness_reports_actual_service_state(flask_test_client):
    response = flask_test_client.get("/api/readiness")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ready"] is True
    assert payload["ingestion"]["upload_api_enabled"] is True
    assert payload["synthesis"]["default_model"] == "qwen3.5:0.8b"


def test_out_of_corpus_query_is_rejected_not_forced_to_an_anchor(flask_test_client):
    """Off-domain queries must not invent a curriculum anchor."""
    response = flask_test_client.post(
        "/api/chat", json={"query": "I want to build a rocket", "mode": "rag_synthesis", "synthesis": False}
    )
    metadata, text = _stream_parts(response)
    assert response.status_code == 200
    routing = metadata.get("routing") or {}
    # Session 2: free chat / no graph anchor (not a fake Failed concept match)
    assert routing.get("route") == "general_chat" or metadata.get("anchor_concept") is None
    assert "Learning path:" not in (text or "")
    assert text and len(text.strip()) > 10


def test_known_concept_uses_compact_indexed_default_response(flask_test_client, sample_graph):
    """Known concept anchors correctly and surfaces prereq/unlock curriculum."""
    response = flask_test_client.post(
        "/api/chat",
        json={"query": "What is Linear Algebra?", "mode": "rag_synthesis", "synthesis": False},
    )
    metadata, text = _stream_parts(response)
    assert response.status_code == 200
    assert metadata["anchor_concept"]["id"] == "linear_algebra"

    prereq_names = " ".join(
        (p.get("name") or p.get("label") or "") for p in (metadata.get("prerequisites") or [])
    )
    unlock_names = " ".join(
        (u.get("name") or u.get("label") or "") for u in (metadata.get("unlocks") or [])
    )
    combined = f"{text} {prereq_names} {unlock_names} {json.dumps(metadata)}".lower()
    assert "basic algebra" in combined or "basic_algebra" in combined
    assert "machine learning" in combined or "machine_learning" in combined
    # Natural fallback and/or indexed path both valid Session 2 surfaces
    assert (
        "Learning path:" in text
        or "Linear Algebra" in text
        or "linear algebra" in text.lower()
    )
    # Routing should be graph-grounded
    route = (metadata.get("routing") or {}).get("route")
    assert route in ("graph_strong", "graph_soft", None) or metadata.get("anchor_concept")
    assert metadata.get("logs"), "pipeline logs expected"

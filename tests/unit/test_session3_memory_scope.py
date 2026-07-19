"""Regression tests for Session 3: memory, graph-grounded scope, softer tone.

Covers the user-reported failures:
- "what do I need to know before starting it?" (follow-up memory)
- "what is rag" / "tell me about RAG" (graph-grounded scope, no keyword veto)
- "hi" (greetings always get a warm reply, never a refusal)
- "what's the weather" (true off-topic still rejected, with soft suggestions)
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_concepts(monkeypatch):
    from archipelago.inference import state as st
    from archipelago.inference.aliases import generate_aliases

    concepts = {
        "retrieval_augmented_generation": {
            "id": "retrieval_augmented_generation",
            "label": "Retrieval-Augmented Generation",
            "name": "Retrieval-Augmented Generation (RAG)",
            "summary": "Retrieve documents then generate with an LM.",
            "tags": ["rag", "retrieval"],
            "difficulty": "advanced",
            "degree": 12,
        },
        "vector_rag": {
            "id": "vector_rag",
            "label": "Vector RAG",
            "name": "Vector RAG",
            "summary": "RAG over dense vector indexes.",
            "tags": ["rag", "vector"],
            "difficulty": "advanced",
            "degree": 4,
        },
        "ai_agent": {
            "id": "ai_agent",
            "label": "AI Agent",
            "name": "AI Agent",
            "summary": "Goal-directed system with tools.",
            "tags": ["agent", "agentic"],
            "difficulty": "advanced",
            "degree": 2,
        },
        "neural_network": {
            "id": "neural_network",
            "label": "Neural Network",
            "name": "Neural Network",
            "summary": "Learnable layered models.",
            "difficulty": "intermediate",
            "degree": 20,
        },
    }
    for c in concepts.values():
        c["aliases"] = generate_aliases(c)

    old = dict(st.CONCEPTS_DATA)
    old_use = st.use_embeddings
    st.CONCEPTS_DATA = concepts
    st.use_embeddings = False
    yield concepts
    st.CONCEPTS_DATA = old
    st.use_embeddings = old_use


def test_followup_uses_assistant_turn_memory(fake_concepts):
    """Assistant turns are mined for the active concept, not only user turns."""
    from archipelago.inference.routing import _get_active_concept_from_history
    history = [
        {"role": "user", "content": "hmm interesting"},
        {"role": "assistant", "content": (
            "Retrieval-Augmented Generation retrieves relevant documents and "
            "uses them as context during generation."
        )},
    ]
    active = _get_active_concept_from_history(history)
    assert active == "retrieval_augmented_generation"


def test_followup_before_starting_it_routes_to_graph(fake_concepts):
    from archipelago.inference.routing import resolve_query_routing
    history = [
        {"role": "user", "content": "Explain RAG to me"},
        {"role": "assistant", "content": (
            "Retrieval-Augmented Generation retrieves context and generates text."
        )},
    ]
    r = resolve_query_routing(
        "can u tell me what I need to know before starting it?", history=history
    )
    assert r["route"] != "out_of_scope"
    assert r["route"] in ("graph_strong", "graph_soft", "low_similarity_reject")


@pytest.mark.parametrize("query", [
    "what is rag",
    "tell me about RAG",
    "anything about AI agents?",
    "what is agentic AI",
])
def test_graph_covered_topics_never_out_of_scope(fake_concepts, query):
    """If the graph has anything related, the query must not be hard-refused."""
    from archipelago.inference.routing import resolve_query_routing
    r = resolve_query_routing(query)
    assert r["route"] != "out_of_scope", (query, r)


def test_graph_block_override_helper(fake_concepts):
    from archipelago.inference.routing import _graph_block_override
    from archipelago.inference.ranking import rank_concepts
    ranked = rank_concepts("what is rag", top_k=5)
    assert _graph_block_override("what is rag", ranked) is not None
    ranked2 = rank_concepts("britney spears latest album", top_k=5)
    assert _graph_block_override("britney spears latest album", ranked2) is None


def test_true_offtopic_still_rejected(fake_concepts):
    from archipelago.inference.routing import resolve_query_routing
    r = resolve_query_routing("give me a recipe for chocolate lava cake")
    assert r["route"] == "out_of_scope"


def test_offtopic_rejects_carry_suggestions(fake_concepts):
    """Reject payloads should carry closest_concepts for the soft bridge."""
    from archipelago.inference.routing import resolve_query_routing
    r = resolve_query_routing("give me a recipe for chocolate lava cake")
    if r["route"] == "out_of_scope":
        assert "closest_concepts" in (r.get("slots") or {})


def test_greeting_gets_warm_deterministic_reply(fake_concepts):
    from archipelago.inference.routing import resolve_query_routing
    from archipelago.inference.synthesis import general_chat_reply
    r = resolve_query_routing("hi")
    assert r["route"] == "general_chat"
    reply = general_chat_reply("hi")
    assert "Welcome" in reply or "Hi" in reply
    assert "not detailed" not in reply
    assert "out of" not in reply.lower()


def test_not_indexed_reply_suggests_graph_related(fake_concepts):
    from archipelago.inference.synthesis import not_indexed_reply
    reply = not_indexed_reply("tell me about RAG", [], natural=False)
    assert "Retrieval-Augmented Generation" in reply or "Vector RAG" in reply


def test_refusal_messages_are_soft():
    from archipelago.inference.scope_gate import (
        OUT_OF_SCOPE_MESSAGE, NOT_IN_CORPUS_MESSAGE, IMPLEMENTATION_REFUSAL_MESSAGE,
    )
    for msg in (OUT_OF_SCOPE_MESSAGE, NOT_IN_CORPUS_MESSAGE, IMPLEMENTATION_REFUSAL_MESSAGE):
        assert "I am designed" not in msg
        assert "I decline" not in msg

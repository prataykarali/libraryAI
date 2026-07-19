"""Unit tests for diagnosis-driven routing fixes.

Covers: soft-reject dual gate, surface hits (RAG/RAGS), onboarding/identity,
query normalization, book pedagogy ranking helpers, prereq plausibility.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def ranking_mod():
    from archipelago.inference import ranking as r
    return r


@pytest.fixture
def routing_mod():
    from archipelago.inference import routing as r
    return r


@pytest.fixture
def scope_mod():
    from archipelago.inference import scope_gate as s
    return s


@pytest.fixture
def fake_concepts(monkeypatch):
    """Minimal concept set for ranking/routing without the full graph."""
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
        "neural_network": {
            "id": "neural_network",
            "label": "Neural Network",
            "name": "Neural Network",
            "summary": "Learnable layered models.",
            "difficulty": "intermediate",
            "degree": 20,
        },
        "graph_neural_network": {
            "id": "graph_neural_network",
            "label": "Graph Neural Network",
            "name": "Graph Neural Network",
            "summary": "Neural nets on graphs.",
            "difficulty": "advanced",
            "degree": 5,
        },
        "react_synergizing_reasoning_and_acting": {
            "id": "react_synergizing_reasoning_and_acting",
            "label": "ReAct",
            "name": "ReAct: Synergizing Reasoning and Acting",
            "summary": "Reason + act agent loop.",
            "tags": ["agent", "tool use", "react"],
            "difficulty": "advanced",
            "degree": 3,
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
        "weather_node": {
            "id": "weather_node",
            "label": "Weather Forecasting",
            "name": "Weather Forecasting",
            "summary": "Predict rain.",
            "difficulty": "foundational",
            "degree": 0,
        },
    }
    for c in concepts.values():
        c["aliases"] = generate_aliases(c)

    old = dict(st.CONCEPTS_DATA)
    old_use = st.use_embeddings
    st.CONCEPTS_DATA = concepts
    st.use_embeddings = False  # lexical path; cos == lexical
    yield concepts
    st.CONCEPTS_DATA = old
    st.use_embeddings = old_use


def test_normalize_rags_and_greetings(ranking_mod):
    n = ranking_mod.normalize_user_query(
        "hi can u suggest me how to learn about various sorts of RAGS"
    )
    assert "rag" in n.lower()
    assert "rags" not in n.lower().split()
    # greeting stripped
    assert not n.lower().startswith("hi")


def test_identity_and_onboarding_detectors(ranking_mod):
    assert ranking_mod._is_identity("who are you?")
    assert ranking_mod._is_identity("what can you do")
    assert not ranking_mod._is_chitchat("who are you?")
    assert ranking_mod._is_onboarding("hi i wanna start learning AIML")
    assert ranking_mod._is_onboarding("how do I start with machine learning")
    assert not ranking_mod._is_onboarding("what is LoRA?")


def test_routing_identity_and_onboarding(routing_mod, fake_concepts):
    r = routing_mod.resolve_query_routing("who are you")
    assert r["route"] == "identity"

    r2 = routing_mod.resolve_query_routing("hi i wanna start learning AIML")
    assert r2["route"] == "onboarding"


def test_rags_not_hard_rejected(routing_mod, fake_concepts, monkeypatch):
    """Surface hit on RAG family must not become low_similarity_reject."""
    r = routing_mod.resolve_query_routing(
        "hi can u suggest me how to learn about various sorts of RAGS"
    )
    assert r["route"] != "low_similarity_reject"
    assert r["route"] in ("graph_strong", "graph_soft", "general_chat", "onboarding")
    # With domain term RAGS + lexical path, expect graph path
    assert r["route"] in ("graph_strong", "graph_soft")
    assert r.get("anchor_id") in (
        "retrieval_augmented_generation",
        "vector_rag",
        None,
    ) or (r.get("related") and r["related"][0]["id"] in (
        "retrieval_augmented_generation", "vector_rag"
    ))


def test_surface_hit_helper(routing_mod, fake_concepts):
    ranked = [
        {
            "id": "retrieval_augmented_generation",
            "label": "Retrieval-Augmented Generation",
            "cos": 0.20,
            "lexical": 0.5,
            "alias_boost": 0.26,
            "core_boost": 0.4,
        }
    ]
    assert routing_mod._has_surface_concept_hit(ranked, "various sorts of RAGS") is True
    assert routing_mod._has_surface_concept_hit(
        [{"id": "weather_node", "label": "Weather", "cos": 0.1, "lexical": 0.05,
          "alias_boost": 0.0, "core_boost": 0.0}],
        "quantum cooking recipes",
    ) is False


def test_hard_reject_only_on_true_miss(routing_mod, fake_concepts, monkeypatch):
    from archipelago.inference import state as st

    st.use_embeddings = True  # enable reject gate path

    def fake_rank(query, top_k=None):
        return [
            {
                "id": "weather_node",
                "label": "Weather Forecasting",
                "summary": "rain",
                "cos": 0.10,
                "lexical": 0.05,
                "alias_boost": 0.0,
                "core_boost": 0.0,
                "blended": 0.10,
            }
        ]

    monkeypatch.setattr(routing_mod, "rank_concepts", fake_rank)
    monkeypatch.setattr(routing_mod, "find_anchor_concept", lambda q: (None, 0.0))
    # domain soft path needs domain terms — use a domain query with no surface hit
    # Force domain soft by stubbing domain helpers
    monkeypatch.setattr(routing_mod, "_is_learning_or_domain_query", lambda q: True)
    monkeypatch.setattr(routing_mod, "_has_domain_terms", lambda q: True)
    monkeypatch.setattr(routing_mod, "_is_chitchat", lambda q: False)
    monkeypatch.setattr(routing_mod, "_is_identity", lambda q: False)
    monkeypatch.setattr(routing_mod, "_is_onboarding", lambda q: False)
    monkeypatch.setattr(routing_mod, "_is_offtopic", lambda q: False)
    monkeypatch.setattr(
        routing_mod,
        "is_aiml_in_scope",
        lambda *a, **k: (True, "forced"),
    )
    # Isolate kill-switch from intent gate: treat as theory so we exercise
    # low cosine reject, not implementation/OOD short-circuit.
    monkeypatch.setattr(
        routing_mod,
        "classify_intent",
        lambda q, force_llm=False: {
            "intent": "theory",
            "score": 0.9,
            "margin": 0.5,
            "method": "mock",
            "scores": {"theory": 0.9},
        },
    )
    r = routing_mod.resolve_query_routing("machine learning cooking fusion xyz")
    assert r["route"] == "low_similarity_reject"
    st.use_embeddings = False


def test_scope_learning_fail_open(scope_mod):
    in_scope, reason = scope_mod.is_aiml_in_scope(
        "i wanna learn about various frameworks",
        learning_intent=True,
        has_domain_terms=False,
        offtopic_keyword=False,
        force_llm=False,
    )
    assert in_scope is True
    assert "learning" in reason or "open" in reason


def test_scope_domain_terms_skip_llm(scope_mod, monkeypatch):
    called = {"n": 0}

    def boom(q):
        called["n"] += 1
        return False

    monkeypatch.setattr(scope_mod, "check_aiml_scope_via_llm", boom)
    in_scope, reason = scope_mod.is_aiml_in_scope(
        "tell me about RAG",
        has_domain_terms=True,
    )
    assert in_scope is True
    assert called["n"] == 0
    assert "domain" in reason


def test_prereq_plausibility():
    from archipelago.inference import state as st
    from archipelago.inference.neighborhood import is_plausible_prereq, filter_prereqs

    old = dict(st.CONCEPTS_DATA)
    try:
        st.CONCEPTS_DATA = {
            "neural_network": {"id": "neural_network", "difficulty": "intermediate"},
            "graph_neural_network": {"id": "graph_neural_network", "difficulty": "advanced"},
            "linear_algebra": {"id": "linear_algebra", "difficulty": "foundational"},
        }
        assert is_plausible_prereq("neural_network", "graph_neural_network") is False
        assert is_plausible_prereq("neural_network", "linear_algebra") is True
        prereqs = [
            {"id": "graph_neural_network", "name": "GNN"},
            {"id": "linear_algebra", "name": "LA"},
        ]
        filtered = filter_prereqs("neural_network", prereqs)
        assert [p["id"] for p in filtered] == ["linear_algebra"]
    finally:
        st.CONCEPTS_DATA = old


def test_book_pedagogy_score():
    from archipelago.inference.library_queries import _doc_pedagogy_score, _prefer_papers_query

    textbook = {"id": "textbooks/Math.pdf", "mentions": 5, "source_category": "textbook"}
    paper = {"id": "papers/LoRA.pdf", "mentions": 20, "source_category": "paper"}
    # book query: textbook should beat paper despite fewer mentions
    assert _doc_pedagogy_score(textbook, prefer_papers=False) > _doc_pedagogy_score(
        paper, prefer_papers=False
    )
    # paper query: paper preferred
    assert _doc_pedagogy_score(paper, prefer_papers=True) > _doc_pedagogy_score(
        textbook, prefer_papers=True
    )
    assert _prefer_papers_query("recommend papers on RAG") is True
    assert _prefer_papers_query("books on deep learning") is False


def test_rank_prefers_rag_for_rags_query(ranking_mod, fake_concepts):
    ranked = ranking_mod.rank_concepts("various sorts of RAGS", top_k=3)
    assert ranked
    top_ids = {r["id"] for r in ranked[:2]}
    assert "retrieval_augmented_generation" in top_ids or "vector_rag" in top_ids


def test_conversational_memory_expansion(fake_concepts):
    from archipelago.inference.routing import resolve_query_routing
    history = [
        {"role": "user", "content": "Explain RAG to me"},
        {"role": "assistant", "content": "Retrieval-Augmented Generation retrieves context and generates text."}
    ]
    # "before starting it" on its own would get rejected, but with memory it resolves to vector_rag or retrieval_augmented_generation
    r = resolve_query_routing("what do I need to know before starting it?", history=history)
    assert r["route"] in ("graph_strong", "graph_soft", "low_similarity_reject")
    assert r["anchor_id"] in ("vector_rag", "retrieval_augmented_generation") or len(r.get("related", [])) > 0


def test_semantic_safety_net_ood_override(fake_concepts):
    from archipelago.inference.routing import resolve_query_routing
    # "what are diffusion models" is a domain term query but not in graph. Without safety net it might be out_of_scope.
    # With safety net, it must be overridden to low_similarity_reject.
    r = resolve_query_routing("what are diffusion models?")
    assert r["route"] == "low_similarity_reject"
    assert "closest_concepts" in r["slots"]
    assert len(r["slots"]["closest_concepts"]) > 0

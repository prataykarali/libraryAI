"""Gauntlet defenses via intent prototypes + kill-switch (no keyword banlists)."""
from __future__ import annotations

import pytest

from archipelago.inference.intent_gate import (
    classify_intent,
    intent_to_block_reason,
    clear_intent_cache,
    INTENT_THEORY,
    INTENT_IMPLEMENTATION,
    INTENT_OUT_OF_DOMAIN,
    INTENT_ENTITY_TRIVIA,
    INTENT_META,
    REASON_IMPLEMENTATION,
    REASON_OUT_OF_SCOPE,
    REASON_NOT_IN_CORPUS,
    REASON_META,
)
from archipelago.inference.routing import resolve_query_routing
from archipelago.inference.scope_gate import (
    OUT_OF_SCOPE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    IMPLEMENTATION_REFUSAL_MESSAGE,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_intent_cache()
    yield
    clear_intent_cache()


# ── Category 4: implementation intent ────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [
        "Write a bash script to download the pre-trained BERT weights.",
        "How can I deploy a RAG system using the LangChain framework?",
        "How do I integrate LoRA into my custom HuggingFace training loop?",
        "Write a Python web scraper to collect text data for a GraphRAG pipeline.",
        "Write a Python script using PyTorch to calculate a Jacobian matrix.",
        "Give me the Python code for a multi-head attention layer.",
        "How do I install the KùzuDB graph database on an Ubuntu server?",
        "Provide a Dockerfile for running a transformer model.",
        "Give me a step-by-step tutorial on setting up AWS SageMaker to train BERT.",
    ],
)
def test_implementation_intent_blocks(query):
    info = classify_intent(query)
    assert info["intent"] == INTENT_IMPLEMENTATION, info
    assert intent_to_block_reason(info["intent"], info["score"]) == REASON_IMPLEMENTATION
    r = resolve_query_routing(query)
    assert r["route"] == "out_of_scope"
    assert "implementation" in (r.get("reason") or "")


# ── Category 3: entity trivia (not keyword lists of companies) ────────

@pytest.mark.parametrize(
    "query",
    [
        "How does OpenAI's API implement LoRA in their backend?",
        "What internal infrastructure does Google use the Transformer for in Google Translate?",
        "Can you explain how to submit my model to the GLUE benchmark leaderboard?",
        "How much money did it cost to train GPT-3 compared to BERT?",
        "What is HuggingFace's corporate role in the LoRA paper?",
        "How did Microsoft's specific engineering team build GraphRAG?",
        "What AWS EC2 instances were used to train BERT?",
        "Tell me about the personal biographies of researchers Yann LeCun and Yoshua Bengio.",
    ],
)
def test_entity_trivia_intent_blocks(query):
    info = classify_intent(query)
    assert info["intent"] == INTENT_ENTITY_TRIVIA, info
    r = resolve_query_routing(query)
    assert r["route"] == "out_of_scope"
    assert "not_in_corpus" in (r.get("reason") or "")


# ── Category 5: OOD / meta / theory-with-style ────────────────────────

def test_meta_system_constraints_blocked():
    q = "As an AI language model, what are your exact system constraints and token limits?"
    info = classify_intent(q)
    assert info["intent"] == INTENT_META
    r = resolve_query_routing(q)
    assert r["route"] == "out_of_scope"
    assert "meta" in (r.get("reason") or "")


def test_emotional_ood_blocked():
    q = "I am so frustrated and crying over my homework. Why is calculus so hard?"
    info = classify_intent(q)
    assert info["intent"] == INTENT_OUT_OF_DOMAIN
    r = resolve_query_routing(q)
    assert r["route"] == "out_of_scope"


def test_movie_matrix_ood_blocked():
    q = "Summarize the plot of the movie The Matrix."
    info = classify_intent(q)
    assert info["intent"] == INTENT_OUT_OF_DOMAIN
    r = resolve_query_routing(q)
    assert r["route"] == "out_of_scope"


def test_math_matrix_stays_theory():
    info = classify_intent(
        "What is the mathematical definition of a covariance matrix?"
    )
    assert info["intent"] == INTENT_THEORY
    info2 = classify_intent("What is a matrix?")
    assert info2["intent"] == INTENT_THEORY
    info3 = classify_intent("Explain the matrix in linear algebra")
    assert info3["intent"] == INTENT_THEORY


def test_sterile_prose_strips_emoji_and_slang():
    from archipelago.inference.synthesis import enforce_sterile_prose
    dirty = "Backprop is lowkey bussin 🔥 fr fr no cap"
    clean = enforce_sterile_prose(dirty, fallback="Backpropagation is the chain rule.")
    assert "🔥" not in clean
    assert "bussin" not in clean.lower()
    assert "lowkey" not in clean.lower()


def test_persona_style_stripped_theory_path():
    q = "Explain backpropagation to me using Gen Z slang and a bunch of emojis."
    info = classify_intent(q)
    assert info["intent"] == INTENT_THEORY
    r = resolve_query_routing(q)
    # Must not hard-block as OOD; theory path (graph or low-sim reject)
    assert r["route"] != "out_of_scope" or "implementation" not in (r.get("reason") or "")
    reason = (r.get("reason") or "").lower()
    assert "implementation" not in reason
    assert "not_in_corpus" not in reason
    assert "meta" not in reason


def test_only_three_refusal_message_constants():
    assert "AI/ML" in OUT_OF_SCOPE_MESSAGE or "AIML" in OUT_OF_SCOPE_MESSAGE.upper()
    assert "not detailed" in NOT_IN_CORPUS_MESSAGE.lower()
    assert "code generation" in IMPLEMENTATION_REFUSAL_MESSAGE.lower()


@pytest.mark.parametrize(
    "query",
    [
        "What foundational math concepts must I learn before studying the Self-Attention mechanism?",
        "Define the Covariance Matrix exactly as it is described in the provided mathematical text.",
        "How does Gradient Descent conceptually connect to Fine-Tuning a large language model?",
    ],
)
def test_theory_not_blocked(query):
    info = classify_intent(query)
    assert info["intent"] == INTENT_THEORY, info
    assert intent_to_block_reason(info["intent"], info["score"]) is None
    r = resolve_query_routing(query)
    reason = (r.get("reason") or "").lower()
    assert "implementation" not in reason
    assert "not_in_corpus" not in reason
    assert "meta" not in reason

"""Unit tests for AIML scope gate and library intent detection."""
from __future__ import annotations

import pytest

from archipelago.inference.scope_gate import (
    parse_scope_answer,
    is_aiml_in_scope,
    check_aiml_scope_via_llm,
    clear_scope_cache,
    OUT_OF_SCOPE_MESSAGE,
)
from archipelago.inference.routing import _detect_library_intent, resolve_query_routing


def test_out_of_scope_message():
    assert "AI/ML" in OUT_OF_SCOPE_MESSAGE or "AIML" in OUT_OF_SCOPE_MESSAGE.upper()
    assert "out" in OUT_OF_SCOPE_MESSAGE.lower() or "outside" in OUT_OF_SCOPE_MESSAGE.lower()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True),
        ("no", False),
        ("YES\n", True),
        ("no, astronomy", False),
        ("", None),
        ("maybe", None),
    ],
)
def test_parse_scope_answer(text, expected):
    assert parse_scope_answer(text) is expected


def test_offtopic_keyword_without_domain_is_out():
    in_scope, reason = is_aiml_in_scope(
        "what is the weather?",
        offtopic_keyword=True,
        has_domain_terms=False,
    )
    assert in_scope is False
    assert "offtopic" in reason


def test_chitchat_skipped():
    in_scope, reason = is_aiml_in_scope("hi", chitchat=True)
    assert in_scope is True
    assert reason == "chitchat_skip"


def test_strong_anchor_fast_path():
    in_scope, reason = is_aiml_in_scope(
        "what is LoRA?",
        strong_anchor=True,
        has_domain_terms=True,
    )
    assert in_scope is True
    assert "fast_path" in reason or "domain" in reason


def test_llm_scope_with_mock(monkeypatch):
    clear_scope_cache()

    class MockClient:
        def __init__(self, host=None):
            pass

        def chat(self, model, messages, options=None, think=None):
            content = messages[-1]["content"].lower()
            ans = "no" if "star" in content or "astron" in content else "yes"
            return {"message": {"content": ans}}

    import ollama

    monkeypatch.setattr(ollama, "Client", MockClient)
    assert check_aiml_scope_via_llm("suggest books about stars") is False
    assert check_aiml_scope_via_llm("suggest books about fine-tuning") is True
    # cache hit
    assert check_aiml_scope_via_llm("suggest books about stars") is False


@pytest.mark.parametrize(
    "query,intent",
    [
        ("suggest me top 5 books for the topic: fine-tuning", "library_books"),
        ("suggest books about fine-tuning", "library_books"),
        ("top 5 books for computer vision", "library_books"),
        ("recommend papers on RAG", "library_books"),
        ("what are the chapters of lora_paper", "library_chapters"),
        ("show chapters of attention_paper.pdf", "library_chapters"),
        ("which chapter of lora_paper discusses Fine-Tuning", "library_chapter_lookup"),
        ("what is LoRA?", None),
        ("hi", None),
    ],
)
def test_detect_library_intent(query, intent):
    got = _detect_library_intent(query)
    if intent is None:
        assert got is None
    else:
        assert got is not None
        assert got["intent"] == intent


def test_routing_library_and_oos(monkeypatch):
    clear_scope_cache()

    class MockClient:
        def __init__(self, host=None):
            pass

        def chat(self, model, messages, options=None, think=None):
            content = messages[-1]["content"].lower()
            ans = "no" if any(x in content for x in ("star", "astron", "weather")) else "yes"
            return {"message": {"content": ans}}

    import ollama

    monkeypatch.setattr(ollama, "Client", MockClient)

    r = resolve_query_routing("suggest me top 5 books for the topic: fine-tuning")
    assert r["route"] == "library_books"
    assert r.get("slots", {}).get("limit") == 5

    r2 = resolve_query_routing("suggest books about stars")
    assert r2["route"] == "out_of_scope"

    # Learning asks the corpus can't serve get an honest not-indexed reply
    # (low_similarity_reject) — never a random graph pin, never generic chat.
    r3 = resolve_query_routing("I want to learn about stars")
    assert r3["route"] in ("out_of_scope", "low_similarity_reject")
    if r3["route"] == "low_similarity_reject":
        assert r3.get("slots", {}).get("closest_concepts") is not None

    r4 = resolve_query_routing("hi")
    assert r4["route"] == "general_chat"

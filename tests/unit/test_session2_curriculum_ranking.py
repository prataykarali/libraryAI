"""Session 2 unit tests: alias ranking + multi-hop curriculum helpers.

No GPU / Ollama required. Uses CONCEPTS_DATA injection and pure helpers.
"""

import re

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def server_module():
    import inference_server as s
    return s


def test_extract_acronym_parenthetical(server_module):
    s = server_module
    assert s.extract_acronym("Low-Rank Adaptation (LoRA)").upper() == "LORA"
    assert s.extract_acronym("Retrieval-Augmented Generation (RAG)").upper() == "RAG"


def test_generate_aliases_includes_lora(server_module):
    s = server_module
    aliases = s.generate_aliases({
        "name": "Low-Rank Adaptation (LoRA)",
        "label": "Low-Rank Adaptation",
        "tags": ["peft", "lora"],
    })
    lower = {a.lower() for a in aliases}
    assert "lora" in lower
    assert any("low-rank" in a or "low rank" in a for a in lower)


def test_pdf_page_url_and_markdown_link(server_module):
    s = server_module
    url = s.pdf_page_url("papers/Hu2021_LoRA.pdf", 4)
    assert url.endswith("/pdfs/papers/Hu2021_LoRA.pdf#page=4")
    md = s.markdown_pdf_link("LoRA p.4", "papers/Hu2021_LoRA.pdf", 4)
    assert md.startswith("[")
    assert "#page=4" in md
    assert "Hu2021_LoRA.pdf" in md


def test_core_concept_bonus_prefers_low_rank_adaptation(server_module):
    s = server_module
    core = s._core_concept_bonus(
        "What is LoRA?",
        "low_rank_adaptation",
        "Low-Rank Adaptation",
        ["lora", "low-rank adaptation"],
    )
    variant = s._core_concept_bonus(
        "What is LoRA?",
        "lora_applied_to_attention_weights",
        "LoRA Applied to Attention Weights",
        ["lora applied to attention weights"],
    )
    assert core > variant


def test_rank_concepts_alias_prefers_core_lora(server_module):
    s = server_module
    old = dict(s.CONCEPTS_DATA)
    try:
        s.CONCEPTS_DATA = {
            "low_rank_adaptation": {
                "id": "low_rank_adaptation",
                "label": "Low-Rank Adaptation",
                "name": "Low-Rank Adaptation (LoRA)",
                "summary": "Parameter-efficient fine-tuning via low-rank matrices.",
                "tags": ["lora", "peft"],
                "degree": 20,
                "aliases": s.generate_aliases({
                    "name": "Low-Rank Adaptation (LoRA)",
                    "label": "Low-Rank Adaptation",
                    "tags": ["lora"],
                }),
            },
            "lora_applied_to_attention_weights": {
                "id": "lora_applied_to_attention_weights",
                "label": "LoRA Applied to Attention Weights",
                "name": "LoRA Applied to Attention Weights",
                "summary": "Applies LoRA to attention projections.",
                "tags": ["lora", "attention"],
                "degree": 3,
                "aliases": s.generate_aliases({
                    "name": "LoRA Applied to Attention Weights",
                    "label": "LoRA Applied to Attention Weights",
                }),
            },
            "weather_node": {
                "id": "weather_node",
                "label": "Weather Forecasting",
                "name": "Weather Forecasting",
                "summary": "Predicting rain.",
                "degree": 0,
            },
        }
        # Force lexical path (no embed tensor dependency)
        old_use = s.use_embeddings
        s.use_embeddings = False
        ranked = s.rank_concepts("What is LoRA?", top_k=5)
        s.use_embeddings = old_use
        assert ranked, "rank_concepts returned empty"
        assert ranked[0]["id"] == "low_rank_adaptation", (
            f"expected core LoRA first, got {ranked[0]['id']} scores={[(r['id'], r.get('blended')) for r in ranked]}"
        )
        aid, score = s.find_anchor_concept("What is LoRA?")
        assert aid == "low_rank_adaptation"
        assert score > 0.5
    finally:
        s.CONCEPTS_DATA = old


def test_format_curriculum_paths_section_has_page_links(server_module):
    s = server_module
    paths = [{
        "labels": ["Transformer", "Low-Rank Adaptation"],
        "markdown": "[Transformer (p.3)](http://localhost:5051/pdfs/papers/x.pdf#page=3) → Low-Rank Adaptation",
        "hops": 1,
    }]
    section = s.format_curriculum_paths_section(paths)
    # 1-hop paths use a quieter label; multi-hop keeps the multi-hop title
    assert (
        "Multi-hop curriculum path" in section
        or "Learning path (from graph prerequisites)" in section
    )
    assert "#page=3" in section


def test_render_indexed_learning_path_includes_curriculum(server_module):
    s = server_module
    target = {"id": "low_rank_adaptation", "label": "Low-Rank Adaptation", "summary": "PEFT method."}
    prereqs = [{"id": "transformer", "name": "Transformer", "summary": "Attention architecture."}]
    unlocks = []
    citation_map = {
        "transformer": [{
            "evidence_id": "S1",
            "doc_id": "papers/Hu2021_LoRA.pdf",
            "page_number": 3,
            "section_title": "Method",
            "text": "Transformer layers.",
        }],
        "low_rank_adaptation": [{
            "evidence_id": "S2",
            "doc_id": "papers/Hu2021_LoRA.pdf",
            "page_number": 4,
            "section_title": "LoRA",
            "text": "Low-rank adaptation.",
        }],
    }
    paths = [{
        "labels": ["Transformer", "Low-Rank Adaptation"],
        "markdown": "Transformer → Low-Rank Adaptation",
        "hops": 1,
        "nodes": [],
    }]
    text = s.render_indexed_learning_path(target, prereqs, unlocks, citation_map, curriculum_paths=paths)
    assert "Learning path: Low-Rank Adaptation" in text
    assert (
        "Multi-hop curriculum path" in text
        or "Learning path (from graph prerequisites)" in text
    )
    assert "1. Learn first" in text
    assert "#page=" in text or "S1" in text


def test_citation_payload_url_has_page(server_module):
    s = server_module
    payload = s.citation_payload(
        {
            "evidence_id": "S1",
            "doc_id": "papers/Lewis2020_RAG.pdf",
            "page_number": 7,
            "section_title": "RAG",
            "text": "Retrieval-augmented generation.",
        },
        "RAG",
    )
    assert payload["url"].endswith("#page=7")
    assert payload["page_number"] == 7

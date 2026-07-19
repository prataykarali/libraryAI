"""Provenance cleanse pass: generator markers → deterministic citations."""
from archipelago.inference.citations import (
    cleanse_model_citations, render_citation_from_payload,
)


PAYLOADS = [
    {
        "evidence_id": "S1",
        "topic": "RAG",
        "doc_id": "papers/Lewis2020_RAG.pdf",
        "page_number": 4,
        "printed_page": None,
        "url": "http://localhost:5051/pdfs/papers/Lewis2020_RAG.pdf#page=4",
    },
    {
        "evidence_id": "S2",
        "topic": "GPT-2",
        "doc_id": "papers/Hu2021_LoRA.pdf",
        "page_number": 5,
        "printed_page": None,
        "url": "http://localhost:5051/pdfs/papers/Hu2021_LoRA.pdf#page=5",
    },
]


def test_valid_marker_expands_to_full_citation():
    text = "RAG retrieves documents to augment generation [S1]."
    out = cleanse_model_citations(text, PAYLOADS)
    assert "[S1: RAG | papers/Lewis2020_RAG.pdf, PDF page 4]" in out
    assert "#page=4" in out


def test_invented_id_is_stripped():
    text = "RAG is great [S9]. GPT-2 is a decoder model [S2]."
    out = cleanse_model_citations(text, PAYLOADS)
    assert "S9" not in out
    assert "[S2: GPT-2 | papers/Hu2021_LoRA.pdf, PDF page 5]" in out


def test_embellished_bracket_normalized_then_expanded():
    # Model wrote its own doc/page inside the bracket — must be replaced with
    # the real provenance, not trusted.
    text = "RAG retrieves documents [S1: RAG | fake_book.pdf, PDF page 99]."
    out = cleanse_model_citations(text, PAYLOADS)
    assert "fake_book" not in out
    assert "page 99" not in out
    assert "[S1: RAG | papers/Lewis2020_RAG.pdf, PDF page 4]" in out


def test_misattached_marker_dropped():
    # [S2] is GPT-2 evidence but the sentence is about GraphRAG → marker dropped.
    text = "GraphRAG builds knowledge graphs for retrieval [S2]."
    out = cleanse_model_citations(text, PAYLOADS)
    # The GPT-2 citation must not be attached inline to the GraphRAG sentence.
    assert "retrieval [S2: GPT-2" not in out
    assert out.startswith("GraphRAG builds knowledge graphs for retrieval.")
    # No inline citation survived → Sources footer guarantees provenance.
    assert "**Sources:**" in out
    assert "Lewis2020_RAG.pdf" in out


def test_no_markers_at_all_gets_sources_footer():
    out = cleanse_model_citations("RAG is a retrieval method.", PAYLOADS)
    assert "**Sources:**" in out


def test_render_citation_uses_printed_page_when_available():
    p = dict(PAYLOADS[0], printed_page="142")
    assert "p. 142" in render_citation_from_payload(p)

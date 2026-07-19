"""
tests/integration/test_staged_pipeline_regression.py — Regression test for the
P0 data-destruction bug: run_pipeline_staged must never re-filter previously
ingested records against a NEW document's chunks. Uploading a tiny unrelated
document used to wipe every prerequisite/unlock from all existing records,
because relation co-occurrence validation only saw the new doc's text.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from okf.pipeline import run_pipeline_staged


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

EXISTING_RESULTS = [
    {
        "concept_name": "Basic Algebra",
        "concept_type": "definition",
        "difficulty": "foundational",
        "summary": "Basic Algebra teaches solving equations with variables.",
        "prerequisites": [],
        "unlocks": ["Linear Algebra"],
        "related_to": [],
        "tags": [],
        "doc_id": "old_math.pdf",
        "chunk_id": "chunk_001",
        "source_passage": "Basic Algebra teaches solving equations. It unlocks Linear Algebra.",
    },
    {
        "concept_name": "Linear Algebra",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Linear Algebra extends basic algebra to vectors and matrices.",
        "prerequisites": ["Basic Algebra"],
        "unlocks": ["Machine Learning"],
        "related_to": [{"concept": "Machine Learning", "relation": "enables"}],
        "tags": [],
        "doc_id": "old_math.pdf",
        "chunk_id": "chunk_002",
        "source_passage": "Linear Algebra requires Basic Algebra and unlocks Machine Learning.",
    },
    {
        "concept_name": "Machine Learning",
        "concept_type": "definition",
        "difficulty": "advanced",
        "summary": "Machine Learning lets systems learn patterns from data.",
        "prerequisites": ["Linear Algebra"],
        "unlocks": [],
        "related_to": [],
        "tags": [],
        "doc_id": "old_ml.pdf",
        "chunk_id": "chunk_001",
        "source_passage": "Machine Learning builds on Linear Algebra to learn from data.",
    },
]

# The new upload is deliberately about something UNRELATED: none of the
# existing concept names co-occur in its text, so re-filtering the old
# records against these chunks (the bug) would delete all their relations.
NEW_DOC_CHUNK = {
    "doc_id": "synthetic_one_pager.pdf",
    "chunk_id": "chunk_001",
    "page_number": 1,
    "section_title": "Intro",
    "text": "Photosynthesis Overview converts sunlight into chemical energy in plants.",
    "chunk_kind": "prose",
}

NEW_DOC_EXTRACTION = [
    {
        "concept_name": "Photosynthesis Overview",
        "concept_type": "definition",
        "difficulty": "foundational",
        "summary": "Photosynthesis converts sunlight into chemical energy.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": [],
        "doc_id": "synthetic_one_pager.pdf",
        "chunk_id": "chunk_001",
    }
]


def _relation_counts(records: list, doc_ids: set) -> dict:
    """Count prereqs/unlocks/related across records from the given docs."""
    counts = {"prerequisites": 0, "unlocks": 0, "related_to": 0}
    for r in records:
        if r.get("doc_id") not in doc_ids:
            continue
        counts["prerequisites"] += len(r.get("prerequisites", []))
        counts["unlocks"] += len(r.get("unlocks", []))
        counts["related_to"] += len(r.get("related_to", []))
    return counts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_staged_ingest_preserves_existing_relations(tmp_path):
    """
    Oracle for Bug A: staged ingest of a small unrelated document must NOT
    reduce concept or relation counts on pre-existing records.
    """
    saved_file = tmp_path / "okf_results.json"
    saved_file.write_text(
        json.dumps(EXISTING_RESULTS, indent=2), encoding="utf-8")

    graph_export_stub = {
        "concepts": {},
        "edges": [],
        "stats": {"total_concepts": 4, "total_edges": 3},
    }

    with patch("okf.pipeline.BASE_DIR", tmp_path), \
         patch("pdf_ingestion.ingest_document", return_value=[dict(NEW_DOC_CHUNK)]), \
         patch("okf.extraction.LOCAL_MODE", False), \
         patch("okf.extraction.extract_okf_v15",
               return_value=[dict(r) for r in NEW_DOC_EXTRACTION]), \
         patch("okf.pipeline.ingest_to_kuzu",
               return_value=(MagicMock(), MagicMock(), graph_export_stub)), \
         patch("okf.evaluate.structural_audit",
               return_value={"self_edges": [], "cycles": []}):
        okf_results, db, graph_export = run_pipeline_staged(
            source_path=str(tmp_path / "synthetic_one_pager.pdf"),
            temp_db_path=str(tmp_path / "temp_graph.db"),
            on_progress=lambda stage, pct, detail: None,
        )

    old_doc_ids = {"old_math.pdf", "old_ml.pdf"}
    before = _relation_counts(EXISTING_RESULTS, old_doc_ids)
    after = _relation_counts(okf_results, old_doc_ids)

    # No pre-existing concept may be dropped by the new doc's ingest
    names = {r.get("concept_name") for r in okf_results}
    assert {"Basic Algebra", "Linear Algebra", "Machine Learning"} <= names

    # No pre-existing relation may be deleted (the P0 bug zeroed these out)
    for field in ("prerequisites", "unlocks", "related_to"):
        assert after[field] >= before[field], (
            f"{field} on pre-existing records shrank from "
            f"{before[field]} to {after[field]} after staged ingest"
        )

    # The new doc's concept must still land in the merged corpus
    assert "Photosynthesis Overview" in names


def test_staged_ingest_still_filters_new_doc_relations(tmp_path):
    """
    Scoping the filters to the new doc must not disable them: a hallucinated
    relation on the NEW record whose endpoints never co-occur in the new
    doc's chunks is still removed.
    """
    saved_file = tmp_path / "okf_results.json"
    saved_file.write_text(
        json.dumps(EXISTING_RESULTS, indent=2), encoding="utf-8")

    hallucinated = [dict(NEW_DOC_EXTRACTION[0])]
    # "Machine Learning" exists in the corpus inventory but never co-occurs
    # with the new concept in the new document's text.
    hallucinated[0]["prerequisites"] = ["Machine Learning"]

    graph_export_stub = {
        "concepts": {},
        "edges": [],
        "stats": {"total_concepts": 4, "total_edges": 3},
    }

    with patch("okf.pipeline.BASE_DIR", tmp_path), \
         patch("pdf_ingestion.ingest_document", return_value=[dict(NEW_DOC_CHUNK)]), \
         patch("okf.extraction.LOCAL_MODE", False), \
         patch("okf.extraction.extract_okf_v15", return_value=hallucinated), \
         patch("okf.pipeline.ingest_to_kuzu",
               return_value=(MagicMock(), MagicMock(), graph_export_stub)), \
         patch("okf.evaluate.structural_audit",
               return_value={"self_edges": [], "cycles": []}):
        okf_results, db, graph_export = run_pipeline_staged(
            source_path=str(tmp_path / "synthetic_one_pager.pdf"),
            temp_db_path=str(tmp_path / "temp_graph.db"),
            on_progress=lambda stage, pct, detail: None,
        )

    new_rec = next(r for r in okf_results
                   if r.get("concept_name") == "Photosynthesis Overview")
    assert "Machine Learning" not in new_rec.get("prerequisites", [])

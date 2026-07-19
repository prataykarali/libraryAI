#!/usr/bin/env python3
"""Edge cases: total extraction failure must not destroy prior data;
also cover the normal (non-resume) run_pipeline path, sandboxed."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
LIB_DIR = HERE.parent.parent.parent
sys.path.insert(0, str(LIB_DIR))

import okf_pipeline
import ollama

from test_add_document import EXISTING_RESULTS, PARA, make_canned_extractor, sandbox, load_results  # noqa: F401


def test_total_extraction_failure_preserves_prior_entries(sandbox, monkeypatch):
    """Re-add a doc where EVERY chunk fails extraction: the doc's previous
    entries must NOT be silently wiped from okf_results.json."""
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", lambda *a, **k: [])
    path = sandbox / "pdfs" / "papers" / "NewDoc2026_Test.txt"

    before = load_results(sandbox)
    stale_before = [r for r in before if r["doc_id"] == "papers/NewDoc2026_Test.txt"]
    assert stale_before, "fixture must contain prior entries for the re-added doc"

    okf_pipeline.add_document(str(path))

    after = load_results(sandbox)
    stale_after = [r for r in after if r["doc_id"] == "papers/NewDoc2026_Test.txt"]
    assert stale_after, (
        "DATA LOSS: total extraction failure erased the doc's prior entries "
        "from okf_results.json"
    )


def test_run_pipeline_full_nonresume(sandbox, monkeypatch):
    """Normal full run (non-resume) works after the refactor, sandboxed."""
    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "Full"))
    out = okf_pipeline.run_pipeline(input_path=str(sandbox / "pdfs"), resume=False, local=False)
    assert out is not None
    okf_results, graph_export, accuracy = out
    assert calls, "extractor never called in full run"
    assert graph_export["stats"]["total_concepts"] > 0
    results = load_results(sandbox)
    assert len(results) == len(calls)  # okf_results.json overwritten by full run
    assert (sandbox / "graph_ui" / "okf_graph.json").exists()

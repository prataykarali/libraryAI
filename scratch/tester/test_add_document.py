#!/usr/bin/env python3
"""Adversarial verification of the incremental --add mode (mocked model).

Everything runs against a sandboxed BASE_DIR (tmp_path). ollama.chat is
poisoned to raise if anything ever touches it. extract_okf_v15 is mocked so
no real inference happens. KuzuDB ingestion runs for real (fast on tiny data)
so finalize_and_build is exercised end-to-end.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
LIB_DIR = HERE.parent.parent.parent  # .../libraryAI/libraryAI
sys.path.insert(0, str(LIB_DIR))

import okf_pipeline
import pdf_ingestion
import ollama


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PARA = (
    "Machine learning models learn statistical patterns from labelled data. "
    "A training procedure adjusts the parameters of the model so that its "
    "predictions on held out examples improve over time. Regularization "
    "techniques such as weight decay prevent the model from memorizing noise. "
)

EXISTING_RESULTS = [
    {
        "concept_name": "Attention Mechanism",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "A mechanism that weights input tokens by relevance.",
        "prerequisites": ["Linear Algebra"],
        "unlocks": ["Transformer Architecture"],
        "related_to": [{"concept": "Recurrent Networks", "relation": "contrasts_with"}],
        "tags": ["attention"],
        "doc_id": "papers/Vaswani2017_Attention_Is_All_You_Need.pdf",
        "chunk_id": "chunk_003",
        "source_category": "paper",
        "page_number": 3,
        "section_title": "3.2 Attention",
        "source_passage": "ORIGINAL PASSAGE VASWANI — must survive untouched",
    },
    {
        "concept_name": "Masked Language Modeling",
        "concept_type": "technique",
        "difficulty": "advanced",
        "summary": "Pretraining objective that predicts masked tokens.",
        "prerequisites": ["Attention Mechanism"],
        "unlocks": ["Transfer Learning"],
        "related_to": [],
        "tags": ["pretraining"],
        "doc_id": "papers/Devlin2018_BERT.pdf",
        "chunk_id": "chunk_007",
        "source_category": "paper",
        "page_number": 5,
        "section_title": "3.1 Pre-training",
        "source_passage": "ORIGINAL PASSAGE BERT — must survive untouched",
    },
    {
        # A prior entry for the doc we will RE-ADD; must get replaced.
        "concept_name": "Stale Old Concept",
        "concept_type": "definition",
        "difficulty": "intermediate",
        "summary": "Old entry from a previous ingestion of NewDoc.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": [],
        "doc_id": "papers/NewDoc2026_Test.txt",
        "chunk_id": "chunk_001",
        "source_category": "paper",
        "page_number": 0,
        "section_title": "Section",
        "source_passage": "stale passage",
    },
]


def make_canned_extractor(call_log, tag):
    """Return an extract_okf_v15 replacement that emits one concept per chunk."""
    def fake_extract(text, doc_id="", chunk_id="", page_number=0, section_title=""):
        call_log.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": text})
        n = len(call_log)
        return [{
            "concept_name": f"{tag} Concept {n:02d}",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": f"Canned summary number {n} for testing purposes.",
            "prerequisites": ["Machine Learning"],
            "unlocks": [f"{tag} Followup {n:02d}"],
            "related_to": [{"concept": "Machine Learning", "relation": "uses"}],
            "tags": ["testing"],
            "doc_id": doc_id,
            "source_category": okf_pipeline.infer_source_category(doc_id),
            "chunk_id": chunk_id,
            "page_number": page_number,
            "section_title": section_title,
        }]
    return fake_extract


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Sandboxed BASE_DIR with fixture results, pdfs tree and graph_ui dir."""
    base = tmp_path / "sandbox"
    (base / "pdfs" / "papers").mkdir(parents=True)
    (base / "pdfs" / "web_syllabi").mkdir(parents=True)
    (base / "graph_ui").mkdir()

    # Fixture corpus: two txt "papers" (txt keeps chunking fast + deterministic)
    doc_a = base / "pdfs" / "papers" / "NewDoc2026_Test.txt"
    doc_a.write_text("\n\n".join([PARA * 3] * 5), encoding="utf-8")  # ~5 chunks
    doc_b = base / "pdfs" / "papers" / "OtherDoc.txt"
    doc_b.write_text("\n\n".join([PARA * 3] * 2), encoding="utf-8")

    # Copy one REAL corpus file in for the compute_doc_id comparison.
    real_md = LIB_DIR / "pdfs" / "web_syllabi" / "AI_ML_Archipelago_Corpus_Seed.md"
    if real_md.exists():
        shutil.copy(real_md, base / "pdfs" / "web_syllabi" / real_md.name)

    with open(base / "okf_results.json", "w", encoding="utf-8") as f:
        json.dump(EXISTING_RESULTS, f, indent=2)

    monkeypatch.setattr(okf_pipeline, "BASE_DIR", base)

    # Pretend the local model is loaded; extraction itself is mocked.
    monkeypatch.setattr(okf_pipeline, "LOCAL_MODE", True)
    monkeypatch.setattr(okf_pipeline, "LOCAL_MODEL", object())
    monkeypatch.setattr(okf_pipeline, "LOCAL_TOKENIZER", object())
    monkeypatch.setattr(okf_pipeline, "load_local_model",
                        lambda: pytest.fail("load_local_model called despite mock"))

    # Poison ollama: any call = instant failure.
    def _no_ollama(*a, **k):
        raise AssertionError("ollama.chat was invoked — --add must never use Ollama")
    monkeypatch.setattr(ollama, "chat", _no_ollama)
    if hasattr(ollama, "generate"):
        monkeypatch.setattr(ollama, "generate", _no_ollama)

    return base


def load_results(base):
    with open(base / "okf_results.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3a. Add a NEW doc
# ---------------------------------------------------------------------------
def test_add_new_document(sandbox, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "Fresh"))

    path = sandbox / "pdfs" / "papers" / "OtherDoc.txt"
    out = okf_pipeline.add_document(str(path))
    assert out is not None, "add_document returned None (aborted?)"
    assert calls, "mock extractor never called"

    results = load_results(sandbox)
    doc_ids = {r["doc_id"] for r in results}
    # New doc present under folder-style doc_id
    assert "papers/OtherDoc.txt" in doc_ids
    # ALL prior entries preserved verbatim
    for orig in EXISTING_RESULTS:
        match = [r for r in results if r["doc_id"] == orig["doc_id"]
                 and r["concept_name"] == orig["concept_name"]]
        assert match, f"prior entry lost: {orig['concept_name']}"
        assert match[0] == orig, f"prior entry mutated: {orig['concept_name']}"

    new_entries = [r for r in results if r["doc_id"] == "papers/OtherDoc.txt"]
    assert len(new_entries) == len(calls)
    assert all(r.get("source_passage") for r in new_entries), "source_passage not attached"

    # Graph outputs written INSIDE the sandbox
    for name in ["okf_graph.json", "graph_audit.json", "accuracy.json"]:
        assert (sandbox / name).exists(), f"{name} missing"
    assert (sandbox / "graph_ui" / "okf_graph.json").exists()
    assert (sandbox / "okf_graph.db").exists()


# ---------------------------------------------------------------------------
# 3b. Re-add the SAME doc → replacement, no dupes, others untouched
# ---------------------------------------------------------------------------
def test_readd_replaces_prior_entries(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "Readd"))

    path = sandbox / "pdfs" / "papers" / "NewDoc2026_Test.txt"
    out = okf_pipeline.add_document(str(path))
    assert out is not None

    results = load_results(sandbox)
    mine = [r for r in results if r["doc_id"] == "papers/NewDoc2026_Test.txt"]
    # Stale entry gone, replaced by exactly the new extractions
    assert all(r["concept_name"] != "Stale Old Concept" for r in mine), \
        "stale entry for re-added doc not replaced"
    assert len(mine) == len(calls), "duplicate entries for re-added doc"

    # Other docs' provenance untouched byte-for-byte
    vaswani = [r for r in results
               if r["doc_id"] == "papers/Vaswani2017_Attention_Is_All_You_Need.pdf"]
    assert len(vaswani) == 1
    assert vaswani[0]["source_passage"] == "ORIGINAL PASSAGE VASWANI — must survive untouched"
    assert vaswani[0]["page_number"] == 3
    bert = [r for r in results if r["doc_id"] == "papers/Devlin2018_BERT.pdf"]
    assert len(bert) == 1
    assert bert[0]["source_passage"] == "ORIGINAL PASSAGE BERT — must survive untouched"
    assert bert[0]["page_number"] == 5

    # Re-add AGAIN — count must stay stable (idempotent replace)
    calls2 = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls2, "Readd"))
    okf_pipeline.add_document(str(path))
    results2 = load_results(sandbox)
    mine2 = [r for r in results2 if r["doc_id"] == "papers/NewDoc2026_Test.txt"]
    assert len(mine2) == len(calls2)
    assert len(results2) == len(results), "result count grew on identical re-add"


# ---------------------------------------------------------------------------
# 3c. compute_doc_id parity with ingest_folder
# ---------------------------------------------------------------------------
def test_compute_doc_id_variants(sandbox, monkeypatch, tmp_path):
    f = okf_pipeline.compute_doc_id
    # absolute path under pdfs/
    assert f(str(sandbox / "pdfs" / "papers" / "NewDoc2026_Test.txt")) == \
        "papers/NewDoc2026_Test.txt"
    # file at pdfs root
    root_file = sandbox / "pdfs" / "root.txt"
    root_file.write_text("x")
    assert f(str(root_file)) == "root.txt"
    # relative path under pdfs/ (resolved against CWD)
    monkeypatch.chdir(sandbox)
    assert f("pdfs/papers/NewDoc2026_Test.txt") == "papers/NewDoc2026_Test.txt"
    # file OUTSIDE pdfs/ → basename fallback
    outside = tmp_path / "elsewhere" / "Loose.pdf"
    outside.parent.mkdir()
    outside.write_text("x")
    assert f(str(outside)) == "Loose.pdf"


def test_compute_doc_id_matches_ingest_folder(sandbox):
    chunks = pdf_ingestion.ingest_folder(str(sandbox / "pdfs"))
    assert chunks, "ingest_folder produced no chunks in sandbox"
    folder_ids = {c["doc_id"] for c in chunks}
    for doc_id in folder_ids:
        path = sandbox / "pdfs" / doc_id
        assert okf_pipeline.compute_doc_id(str(path)) == doc_id, \
            f"mismatch for {doc_id}"
    # includes a REAL corpus file if it was copied in
    real = [d for d in folder_ids if d.endswith("AI_ML_Archipelago_Corpus_Seed.md")]
    if real:
        assert real[0] == "web_syllabi/AI_ML_Archipelago_Corpus_Seed.md"


# ---------------------------------------------------------------------------
# 3d. --limit caps mock call count
# ---------------------------------------------------------------------------
def test_limit_caps_chunks(sandbox, monkeypatch):
    path = sandbox / "pdfs" / "papers" / "NewDoc2026_Test.txt"
    all_chunks = pdf_ingestion.ingest_document(str(path))
    prose_total = sum(1 for c in all_chunks if c.get("chunk_kind", "prose") == "prose")
    assert prose_total >= 3, f"fixture too small ({prose_total} prose chunks)"

    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "Lim"))
    okf_pipeline.add_document(str(path), limit=2)
    assert len(calls) == 2, f"--limit 2 processed {len(calls)} chunks"

    # limit larger than doc → everything processed, no crash
    calls2 = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls2, "Lim"))
    okf_pipeline.add_document(str(path), limit=999)
    assert len(calls2) == prose_total


# ---------------------------------------------------------------------------
# 3e. Ollama never touched (poisoned in fixture; explicit re-check here)
# ---------------------------------------------------------------------------
def test_add_never_calls_ollama(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "NoOllama"))
    okf_pipeline.add_document(str(sandbox / "pdfs" / "papers" / "OtherDoc.txt"))
    # sandbox fixture poisons ollama.chat with AssertionError; reaching here
    # with successful extraction means it was never invoked.
    assert calls


def test_add_aborts_without_local_model(sandbox, monkeypatch):
    """If local model can't load, --add must abort — NOT fall back to Ollama."""
    monkeypatch.setattr(okf_pipeline, "LOCAL_MODEL", None)
    monkeypatch.setattr(okf_pipeline, "LOCAL_MODE", False)
    monkeypatch.setattr(okf_pipeline, "load_local_model", lambda: None)  # load "fails"
    before = load_results(sandbox)
    out = okf_pipeline.add_document(str(sandbox / "pdfs" / "papers" / "OtherDoc.txt"))
    assert out is None, "should abort when local model unavailable"
    assert load_results(sandbox) == before, "aborted add must not touch okf_results.json"


# ---------------------------------------------------------------------------
# 3f. run_pipeline(resume=True) still works after refactor
# ---------------------------------------------------------------------------
def test_run_pipeline_resume(sandbox, monkeypatch):
    monkeypatch.setattr(
        okf_pipeline, "extract_okf_v15",
        lambda *a, **k: pytest.fail("resume path must not extract"))
    out = okf_pipeline.run_pipeline(resume=True, local=False)
    assert out is not None
    okf_results, graph_export, accuracy = out
    # Fixture concepts made it into the graph
    names = {c["name"] for c in graph_export["concepts"].values()}
    assert "Attention Mechanism" in names
    assert "Masked Language Modeling" in names
    assert (sandbox / "okf_graph.json").exists()
    assert (sandbox / "graph_ui" / "okf_graph.json").exists()


# ---------------------------------------------------------------------------
# 5. graph_ui data contract
# ---------------------------------------------------------------------------
def test_graph_ui_data_contract(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(okf_pipeline, "extract_okf_v15", make_canned_extractor(calls, "UI"))
    okf_pipeline.add_document(str(sandbox / "pdfs" / "papers" / "OtherDoc.txt"))

    with open(sandbox / "graph_ui" / "okf_graph.json", encoding="utf-8") as f:
        data = json.load(f)

    # index.html: `if (data.visualization && data.edges)` then reads
    # data.visualization.nodes and maps data.edges via from_id/to_id/relation/
    # edge_type/source; nodes need id/label/concept_type/summary/sources[].doc_id
    assert "visualization" in data and "edges" in data and "concepts" in data
    assert "stats" in data and "graph_rag_index" in data
    vis = data["visualization"]
    for key in ("nodes", "links", "clusters", "stats"):
        assert key in vis, f"visualization.{key} missing"
    assert vis["nodes"], "no visualization nodes"
    node = vis["nodes"][0]
    for key in ("id", "label", "concept_type", "difficulty", "summary",
                "sources", "prerequisites", "unlocks", "degree", "source_count"):
        assert key in node, f"node.{key} missing"
    sourced = [n for n in vis["nodes"] if n["sources"]]
    assert sourced and all("doc_id" in s for s in sourced[0]["sources"])
    if data["edges"]:
        e = data["edges"][0]
        for key in ("from_id", "to_id", "from_name", "to_name",
                    "relation", "edge_type", "source"):
            assert key in e, f"edge.{key} missing"
    for key in ("by_type", "by_difficulty", "by_source_category"):
        assert key in vis["clusters"]

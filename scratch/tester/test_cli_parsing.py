#!/usr/bin/env python3
"""Test the actual __main__ CLI parsing block of okf_pipeline.py.

We exec the real source of the `if __name__ == "__main__":` tail inside a
namespace where run_pipeline/add_document are stubs, so the genuine parsing
code runs but nothing real executes.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
LIB_DIR = HERE.parent.parent.parent
sys.path.insert(0, str(LIB_DIR))

SOURCE = (LIB_DIR / "okf_pipeline.py").read_text(encoding="utf-8")
MARKER = 'if __name__ == "__main__":'
TAIL = SOURCE[SOURCE.index(MARKER):]


def run_cli(argv, local_mode=True):
    """Execute the real __main__ block with stubbed pipeline entry points."""
    calls = {}

    def fake_run_pipeline(input_path=None, resume=False, local=False):
        calls["run_pipeline"] = dict(input_path=input_path, resume=resume, local=local)

    def fake_add_document(path, limit=None):
        calls["add_document"] = dict(path=path, limit=limit)

    class FakeSys:
        pass
    fake_sys = FakeSys()
    fake_sys.argv = ["okf_pipeline.py"] + argv

    ns = {
        "__name__": "__main__",
        "sys": fake_sys,
        "LOCAL_MODE": local_mode,
        "MAX_PAGES_PER_DOC": None,
        "run_pipeline": fake_run_pipeline,
        "add_document": fake_add_document,
    }
    exec(TAIL, ns)
    calls["MAX_PAGES_PER_DOC"] = ns["MAX_PAGES_PER_DOC"]
    return calls


def test_add_space_form():
    c = run_cli(["--add", "pdfs/papers/X.pdf"])
    assert c["add_document"] == {"path": "pdfs/papers/X.pdf", "limit": None}
    assert "run_pipeline" not in c


def test_add_equals_form_with_limit_equals():
    c = run_cli(["--add=pdfs/papers/X.pdf", "--limit=3"])
    assert c["add_document"] == {"path": "pdfs/papers/X.pdf", "limit": 3}


def test_add_with_limit_space_form_and_local():
    c = run_cli(["--add", "doc.pdf", "--limit", "5", "--local"])
    assert c["add_document"] == {"path": "doc.pdf", "limit": 5}


def test_add_with_max_pages():
    c = run_cli(["--max-pages", "10", "--add", "doc.pdf"])
    assert c["MAX_PAGES_PER_DOC"] == 10
    assert c["add_document"]["path"] == "doc.pdf"
    c2 = run_cli(["--max-pages=7", "--add=doc.pdf"])
    assert c2["MAX_PAGES_PER_DOC"] == 7


def test_plain_run_defaults_local_when_available():
    c = run_cli([], local_mode=True)
    assert c["run_pipeline"] == {"input_path": None, "resume": False, "local": True}


def test_resume_flag():
    c = run_cli(["--resume"])
    assert c["run_pipeline"]["resume"] is True


def test_ollama_flag_disables_local():
    c = run_cli(["--ollama"])
    assert c["run_pipeline"]["local"] is False


def test_local_flag_stripped_not_treated_as_path():
    c = run_cli(["--local"])
    assert c["run_pipeline"]["input_path"] is None


def test_positional_path_still_works():
    c = run_cli(["some_folder", "--resume"])
    assert c["run_pipeline"] == {"input_path": "some_folder", "resume": True, "local": True}


def test_flag_order_independent():
    c = run_cli(["--limit", "2", "--add", "doc.pdf", "--max-pages", "4", "--resume"])
    assert c["add_document"] == {"path": "doc.pdf", "limit": 2}
    assert c["MAX_PAGES_PER_DOC"] == 4

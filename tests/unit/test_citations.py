"""Unit acceptance tests for Session 3 — Citation Correctness.

Contract under test (inference_server.py):

- build_concept_citation_map(...) assigns sequential evidence IDs S1, S2, ...
  across the whole response (no gaps, no duplicates, in order).
- validate_citations(model_output, evidence_ids) is truthy only when every
  ``[S\\d+:`` marker in the output refers to an id in evidence_ids.
- citation_payload(...) returns a dict with keys: evidence_id, topic, doc_id,
  page_number, printed_page, section_title, url, text_span.
- Page rule: a printed page label (e.g. "112") is emitted only when the
  document has a page_label_map; otherwise printed_page is None and any
  rendered label must say "PDF page N" (never a fabricated "p. N").

All database access is mocked/monkeypatched: no real Kuzu queries are issued
by these tests and no LLM/network call is made. The implementation may not
have landed yet; missing functions produce test FAILURES with explicit
messages rather than collection errors.
"""

import inspect
import re

import pytest

pytestmark = pytest.mark.unit

_SID_RE = re.compile(r"^S\d+$")


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_inference_server():
    try:
        import inference_server
    except Exception as exc:  # pragma: no cover - guard for concurrent edits
        pytest.fail(f"inference_server failed to import: {exc}")
    return inference_server


def _require(module, name):
    fn = getattr(module, name, None)
    if not callable(fn):
        pytest.fail(
            f"{module.__name__}.{name} is not implemented yet "
            f"(required by the Session 3 citation contract)"
        )
    return fn


def _evidence_record(concept_key, page_number=3, page_label_map=None):
    """A fake evidence record shaped per the get_evidence_for_concept contract."""
    text = f"Passage supporting {concept_key}."
    return {
        "chunk_id": f"chunk_{concept_key}",
        "doc_id": "math_for_ml.pdf",
        "page_number": page_number,
        "section_title": f"Section on {concept_key}",
        "text": text,
        "text_passage": text,
        "text_offset_start": 100,
        "text_offset_end": 100 + len(text),
        "block_bbox": [72.0, 144.0, 468.0, 180.0],
        "doc_title": "Mathematics for Machine Learning",
        "page_label_map": page_label_map,
    }


def _walk_sids(obj, out, mode):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (mode == "field" and key == "evidence_id"
                    and isinstance(value, str) and _SID_RE.match(value)):
                out.append(value)
            if mode == "key" and isinstance(key, str) and _SID_RE.match(key):
                out.append(key)
            _walk_sids(value, out, mode)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_sids(item, out, mode)


def _extract_sids(citation_map):
    """Collect emitted S# evidence ids, preferring explicit evidence_id fields."""
    out = []
    _walk_sids(citation_map, out, "field")
    if out:
        return out
    out = []
    _walk_sids(citation_map, out, "key")
    return out


def _validation_ok(result):
    """Contract-lenient unwrap: bool, or (bool, ids) tuple/list."""
    if isinstance(result, bool):
        return result
    try:
        return bool(result[0])
    except (TypeError, KeyError, IndexError):
        return bool(result)


_ID_PARAM_NAMES = ("evidence_id", "sid", "eid", "citation_id")
_TOPIC_PARAM_NAMES = ("topic", "topic_name", "concept", "concept_name", "name")
_EVIDENCE_PARAM_NAMES = ("evidence", "record", "source", "chunk", "row", "ev", "citation")


def _call_citation_payload(fn, evidence_id, topic, evidence):
    """Invoke citation_payload against a not-yet-frozen signature.

    Tries a signature-aware keyword call first (mapping parameter names onto
    the evidence record fields), then common positional orders. Fails the
    test (not errors) when nothing fits.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = None

    if params:
        kwargs = {}
        satisfiable = True
        for pname, p in params.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if pname in _ID_PARAM_NAMES:
                kwargs[pname] = evidence_id
            elif pname in _TOPIC_PARAM_NAMES:
                kwargs[pname] = topic
            elif pname in _EVIDENCE_PARAM_NAMES:
                kwargs[pname] = evidence
            elif pname in evidence:
                kwargs[pname] = evidence[pname]
            elif p.default is not p.empty:
                continue
            else:
                satisfiable = False
                break
        if satisfiable:
            try:
                return fn(**kwargs)
            except TypeError:
                pass

    for args in ((evidence_id, topic, evidence),
                 (evidence, evidence_id, topic),
                 (evidence,)):
        try:
            return fn(*args)
        except TypeError:
            continue

    pytest.fail(
        "citation_payload(...) could not be invoked with contract-shaped "
        "arguments (evidence_id, topic, evidence record); check its signature"
    )


def _patch_evidence_retrieval(monkeypatch, server, fake_concept_evidence):
    """Route every evidence lookup path to the fake; forbid real Kuzu access."""

    def fake_edge_evidence(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "get_concept_citations", fake_concept_evidence,
                        raising=False)
    monkeypatch.setattr(server, "get_evidence_for_concept", fake_concept_evidence,
                        raising=False)
    monkeypatch.setattr(server, "get_evidence_for_edge", fake_edge_evidence,
                        raising=False)
    try:
        import okf.graph_db as graph_db
    except Exception:
        graph_db = None
    if graph_db is not None:
        monkeypatch.setattr(graph_db, "get_evidence_for_concept",
                            fake_concept_evidence, raising=False)
        monkeypatch.setattr(graph_db, "get_evidence_for_edge",
                            fake_edge_evidence, raising=False)


# ── tests ────────────────────────────────────────────────────────────────────

def test_evidence_ids_assigned_sequentially(monkeypatch):
    """2 prereqs + target + 1 unlock -> S1..S4, sequential, no gaps/duplicates."""
    server = _import_inference_server()
    build = _require(server, "build_concept_citation_map")

    def fake_concept_evidence(concept, *args, **kwargs):
        return [_evidence_record(str(concept))]

    _patch_evidence_retrieval(monkeypatch, server, fake_concept_evidence)

    prereqs = [
        {"id": "basic_algebra", "name": "Basic Algebra",
         "summary": "Solving equations with variables."},
        {"id": "probability_theory", "name": "Probability Theory",
         "summary": "Modeling uncertainty."},
    ]
    unlocks = [
        {"id": "deep_learning", "name": "Deep Learning",
         "summary": "Multi-layer neural networks."},
    ]

    try:
        citation_map = build("machine_learning", prereqs, unlocks)
    except TypeError as exc:
        pytest.fail(
            "build_concept_citation_map no longer accepts "
            f"(target_id, prereqs, unlocks): {exc}"
        )

    sids = _extract_sids(citation_map)
    assert sids, (
        "build_concept_citation_map emitted no S# evidence ids; the contract "
        f"requires sequential ids S1..Sn across the response. Got: {citation_map!r}"
    )
    assert len(set(sids)) == len(sids), f"duplicate evidence ids emitted: {sids}"
    numbers = [int(s[1:]) for s in sids]
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"evidence ids must be S1..Sn in order with no gaps; got {sids}"
    )
    assert len(sids) == 4, (
        "2 prereqs + target + 1 unlock (one evidence record each) must yield "
        f"exactly S1..S4; got {sids}"
    )


def test_validate_citations_accepts_valid():
    server = _import_inference_server()
    validate = _require(server, "validate_citations")

    output = (
        "Linear algebra extends basic algebra to vectors and matrices "
        "[S1: math_for_ml.pdf, p. 12]. Mastering it unlocks machine learning "
        "[S2: math_for_ml.pdf, p. 20]."
    )
    result = validate(output, {"S1", "S2"})
    assert _validation_ok(result), (
        "validate_citations rejected an output whose [S#:] markers are all "
        f"present in evidence_ids; result={result!r}"
    )


def test_validate_citations_rejects_hallucinated():
    server = _import_inference_server()
    validate = _require(server, "validate_citations")

    output = (
        "Attention relates sequence positions [S1: attention.pdf, p. 3]. "
        "It also proves quantum gravity [S99: attention.pdf, p. 7]."
    )
    result = validate(output, {"S1", "S2"})
    assert not _validation_ok(result), (
        "validate_citations accepted output citing S99, which was never in "
        f"evidence_ids; result={result!r}"
    )


def test_citation_payload_structure():
    server = _import_inference_server()
    payload_fn = _require(server, "citation_payload")

    evidence = _evidence_record("gradient_descent", page_number=3)
    payload = _call_citation_payload(payload_fn, "S1", "Gradient Descent", evidence)

    assert isinstance(payload, dict), f"citation_payload must return a dict, got {type(payload)}"
    required = {"evidence_id", "topic", "doc_id", "page_number", "printed_page",
                "section_title", "url", "text_span"}
    missing = required - set(payload.keys())
    assert not missing, f"citation_payload missing contract keys: {sorted(missing)}"

    assert payload["evidence_id"] == "S1"
    assert payload["topic"] == "Gradient Descent"
    assert payload["doc_id"] == "math_for_ml.pdf"
    assert isinstance(payload["page_number"], int), (
        f"page_number must be an int (PDF 1-based), got {payload['page_number']!r}"
    )
    assert payload["page_number"] == 3
    assert isinstance(payload["url"], str) and "#page=" in payload["url"], (
        f"url must deep-link with a #page= fragment, got {payload['url']!r}"
    )
    assert payload["section_title"] == evidence["section_title"]
    assert payload["text_span"], "text_span must be a non-empty supporting span"


def test_printed_vs_pdf_page():
    server = _import_inference_server()
    payload_fn = _require(server, "citation_payload")

    # With a page_label_map mapping printed "112" -> pdf page 3, the payload
    # must surface the printed label.
    ev_with_map = _evidence_record(
        "self_attention", page_number=3, page_label_map={"112": 3}
    )
    with_map = _call_citation_payload(payload_fn, "S1", "Self-Attention", ev_with_map)
    assert with_map["printed_page"] == "112", (
        "with a page_label_map (printed '112' -> pdf 3), printed_page must be "
        f"'112'; got {with_map['printed_page']!r}"
    )
    assert with_map["page_number"] == 3

    # Without a map there is no printed label: printed_page is None and no
    # rendered text may fabricate a "p. N" printed-page claim; any rendered
    # page label must use the "PDF page N" form.
    ev_no_map = _evidence_record("self_attention", page_number=3,
                                 page_label_map=None)
    no_map = _call_citation_payload(payload_fn, "S1", "Self-Attention", ev_no_map)
    assert no_map["printed_page"] is None, (
        "without a page_label_map printed_page must be None; got "
        f"{no_map['printed_page']!r}"
    )
    for key, value in no_map.items():
        if key == "url" or not isinstance(value, str):
            continue
        assert "p. 3" not in value and "p. 112" not in value, (
            f"payload field {key!r} fabricates a printed page label without a "
            f"page_label_map: {value!r}"
        )
        if key in ("page_label", "label", "rendered", "display", "citation_label"):
            assert "PDF page" in value, (
                f"rendered label must say 'PDF page N' when no page_label_map "
                f"exists; got {value!r}"
            )

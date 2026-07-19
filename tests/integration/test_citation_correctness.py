"""Integration acceptance tests for Session 3 — Citation Correctness.

Uses a REAL Kuzu database created under pytest's tmp_path (never the repo
okf_graph.db). Exercises the okf.graph_db contract:

- add_document(doc_hash, page_count, title, edition, page_label_map)
- add_chunk(..., text_offset_start, text_offset_end, block_x/y/w/h, ...)
- get_evidence_for_concept(concept_name) -> list of dicts with chunk_id,
  doc_id, page_number (int, PDF 1-based), section_title, text,
  text_offset_start, text_offset_end, block_bbox, doc_title,
  page_label_map (dict or None)

and the inference_server citation-map build on top of that graph. No network
and no Ollama/LLM calls are made: only retrieval-side functions run, and the
Ollama client is mocked defensively. Missing contract functions fail tests
with explicit messages instead of erroring at collection.
"""

import inspect
import json

import kuzu
import pytest

pytestmark = pytest.mark.integration

DOC_ID = "flow_matching_notes.pdf"
DOC_TITLE = "Flow Matching Lecture Notes"


# ── adaptive call helpers (signatures land with ENGINEER-1/2 work) ──────────

def _import_graph_db():
    try:
        import okf.graph_db as graph_db
    except Exception as exc:
        pytest.fail(f"okf.graph_db failed to import: {exc}")
    return graph_db


def _require(module, name):
    fn = getattr(module, name, None)
    if not callable(fn):
        pytest.fail(
            f"{module.__name__}.{name} is not implemented yet "
            f"(required by the Session 3 citation contract)"
        )
    return fn


def _call_adaptive(fn, conn, /, *args, **kwargs):
    """Call fn with kwargs filtered to its signature, passing conn if wanted.

    The contract fixes the field names but not whether functions take an
    explicit connection first argument (the existing okf.graph_db style) or
    manage one internally.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = None

    accepted = kwargs
    wants_conn = False
    if params is not None:
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            accepted = {k: v for k, v in kwargs.items() if k in params}
        first = next(iter(params), "")
        wants_conn = first in ("conn", "connection", "db", "database")

    if wants_conn:
        return fn(conn, *args, **accepted)
    try:
        return fn(*args, **accepted)
    except TypeError:
        return fn(conn, *args, **accepted)


def _ensure_schema(graph_db, conn):
    """Prefer a schema helper from graph_db; otherwise create contract schema."""
    for name in ("create_schema", "ensure_schema", "init_schema", "setup_schema",
                 "_create_schema"):
        fn = getattr(graph_db, name, None)
        if callable(fn):
            try:
                fn(conn)
                return
            except TypeError:
                pass

    statements = [
        """CREATE NODE TABLE Document (
               id STRING PRIMARY KEY,
               doc_hash STRING,
               page_count INT64,
               title STRING,
               edition STRING,
               page_label_map STRING
           )""",
        """CREATE NODE TABLE Chunk (
               id STRING PRIMARY KEY,
               chunk_id STRING,
               page_number INT64,
               section_title STRING,
               text_passage STRING,
               text_offset_start INT64,
               text_offset_end INT64,
               block_x DOUBLE,
               block_y DOUBLE,
               block_w DOUBLE,
               block_h DOUBLE
           )""",
        """CREATE NODE TABLE Concept (
               id STRING PRIMARY KEY,
               name STRING,
               concept_type STRING,
               difficulty STRING,
               summary STRING
           )""",
        "CREATE REL TABLE HAS_CHUNK (FROM Document TO Chunk)",
        "CREATE REL TABLE MENTIONS (FROM Chunk TO Concept)",
        "CREATE REL TABLE REQUIRES (FROM Concept TO Concept, relation_type STRING, source STRING)",
        "CREATE REL TABLE UNLOCKS (FROM Concept TO Concept, relation_type STRING, source STRING)",
        "CREATE REL TABLE RELATED (FROM Concept TO Concept, relation_type STRING, source STRING)",
    ]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except Exception:
            pass  # table already created by a graph_db-side helper


def _add_document(graph_db, conn, doc_id, page_count, page_label_map):
    add_document = _require(graph_db, "add_document")
    return _call_adaptive(
        add_document, conn,
        doc_id=doc_id,
        doc_hash=f"hash_{doc_id}",
        page_count=page_count,
        title=DOC_TITLE,
        edition="1st",
        page_label_map=json.dumps(page_label_map) if page_label_map else None,
    )


def _add_chunk(graph_db, conn, chunk):
    add_chunk = _require(graph_db, "add_chunk")
    result = _call_adaptive(
        add_chunk, conn,
        doc_id=chunk["doc_id"],
        chunk_id=chunk["chunk_id"],
        page_number=chunk["page_number"],
        section_title=chunk["section_title"],
        text=chunk["text"],
        text_passage=chunk["text"],
        text_offset_start=chunk["text_offset_start"],
        text_offset_end=chunk["text_offset_end"],
        block_x=chunk["block_x"],
        block_y=chunk["block_y"],
        block_w=chunk["block_w"],
        block_h=chunk["block_h"],
    )
    if isinstance(result, str) and result:
        return result
    return f"{chunk['doc_id']}_{chunk['chunk_id']}"


def _add_concept(conn, concept_id, name, summary):
    safe_name = name.replace("'", "\\'")
    safe_summary = summary.replace("'", "\\'")
    conn.execute(f"""
        MERGE (c:Concept {{id: '{concept_id}'}})
        ON CREATE SET c.name = '{safe_name}',
                      c.concept_type = 'definition',
                      c.difficulty = 'intermediate',
                      c.summary = '{safe_summary}'
    """)


def _link_mention(conn, chunk_db_id, concept_id):
    safe_chunk = chunk_db_id.replace("'", "\\'")
    conn.execute(f"""
        MATCH (ch:Chunk {{id: '{safe_chunk}'}}), (c:Concept {{id: '{concept_id}'}})
        MERGE (ch)-[:MENTIONS]->(c)
    """)


def _link_requires(conn, from_id, to_id, source):
    conn.execute(f"""
        MATCH (a:Concept {{id: '{from_id}'}}), (b:Concept {{id: '{to_id}'}})
        MERGE (a)-[r:REQUIRES]->(b)
        ON CREATE SET r.relation_type = 'requires', r.source = '{source}'
    """)


def _get_evidence_for_concept(graph_db, conn, concept_name):
    fn = _require(graph_db, "get_evidence_for_concept")
    evidence = _call_adaptive(fn, conn, concept_name)
    assert isinstance(evidence, list), (
        f"get_evidence_for_concept must return a list of dicts, got {type(evidence)}"
    )
    return evidence


def _walk_citation_records(obj, out):
    """Collect dicts that look like emitted citation/evidence records."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if "doc_id" in keys and ("chunk_id" in keys or "page_number" in keys):
            out.append(obj)
        for value in obj.values():
            _walk_citation_records(value, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_citation_records(item, out)


def _resolve_chunk(conn, doc_id, chunk_id):
    """Fetch the real chunk row (doc, chunk_id, page, text) or None."""
    safe_doc = str(doc_id).replace("'", "\\'")
    safe_chunk = str(chunk_id).replace("'", "\\'")
    res = conn.execute(f"""
        MATCH (d:Document {{id: '{safe_doc}'}})-[:HAS_CHUNK]->(ch:Chunk)
        WHERE ch.chunk_id = '{safe_chunk}' OR ch.id = '{safe_chunk}'
        RETURN ch.chunk_id, ch.page_number, ch.text_passage, ch.section_title
    """)
    if res.has_next():
        row = res.get_next()
        return {
            "chunk_id": row[0],
            "page_number": int(row[1]) if row[1] is not None else None,
            "text": row[2],
            "section_title": row[3],
        }
    return None


# ── fixtures ─────────────────────────────────────────────────────────────────

CHUNKS = [
    {
        "doc_id": DOC_ID, "chunk_id": "chunk_p5", "page_number": 5,
        "section_title": "1. Background",
        "text": "Ordinary differential equations describe continuous change and underpin flow-based models.",
        "text_offset_start": 0, "text_offset_end": 93,
        "block_x": 72.0, "block_y": 100.0, "block_w": 450.0, "block_h": 60.0,
    },
    {
        "doc_id": DOC_ID, "chunk_id": "chunk_p6", "page_number": 6,
        "section_title": "2. Flow Matching",
        "text": "Flow matching trains a vector field by regressing onto conditional probability paths.",
        "text_offset_start": 93, "text_offset_end": 179,
        "block_x": 72.0, "block_y": 150.0, "block_w": 450.0, "block_h": 48.0,
    },
    {
        "doc_id": DOC_ID, "chunk_id": "chunk_p7", "page_number": 7,
        "section_title": "3. Applications",
        "text": "Diffusion transformers apply flow objectives at scale for image generation.",
        "text_offset_start": 179, "text_offset_end": 255,
        "block_x": 72.0, "block_y": 90.0, "block_w": 450.0, "block_h": 52.0,
    },
]

CONCEPTS = {
    "ordinary_differential_equations": {
        "name": "Ordinary Differential Equations",
        "summary": "Equations describing continuous change.",
        "chunk": "chunk_p5",
    },
    "flow_matching": {
        "name": "Flow Matching",
        "summary": "Training vector fields via conditional probability paths.",
        "chunk": "chunk_p6",
    },
    "diffusion_transformers": {
        "name": "Diffusion Transformers",
        "summary": "Scaled generative models using flow objectives.",
        "chunk": "chunk_p7",
    },
}


@pytest.fixture
def citation_graph(tmp_path):
    """Real Kuzu db under tmp_path populated via the graph_db contract API.

    One document spanning pages 5-7 (page_count 7), three chunks with offsets
    and bboxes, three concepts each mentioned by exactly one chunk, and
    REQUIRES edges ODE <- FlowMatching <- DiffusionTransformers.
    """
    graph_db = _import_graph_db()

    db_path = str(tmp_path / "citation_test.kuzu")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    _ensure_schema(graph_db, conn)

    _add_document(graph_db, conn, DOC_ID, page_count=7, page_label_map=None)

    chunk_db_ids = {}
    for chunk in CHUNKS:
        chunk_db_ids[chunk["chunk_id"]] = _add_chunk(graph_db, conn, chunk)

    for cid, spec in CONCEPTS.items():
        _add_concept(conn, cid, spec["name"], spec["summary"])
        _link_mention(conn, chunk_db_ids[spec["chunk"]], cid)

    _link_requires(conn, "flow_matching", "ordinary_differential_equations",
                   f"{DOC_ID}:chunk_p6")
    _link_requires(conn, "diffusion_transformers", "flow_matching",
                   f"{DOC_ID}:chunk_p7")

    yield {"db": db, "conn": conn, "graph_db": graph_db,
           "chunk_db_ids": chunk_db_ids}

    del conn
    del db


@pytest.fixture
def patched_inference_server(citation_graph, monkeypatch):
    """inference_server pointed at the tmp db, with any LLM path neutralized."""
    try:
        import inference_server
    except Exception as exc:
        pytest.fail(f"inference_server failed to import: {exc}")

    # Inject the test database connection into okf.graph_db module-level functions
    import okf.graph_db as graph_db_module
    graph_db_module.set_default_connection(citation_graph["conn"])

    monkeypatch.setattr(inference_server, "db", citation_graph["db"],
                        raising=False)
    # get_evidence_for_concept may resolve its connection through module state
    # rather than the db handle; cover the common spellings.
    for attr in ("conn", "GRAPH_CONN", "graph_conn", "DB_CONN"):
        if hasattr(inference_server, attr):
            monkeypatch.setattr(inference_server, attr, citation_graph["conn"],
                                raising=False)

    # Defensive: no Ollama/LLM call may run during retrieval-side tests.
    import ollama
    from unittest.mock import MagicMock
    monkeypatch.setattr(ollama, "Client",
                        MagicMock(side_effect=AssertionError(
                            "LLM call attempted during citation retrieval test")),
                        raising=False)
    monkeypatch.setattr(ollama, "chat",
                        MagicMock(side_effect=AssertionError(
                            "LLM call attempted during citation retrieval test")),
                        raising=False)

    return inference_server


# ── tests ────────────────────────────────────────────────────────────────────

def test_every_citation_resolves_to_chunk(citation_graph, patched_inference_server):
    """Every citation emitted by the citation-map build must resolve to a real
    chunk in the database with matching doc, page and text."""
    conn = citation_graph["conn"]
    graph_db = citation_graph["graph_db"]
    server = patched_inference_server

    # Sanity: the graph_db evidence contract itself resolves.
    evidence = _get_evidence_for_concept(graph_db, conn, "Flow Matching")
    assert evidence, "get_evidence_for_concept('Flow Matching') returned no evidence"
    for record in evidence:
        for key in ("chunk_id", "doc_id", "page_number", "section_title",
                    "text", "text_offset_start", "text_offset_end",
                    "block_bbox", "doc_title", "page_label_map"):
            assert key in record, (
                f"evidence record missing contract key {key!r}: {record!r}"
            )
        assert isinstance(record["page_number"], int)
        assert record["page_label_map"] is None or isinstance(
            record["page_label_map"], dict)

    # Build the citation map exactly as the chat endpoint would.
    build = getattr(server, "build_concept_citation_map", None)
    assert callable(build), "inference_server.build_concept_citation_map missing"
    prereqs = [{"id": "ordinary_differential_equations",
                "name": "Ordinary Differential Equations",
                "summary": CONCEPTS["ordinary_differential_equations"]["summary"]}]
    unlocks = [{"id": "diffusion_transformers",
                "name": "Diffusion Transformers",
                "summary": CONCEPTS["diffusion_transformers"]["summary"]}]
    citation_map = build("flow_matching", prereqs, unlocks)

    emitted = []
    _walk_citation_records(citation_map, emitted)
    assert emitted, (
        "citation-map build emitted no citation records for a concept with "
        f"indexed evidence; got {citation_map!r}"
    )

    for citation in emitted:
        doc_id = citation.get("doc_id")
        assert doc_id == DOC_ID, (
            f"citation references unknown document {doc_id!r}: {citation!r}"
        )
        chunk_id = citation.get("chunk_id")
        assert chunk_id, (
            f"emitted citation lacks a chunk_id, so it cannot be verified "
            f"against the corpus: {citation!r}"
        )
        real = _resolve_chunk(conn, doc_id, chunk_id)
        assert real is not None, (
            f"citation chunk {doc_id!r}/{chunk_id!r} does not exist in the db"
        )
        page = citation.get("page_number")
        assert isinstance(page, int) and page == real["page_number"], (
            f"citation page {page!r} does not match the chunk's stored page "
            f"{real['page_number']!r}"
        )
        cited_text = citation.get("text") or citation.get("text_passage") or ""
        assert cited_text.strip() == (real["text"] or "").strip(), (
            f"citation text does not match the stored chunk text.\n"
            f"cited:  {cited_text!r}\nstored: {real['text']!r}"
        )


def test_multipage_chunk_correct_page(citation_graph):
    """Material spans pages 5-7; a claim grounded on page 6 must cite page 6
    (the supporting chunk's own page), never the document's first page."""
    conn = citation_graph["conn"]
    graph_db = citation_graph["graph_db"]

    evidence = _get_evidence_for_concept(graph_db, conn, "Flow Matching")
    assert evidence, "no evidence returned for 'Flow Matching'"

    page6_chunk = next(c for c in CHUNKS if c["chunk_id"] == "chunk_p6")
    for record in evidence:
        assert record["doc_id"] == DOC_ID
        assert record["page_number"] == 6, (
            "evidence grounded on page 6 must cite page 6 (per-chunk page), "
            f"not the document's first page (5); got {record['page_number']!r}"
        )
        assert record["page_number"] != 5, (
            "citation fell back to the document's first content page instead "
            "of the supporting chunk's page"
        )
        assert (record.get("text") or "").strip() == page6_chunk["text"], (
            f"evidence text does not match the page-6 chunk: {record!r}"
        )
        assert record["text_offset_start"] == page6_chunk["text_offset_start"]
        assert record["text_offset_end"] == page6_chunk["text_offset_end"]

    # The neighboring concepts must likewise cite their own pages.
    ode = _get_evidence_for_concept(graph_db, conn,
                                    "Ordinary Differential Equations")
    assert ode and all(r["page_number"] == 5 for r in ode), (
        f"ODE evidence must cite page 5: {ode!r}"
    )
    dit = _get_evidence_for_concept(graph_db, conn, "Diffusion Transformers")
    assert dit and all(r["page_number"] == 7 for r in dit), (
        f"Diffusion Transformers evidence must cite page 7: {dit!r}"
    )


def test_get_evidence_for_edge_signature(citation_graph):
    """Verify that get_evidence_for_edge works correctly when called with/without conn."""
    conn = citation_graph["conn"]
    graph_db = citation_graph["graph_db"]

    get_evidence_for_edge = _require(graph_db, "get_evidence_for_edge")

    # 1. Test calling without conn
    evidence_no_conn = get_evidence_for_edge("Flow Matching", "Ordinary Differential Equations")
    assert evidence_no_conn is not None
    assert evidence_no_conn["doc_id"] == DOC_ID
    assert evidence_no_conn["chunk_id"] == "chunk_p6"

    # 2. Test calling with conn
    evidence_with_conn = get_evidence_for_edge(conn, "Flow Matching", "Ordinary Differential Equations")
    assert evidence_with_conn is not None
    assert evidence_with_conn["doc_id"] == DOC_ID
    assert evidence_with_conn["chunk_id"] == "chunk_p6"


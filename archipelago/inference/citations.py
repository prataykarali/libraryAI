"""Citation map, payloads, validation, and narrative recipe."""
from __future__ import annotations

import itertools

from archipelago.inference import state as st
from archipelago.inference.aliases import _node_name, pdf_page_url, markdown_pdf_link
from archipelago.inference.neighborhood import get_concept_citations
from archipelago.inference.graph_access import _default_graph_db

def compile_narrative_recipe(query, target_concept, prereqs, unlocks, citations):
    target_name = target_concept.get("label", target_concept.get("name", ""))
    target_summary = target_concept.get("summary", "")
    
    prereqs_str = "\n".join([f"   - {p['name']}: {p['summary']}" for p in prereqs]) if prereqs else "   - None extracted"
    unlocks_str = ", ".join([u['name'] for u in unlocks]) if unlocks else "None extracted"
    
    citations_str = ""
    if citations:
        for i, c in enumerate(citations[:3], 1):
            passage = c.get('text_passage') or ""
            passage = str(passage).strip()
            citations_str += (
                f"Citation {i}:\n"
                f"- Document: {c['doc_id']} (Page: {c['page_number']}, Section: {c['section_title']})\n"
                f"- Text Passage: \"{passage}\"\n\n"
            )
    else:
        citations_str = "No specific source citations found in the database.\n"
        
    recipe = (
        f"USER QUERY:\n"
        f"{query}\n\n"
        f"STRUCTURED TOPOLOGY (Narrative Recipe):\n"
        f"1. Upstream Prerequisites:\n"
        f"{prereqs_str}\n\n"
        f"2. Target Concept:\n"
        f"   - {target_name}: {target_summary}\n\n"
        f"3. Downstream Applications (Unlocks):\n"
        f"   - {unlocks_str}\n\n"
        f"TEXTUAL CITATIONS:\n"
        f"{citations_str}"
        f"INSTRUCTION:\n"
        f"You are the Generator model for the Archipelago knowledge system. Synthesize the user query, structured topology (Prerequisite -> Target -> Unlock), and textual citations into a coherent, fluid, and natural explanation. Do NOT output a JSON list of concepts. Instead, write a conversational and academic response explaining the relationship, using direct references to the concepts and citations above."
    )
    return recipe


def _resolve_printed_page(evidence):
    """Reverse-map a physical PDF page to its printed label, if a map exists.

    ``page_label_map`` maps printed labels to physical (1-based) PDF pages;
    return the label whose page matches, or None when no map exists or the
    page does not resolve.  We never guess a printed page.
    """
    label_map = evidence.get("page_label_map")
    page_number = evidence.get("page_number")
    if not isinstance(label_map, dict) or not isinstance(page_number, int):
        return None
    for label, pdf_page in label_map.items():
        try:
            if int(pdf_page) == page_number:
                return str(label)
        except (TypeError, ValueError):
            continue
    return None


def _page_display(evidence):
    """Render 'p. <printed>' only when a label map resolves; else 'PDF page <n>'."""
    printed = _resolve_printed_page(evidence)
    if printed is not None:
        return f"p. {printed}"
    page = evidence.get("page_number")
    if isinstance(page, int) and page > 0:
        return f"PDF page {page}"
    return ""


def _citation_label(topic_name, evidence_list):
    """Format only provenance we actually retrieved; never invent a page."""
    if not evidence_list:
        return ""
    evidence = evidence_list[0]
    evidence_id = evidence.get("evidence_id") or "S?"
    document = evidence.get("doc_id") or "Unknown document"
    page_display = _page_display(evidence)
    page_suffix = f", {page_display}" if page_display else ""
    return f" [{evidence_id}: {topic_name} | {document}{page_suffix}]"


def _normalize_legacy_citation(citation):
    """Lift a legacy get_concept_citations record into the evidence contract."""
    return {
        "chunk_id": None,
        "doc_id": citation.get("doc_id"),
        "page_number": citation.get("page_number"),
        "section_title": citation.get("section_title"),
        "text": citation.get("text_passage"),
        "text_offset_start": None,
        "text_offset_end": None,
        "block_bbox": None,
        "doc_title": None,
        "page_label_map": None,
    }


def _evidence_for_concept(graph_db, concept_id, concept_name):
    """Fetch evidence for one concept, falling back to the legacy Kuzu lookup."""
    if graph_db is not None:
        try:
            evidence = graph_db.get_evidence_for_concept(concept_name)
            if evidence:
                return evidence
        except Exception as e:
            print(f"Evidence retrieval error for concept '{concept_name}': {e}")
    return [_normalize_legacy_citation(c) for c in get_concept_citations(concept_id, limit=1)]


def _evidence_for_prerequisite(graph_db, prereq_name, target_name):
    """Fetch edge-level evidence for a prerequisite relation, if indexed.

    REQUIRES edges are stored target -> prerequisite (see get_graph_neighborhood),
    but the evidence API's source/target orientation is the edge author's call,
    so both orders are tried before giving up.
    """
    if graph_db is None:
        return None
    for source, target in ((target_name, prereq_name), (prereq_name, target_name)):
        try:
            evidence = graph_db.get_evidence_for_edge(source, target)
        except Exception as e:
            print(f"Evidence retrieval error for edge {source} -> {target}: {e}")
            return None
        if evidence:
            # Accept either a single evidence dict or a list of them.
            return [evidence] if isinstance(evidence, dict) else evidence
    return None


def build_concept_citation_map(target_concept, prereqs, unlocks, graph_db=None):
    """Return evidence keyed by concept id for all displayed roadmap steps.

    Evidence IDs (``S1``, ``S2``, ...) come from a single counter assigned in
    render order — prerequisites, then the target, then unlocks — so the IDs
    in the rendered text, the model contract, and the structured payloads all
    agree.  Prerequisite steps prefer edge-level evidence (the passage that
    grounds the REQUIRES relation itself) and fall back to concept-level
    evidence.  ``graph_db`` may be injected (e.g. a mock) for tests; it
    defaults to okf.graph_db, with the legacy Kuzu lookup as a last resort.
    """
    if graph_db is None:
        graph_db = _default_graph_db()
    
    # Accept both dict (with "id" key) and string (concept_id) for target_concept
    if isinstance(target_concept, str):
        target_id = target_concept
        target_name = target_concept
    else:
        target_id = target_concept.get("id", "")
        target_name = _node_name(target_concept)
    
    counter = itertools.count(1)
    citation_map = {}

    def assign(concept_id, evidence_list):
        tagged = []
        for evidence in (evidence_list or [])[:st.EVIDENCE_PER_CONCEPT]:
            entry = dict(evidence)
            entry["evidence_id"] = f"S{next(counter)}"
            tagged.append(entry)
        citation_map[concept_id] = tagged

    for item in prereqs[:st.MAX_PREREQS_SHOWN]:
        concept_id = item.get("id")
        if not concept_id or concept_id in citation_map:
            continue
        name = _node_name(item)
        evidence = _evidence_for_prerequisite(graph_db, name, target_name)
        if not evidence:
            evidence = _evidence_for_concept(graph_db, concept_id, name)
        assign(concept_id, evidence)

    if target_id not in citation_map:
        assign(target_id, _evidence_for_concept(graph_db, target_id, target_name))

    for item in unlocks[:st.MAX_UNLOCKS_SHOWN]:
        concept_id = item.get("id")
        if not concept_id or concept_id in citation_map:
            continue
        assign(concept_id, _evidence_for_concept(graph_db, concept_id, _node_name(item)))

    return citation_map


def validate_citations(model_output, evidence_ids):
    """Check that model output cites only supplied evidence IDs.

    Returns a ``(is_valid, offending_ids)`` tuple: ``is_valid`` is True when
    every ``[S<n>:`` citation bracket found in ``model_output`` names an ID
    from ``evidence_ids``; ``offending_ids`` lists the hallucinated IDs in
    numeric order (empty when valid).
    """
    supplied = set(evidence_ids or [])
    found = set(st.CITATION_ID_PATTERN.findall(model_output or ""))
    offending = sorted(found - supplied, key=lambda eid: int(eid[1:]))
    return (not offending, offending)


def citation_payload(evidence, topic, evidence_id=None):
    """Build one structured citation metadata dict for the response JSON.

    ``page_number`` is the physical (1-based) PDF page; ``printed_page`` is
    the document's printed label resolved via ``page_label_map`` (None when
    no map exists or the page does not resolve).  ``text_span`` is the exact
    supporting text: the offset-sliced span when the offsets index into the
    chunk text, otherwise the full chunk text.
    """
    if evidence_id is None:
        evidence_id = evidence.get("evidence_id")
    doc_id = evidence.get("doc_id") or ""
    page_number = evidence.get("page_number")
    url = f"{st.PDF_BASE_URL}/pdfs/{doc_id}"
    if isinstance(page_number, int) and page_number > 0:
        url += f"#page={page_number}"
    text = evidence.get("text") or ""
    start = evidence.get("text_offset_start")
    end = evidence.get("text_offset_end")
    text_span = text
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        text_span = text[start:end]
    return {
        "evidence_id": evidence_id,
        "topic": topic,
        "doc_id": doc_id,
        "page_number": page_number,
        "printed_page": _resolve_printed_page(evidence),
        "section_title": evidence.get("section_title") or "",
        "url": url,
        "text_span": text_span,
    }


def build_citation_payloads(target_concept, prereqs, unlocks, citation_map):
    """Flatten the citation map into payload dicts in evidence-ID order."""
    topics = {target_concept.get("id", ""): _node_name(target_concept)}
    for item in list(prereqs) + list(unlocks):
        topics.setdefault(item.get("id"), _node_name(item))
    payloads = []
    for concept_id, evidence_list in citation_map.items():
        for evidence in evidence_list:
            payloads.append(citation_payload(evidence, topics.get(concept_id, "")))
    return payloads


def _cite_with_link(name, evidence_list):
    """Citation bracket plus optional markdown PDF deep-link for chat bubbles."""
    base = _citation_label(name, evidence_list)
    if not evidence_list:
        return base
    ev = evidence_list[0]
    doc_id = ev.get("doc_id")
    page = ev.get("page_number")
    url = pdf_page_url(doc_id, page if isinstance(page, int) else None)
    if not url:
        return base
    page_disp = _page_display(ev) or "source"
    link = markdown_pdf_link(page_disp, doc_id, page if isinstance(page, int) else None)
    return f"{base} ({link})"


def _citation_marker(evidence_list, bare_markers=False):
    """Bare ' [S#]' marker for the generator contract (expanded post-hoc)."""
    if not bare_markers:
        return ""
    if not evidence_list:
        return ""
    eid = evidence_list[0].get("evidence_id")
    return f" [{eid}]" if eid else ""


def render_citation_from_payload(payload):
    """Deterministically render one payload as the full inline citation.

    This is the ONLY place a bracket shown to the user is built at synthesis
    time — the generator never writes doc names or page numbers itself.
    """
    eid = payload.get("evidence_id") or "S?"
    topic = payload.get("topic") or ""
    doc_id = payload.get("doc_id") or "Unknown document"
    printed = payload.get("printed_page")
    page_number = payload.get("page_number")
    if printed is not None:
        page_disp = f"p. {printed}"
    elif isinstance(page_number, int) and page_number > 0:
        page_disp = f"PDF page {page_number}"
    else:
        page_disp = ""
    page_suffix = f", {page_disp}" if page_disp else ""
    base = f"[{eid}: {topic} | {doc_id}{page_suffix}]"
    url = payload.get("url")
    if url and page_disp:
        return f"{base} ([{page_disp}]({url}))"
    return base


_MARKER_WITH_TAIL_RE = None  # compiled lazily in cleanse_model_citations


def cleanse_model_citations(text, citation_payloads):
    """Post-generation provenance pass (the anti-hallucination 'filler').

    The generator is only allowed bare ``[S1]`` markers. This pass:
      1. normalizes any ``[S1: invented stuff]`` back to ``[S1]``;
      2. strips markers whose ID has no real evidence payload;
      3. drops markers whose surrounding sentence never mentions the concept
         that evidence belongs to (misattached citations);
      4. expands surviving markers into the full deterministic bracket +
         PDF deep-link rendered from graph provenance;
      5. if nothing survives inline, appends a Sources footer so the reply is
         never shown without verifiable provenance.

    Returns the cleansed text. Never invents pages, docs, or IDs.
    """
    import re as _re

    if not text:
        return text
    payloads = [p for p in (citation_payloads or []) if p.get("evidence_id")]
    by_id = {p["evidence_id"]: p for p in payloads}

    # 1. Normalize model-embellished brackets to bare markers.
    text = _re.sub(r"\[\s*(S\d+)\s*:[^\]]*\]", r"[\1]", text)
    # Also collapse comma-run markers like [S1, S2] into [S1] [S2].
    text = _re.sub(
        r"\[\s*(S\d+(?:\s*,\s*S\d+)+)\s*\]",
        lambda m: " ".join(f"[{x.strip()}]" for x in m.group(1).split(",")),
        text,
    )

    def _topic_in_sentence(sentence, topic):
        words = [w for w in _re.split(r"[^a-z0-9]+", (topic or "").lower()) if len(w) >= 3]
        if not words:
            return True
        s = sentence.lower()
        return any(w in s for w in words)

    out = []
    last = 0
    seen_inline = 0
    for m in _re.finditer(r"\[\s*(S\d+)\s*\]", text):
        out.append(text[last:m.start()])
        last = m.end()
        payload = by_id.get(m.group(1))
        if payload is None:
            continue  # invented ID → strip silently
        # 3. Sentence check: from the previous sentence boundary to the marker.
        prefix = text[:m.start()]
        boundary = max(prefix.rfind(". "), prefix.rfind("\n"), prefix.rfind("* "))
        sentence = prefix[boundary + 1:]
        if not _topic_in_sentence(sentence, payload.get("topic")):
            continue  # misattached → drop the marker, keep the prose
        out.append(" " + render_citation_from_payload(payload))
        seen_inline += 1
    out.append(text[last:])
    cleansed = _re.sub(r"[ \t]+([.,;:!?])", r"\1", "".join(out))
    cleansed = _re.sub(r"[ \t]{2,}", " ", cleansed)

    # 5. Provenance guarantee: no inline citation survived → Sources footer.
    if seen_inline == 0 and payloads:
        lines = ["", "**Sources:**"]
        for p in payloads[:4]:
            lines.append(f"- {render_citation_from_payload(p)}")
        cleansed = cleansed.rstrip() + "\n" + "\n".join(lines)
    return cleansed



"""Kuzu neighborhood traversal and concept citations."""
from __future__ import annotations

import kuzu

from ingestion_worker import graph_lock
from archipelago.inference import state as st

_DIFFICULTY_RANK = {
    "foundational": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

# Explicit inverted / nonsense REQUIRES pairs to drop at render time
# (from_id, to_id) means "from requires to" which is wrong and should not be shown.
_BLOCKED_REQUIRES = frozenset({
    ("neural_network", "graph_neural_network"),
    ("neural_network", "dimensionality_reduction"),
    ("linear_regression", "neural_network"),
    ("eigenvalue", "neural_network"),
    ("linear_algebra", "linear_regression"),
    ("linear_regression", "support_vector_machine"),
    ("matrix_inverse", "linear_regression"),
    ("linear_algebra", "dimensionality_reduction"),
    ("linear_algebra", "nearest_neighbor"),
    ("vector", "partial_derivative"),
    ("vector", "orthonormal_basis"),
})


def _difficulty_rank(concept_id: str) -> int | None:
    node = st.CONCEPTS_DATA.get(concept_id) or {}
    diff = (node.get("difficulty") or "").lower().strip()
    return _DIFFICULTY_RANK.get(diff)


def is_plausible_prereq(target_id: str, prereq_id: str) -> bool:
    """Return False for known inverted or grossly difficulty-incoherent edges.

    Difficulty tags in the pilot graph are noisy, so we only drop when the
    prereq is *much* more advanced than the target (rank gap ≥ 2), plus an
    explicit blocklist. Specialization cases (GNN as prereq of NN) are blocked.
    """
    if not target_id or not prereq_id or target_id == prereq_id:
        return False
    if (target_id, prereq_id) in _BLOCKED_REQUIRES:
        return False
    # Specialization of neural nets should not be a prereq of the parent
    t_node = st.CONCEPTS_DATA.get(target_id) or {}
    p_node = st.CONCEPTS_DATA.get(prereq_id) or {}
    t_name = (t_node.get("label") or t_node.get("name") or target_id).lower()
    p_name = (p_node.get("label") or p_node.get("name") or prereq_id).lower()
    if "neural network" in t_name and "neural" in p_name:
        if any(m in p_name for m in ("graph neural", "convolutional", "recurrent")):
            if "graph" not in t_name and "convolutional" not in t_name:
                return False
    t_rank = _difficulty_rank(target_id)
    p_rank = _difficulty_rank(prereq_id)
    # Only drop egregious inversions (foundational requiring advanced/expert)
    if t_rank is not None and p_rank is not None and (p_rank - t_rank) >= 2:
        return False
    return True


def filter_prereqs(target_id: str, prereqs: list) -> list:
    """Drop inverted / blocked prerequisite neighbors."""
    out = []
    for p in prereqs or []:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid and is_plausible_prereq(target_id, pid):
            out.append(p)
    return out


def _citation_quality_score(section_title: str, text_passage: str, page_number: int) -> float:
    """Prefer definitional / body sections over bibliography pages."""
    score = 1.0
    sec = (section_title or "").lower()
    text = (text_passage or "").lower()
    # Heavy penalty for reference lists
    if any(t in sec for t in ("reference", "bibliograph", "works cited", "acknowledg")):
        score -= 2.5
    if text.count("et al") >= 3 or text.count("arxiv") >= 2:
        score -= 1.5
    # Prefer early-ish pages (definitions often front-loaded) but not page 0
    if isinstance(page_number, int) and page_number > 0:
        if page_number <= 5:
            score += 0.4
        elif page_number >= 200:
            score -= 0.2
    # Prefer sections that look definitional
    if any(t in sec for t in ("intro", "background", "method", "preliminar", "definition")):
        score += 0.5
    return score


def get_concept_citations(concept_id, limit=2):
    """Get source passages for one concept, with page metadata intact.

    Prefers non-bibliography, definitional passages over reference-list pages
    (fixes the 'everything cites LoRA page 14' symptom).
    """
    safe_id = str(concept_id).replace("'", "\\'")
    citations = []
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            # Fetch a wider pool, then rank
            fetch_n = max(8, int(limit) * 4)
            res = conn.execute(f"""
                MATCH (d:Document)-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->
                      (c:Concept {{id: '{safe_id}'}})
                RETURN d.id, chk.page_number, chk.section_title, chk.text_passage
                LIMIT {fetch_n}
            """)
            while res.has_next():
                row = res.get_next()
                citations.append({
                    "doc_id": row[0],
                    "page_number": int(row[1]) if row[1] is not None else 0,
                    "section_title": row[2] or "",
                    "text_passage": row[3] or "",
                })
    except Exception as e:
        print(f"Citation retrieval error for {concept_id}: {e}")
    citations.sort(
        key=lambda c: _citation_quality_score(
            c.get("section_title") or "",
            c.get("text_passage") or "",
            c.get("page_number") or 0,
        ),
        reverse=True,
    )
    return citations[: max(1, int(limit))]


def get_graph_neighborhood(concept_id, k=2):
    # Escape concept_id to prevent Cypher injection
    concept_id = str(concept_id).replace("'", "\\'")
    prereqs = []
    unlocks = []
    citations = []

    with graph_lock.read_lock():
        conn = kuzu.Connection(st.db)

        try:
            query = f"MATCH (a:Concept {{id: '{concept_id}'}})-[:REQUIRES*1..{k}]->(b:Concept) RETURN DISTINCT b.id, b.name, b.summary"
            res = conn.execute(query)
            while res.has_next():
                row = res.get_next()
                prereqs.append({"id": row[0], "name": row[1], "summary": row[2]})
        except Exception as e:
            print(f"Traversal Upstream error: {e}")

        try:
            # Traverse downstream unlocks via reverse REQUIRES (things that require this concept)
            query = f"MATCH (b:Concept)-[:REQUIRES*1..{k}]->(a:Concept {{id: '{concept_id}'}}) RETURN DISTINCT b.id, b.name, b.summary"
            res = conn.execute(query)
            while res.has_next():
                row = res.get_next()
                unlocks.append({"id": row[0], "name": row[1], "summary": row[2]})
        except Exception as e:
            print(f"Traversal Downstream (reverse REQUIRES) error: {e}")

        try:
            # Also traverse the explicit UNLOCKS edge table (concept_id UNLOCKS target)
            query = f"MATCH (a:Concept {{id: '{concept_id}'}})-[:UNLOCKS*1..{k}]->(b:Concept) RETURN DISTINCT b.id, b.name, b.summary"
            res = conn.execute(query)
            seen_unlock_ids = {u["id"] for u in unlocks}
            while res.has_next():
                row = res.get_next()
                if row[0] not in seen_unlock_ids:
                    unlocks.append({"id": row[0], "name": row[1], "summary": row[2]})
                    seen_unlock_ids.add(row[0])
        except Exception as e:
            print(f"Traversal Downstream (UNLOCKS) error: {e}")

    # Drop inverted / difficulty-incoherent prereqs (graph may still hold bad edges)
    real_id = str(concept_id).replace("\\'", "'")
    prereqs = filter_prereqs(real_id, prereqs)
    # Unlocks that only exist because of a blocked REQUIRES pair should not appear
    unlocks = [
        u for u in unlocks
        if u.get("id")
        and (u["id"], real_id) not in _BLOCKED_REQUIRES
        and (real_id, u["id"]) not in _BLOCKED_REQUIRES
    ]

    citations = get_concept_citations(concept_id, limit=3)

    if not citations and prereqs:
        for p in prereqs[:2]:
            citations.extend(get_concept_citations(p["id"], limit=1))

    return prereqs, unlocks, citations



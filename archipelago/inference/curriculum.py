"""Multi-hop curriculum chains with book/page links."""
from __future__ import annotations

import kuzu

from ingestion_worker import graph_lock
from archipelago.inference import state as st
from archipelago.inference.aliases import _node_name, pdf_page_url, markdown_pdf_link
from archipelago.inference.neighborhood import get_concept_citations, is_plausible_prereq
from archipelago.inference.citations import _normalize_legacy_citation

def _hop_provenance(concept_id, citation_map=None):
    """Resolve doc/page/url for a hop from citation_map or Kuzu evidence."""
    evidence = None
    if citation_map and concept_id in citation_map and citation_map[concept_id]:
        evidence = citation_map[concept_id][0]
    if not evidence:
        cites = get_concept_citations(concept_id, limit=1)
        if cites:
            evidence = _normalize_legacy_citation(cites[0])
    if not evidence:
        return {"doc_id": None, "page_number": None, "url": "", "evidence_id": None}
    doc_id = evidence.get("doc_id")
    page = evidence.get("page_number")
    if isinstance(page, int) and page <= 0:
        page = None
    return {
        "doc_id": doc_id,
        "page_number": page if isinstance(page, int) else None,
        "url": pdf_page_url(doc_id, page if isinstance(page, int) else None),
        "evidence_id": evidence.get("evidence_id"),
    }


def find_curriculum_chains(concept_id, max_hops=3, citation_map=None, max_paths=4):
    """Find multi-hop prerequisite chains (2–3 hops max) along REQUIRES edges.

    Returns a list of path dicts::
        {
          "nodes": [{id, name, summary, hop, doc_id, page_number, url}, ...],
          "labels": ["LoRA", "Transformer", ...],  # root → … → deepest prereq
          "markdown": "A → [B](url#page=N) → …"
        }

    Direction: root concept REQUIRES prereq1 REQUIRES prereq2 … (learn deepest first).
    Uses in-memory st.CONCEPTS_DATA + Kuzu; fails soft to [].
    """
    max_hops = max(1, min(int(max_hops or 3), 3))
    root_id = str(concept_id or "").replace("'", "\\'")
    if not root_id:
        return []

    # Build adjacency from Kuzu: from_id REQUIRES to_id (to is prerequisite)
    adj = {}
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            # Collect all REQUIRES edges among concepts reachable within max_hops
            res = conn.execute(
                f"""
                MATCH (a:Concept {{id: '{root_id}'}})-[:REQUIRES*1..{max_hops}]->(b:Concept)
                RETURN DISTINCT b.id
                """
            )
            reachable = {root_id}
            while res.has_next():
                reachable.add(res.get_next()[0])
            # Fetch edges among root + reachable
            res = conn.execute("MATCH (a:Concept)-[:REQUIRES]->(b:Concept) RETURN a.id, b.id, b.name, b.summary")
            while res.has_next():
                a_id, b_id, b_name, b_sum = res.get_next()
                if a_id not in reachable and a_id != root_id:
                    continue
                if not is_plausible_prereq(a_id, b_id):
                    continue
                adj.setdefault(a_id, []).append({
                    "id": b_id,
                    "name": b_name,
                    "summary": b_sum or "",
                })
    except Exception as e:
        print(f"find_curriculum_chains graph error: {e}")
        # Fallback: use st.CONCEPTS_DATA prerequisites lists if present
        adj = {}
        node = st.CONCEPTS_DATA.get(concept_id) or st.CONCEPTS_DATA.get(root_id) or {}
        # Build id→name map for name-based prereq lists
        name_to_id = {}
        for cid, c in st.CONCEPTS_DATA.items():
            name_to_id[(c.get("label") or c.get("name") or "").lower()] = cid
        def prereq_ids_for(cid):
            c = st.CONCEPTS_DATA.get(cid) or {}
            out = []
            for p in c.get("prerequisites") or []:
                if isinstance(p, dict) and p.get("id"):
                    if is_plausible_prereq(cid, p["id"]):
                        out.append(p)
                elif isinstance(p, str):
                    pid = name_to_id.get(p.lower())
                    if pid and is_plausible_prereq(cid, pid):
                        pc = st.CONCEPTS_DATA.get(pid, {})
                        out.append({
                            "id": pid,
                            "name": pc.get("label") or pc.get("name") or p,
                            "summary": pc.get("summary") or "",
                        })
            return out
        # BFS expand adjacency via st.CONCEPTS_DATA only
        frontier = [concept_id]
        seen = {concept_id}
        for _ in range(max_hops):
            nxt = []
            for cid in frontier:
                kids = prereq_ids_for(cid)
                if kids:
                    adj[cid] = kids
                for k in kids:
                    kid = k["id"]
                    if kid not in seen:
                        seen.add(kid)
                        nxt.append(kid)
            frontier = nxt

    # DFS/BFS collect simple paths root → prereq → … depth 1..max_hops
    paths = []

    def walk(current_id, trail):
        if len(paths) >= max_paths:
            return
        children = adj.get(current_id) or []
        if not children:
            if len(trail) >= 2:  # at least root + one prereq
                paths.append(list(trail))
            return
        extended = False
        for child in children:
            cid = child["id"]
            if any(h["id"] == cid for h in trail):
                continue  # cycle
            if len(trail) >= max_hops + 1:
                continue
            hop = {
                "id": cid,
                "name": child.get("name") or cid,
                "summary": child.get("summary") or "",
                "hop": len(trail),  # 0=root, 1=first prereq, …
            }
            walk(cid, trail + [hop])
            extended = True
            if len(paths) >= max_paths:
                return
        if not extended and len(trail) >= 2:
            paths.append(list(trail))

    root_node = st.CONCEPTS_DATA.get(concept_id) or st.CONCEPTS_DATA.get(root_id) or {}
    root_hop = {
        "id": concept_id,
        "name": _node_name(root_node) if root_node else concept_id,
        "summary": (root_node.get("summary") or "") if root_node else "",
        "hop": 0,
    }
    walk(concept_id, [root_hop])

    # Enrich with provenance + markdown
    results = []
    for trail in paths[:max_paths]:
        nodes = []
        for hop in trail:
            prov = _hop_provenance(hop["id"], citation_map)
            node = {**hop, **prov}
            nodes.append(node)
        # Learning order: deepest prerequisite first → target last
        learn_order = list(reversed(nodes))
        labels = [n["name"] for n in learn_order]
        md_parts = []
        for n in learn_order:
            if n.get("url") and n.get("doc_id"):
                page = n.get("page_number")
                page_lbl = f"p.{page}" if page else "PDF"
                md_parts.append(markdown_pdf_link(f"{n['name']} ({page_lbl})", n["doc_id"], page))
            else:
                md_parts.append(n["name"])
        results.append({
            "nodes": nodes,
            "learn_order": learn_order,
            "labels": labels,
            "markdown": " → ".join(md_parts),
            "hops": max(0, len(nodes) - 1),
        })
    # Prefer longer (richer) chains first
    results.sort(key=lambda p: p["hops"], reverse=True)
    return results


def format_curriculum_paths_section(curriculum_paths):
    """Markdown section for curriculum chains with clickable page links.

    Prefer multi-hop paths; fall back to 1-hop with a quieter label so the UI
    does not advertise "multi-hop" spam for single edges.
    """
    if not curriculum_paths:
        return ""
    multi = [p for p in curriculum_paths if int(p.get("hops") or 0) >= 2]
    paths = multi[:3] if multi else curriculum_paths[:2]
    title = (
        "\n**Multi-hop curriculum path:**"
        if multi
        else "\n**Learning path (from graph prerequisites):**"
    )
    lines = [title]
    for i, path in enumerate(paths, 1):
        md = path.get("markdown") or " → ".join(path.get("labels") or [])
        hops = path.get("hops", 0)
        if multi:
            lines.append(f"{i}. {md}" + (f"  \n   _{hops} hop{'s' if hops != 1 else ''}_" if hops else ""))
        else:
            lines.append(f"{i}. {md}")
    lines.append("\n_Click a book link to open the PDF at the cited page._")
    return "\n".join(lines)



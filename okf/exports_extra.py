"""exports_extra.py — Auxiliary graph export payloads and diffing utilities."""
import json

def export_gold_template(db, output_path: str):
    """Export a gold-standard template based on current database content."""
    import kuzu
    if hasattr(db, 'execute'):
        conn = db
    else:
        conn = kuzu.Connection(db)

    prereqs_map = {}
    unlocks_map = {}
    related_map = {}

    try:
        res = conn.execute("MATCH (a:Concept)-[:REQUIRES]->(b:Concept) RETURN a.id, b.name")
        while res.has_next():
            row = res.get_next()
            prereqs_map.setdefault(row[0], []).append(row[1])
    except Exception:
        pass

    try:
        res = conn.execute("MATCH (a:Concept)-[:UNLOCKS]->(b:Concept) RETURN a.id, b.name")
        while res.has_next():
            row = res.get_next()
            unlocks_map.setdefault(row[0], []).append(row[1])
    except Exception:
        pass

    try:
        res = conn.execute("MATCH (a:Concept)-[r:RELATED]->(b:Concept) RETURN a.id, b.name, r.relation_type")
        while res.has_next():
            row = res.get_next()
            related_map.setdefault(row[0], []).append({"concept": row[1], "relation": row[2] or "related"})
    except Exception:
        pass

    res = conn.execute("""
        MATCH (doc:Document)-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->(c:Concept)
        RETURN doc.id, chk.chunk_id, c.id, c.name, c.concept_type, c.difficulty, c.summary
    """)

    chunks_dict = {}
    while res.has_next():
        row = res.get_next()
        doc_id = row[0]
        chunk_id = row[1]
        cid = row[2]
        cname = row[3]
        ctype = row[4]
        cdiff = row[5]
        csummary = row[6]

        key = (doc_id, chunk_id)
        if key not in chunks_dict:
            chunks_dict[key] = []

        if not any(x["concept_name"] == cname for x in chunks_dict[key]):
            chunks_dict[key].append({
                "concept_name": cname,
                "concept_type": ctype,
                "difficulty": cdiff,
                "summary": csummary,
                "prerequisites": sorted(prereqs_map.get(cid, [])),
                "unlocks": sorted(unlocks_map.get(cid, [])),
                "related_to": related_map.get(cid, []),
                "tags": []
            })

    gold_template = []
    for (doc_id, chunk_id), concepts_list in chunks_dict.items():
        gold_template.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "concepts": concepts_list
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_template, f, indent=2, ensure_ascii=False)


def export_cytoscape(db, output_path: str):
    """Export database graph to Cytoscape JSON format."""
    import kuzu
    if hasattr(db, 'execute'):
        conn = db
    else:
        conn = kuzu.Connection(db)

    from okf.graph_db import export_graph
    graph_export = export_graph(conn)

    elements = []

    # Add nodes
    for cid, concept in graph_export.get("concepts", {}).items():
        elements.append({
            "data": {
                "id": cid,
                "label": concept.get("name", cid),
                "concept_type": concept.get("concept_type", "definition"),
                "difficulty": concept.get("difficulty", "intermediate"),
                "summary": concept.get("summary", "")
            }
        })

    # Add edges
    for idx, edge in enumerate(graph_export.get("edges", []), 1):
        elements.append({
            "data": {
                "id": f"edge_{idx}",
                "source": edge["from_id"],
                "target": edge["to_id"],
                "relation": edge.get("relation", "related"),
                "edge_type": edge.get("edge_type", "RELATED")
            }
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)


def diff_graphs(old_json, new_json) -> dict:
    """Calculate differences between two graph structures."""
    def _load_graph_dict(g):
        if isinstance(g, str):
            with open(g, "r", encoding="utf-8") as f:
                return json.load(f)
        return g or {}

    old_graph = _load_graph_dict(old_json)
    new_graph = _load_graph_dict(new_json)

    old_concepts = old_graph.get("concepts", {})
    new_concepts = new_graph.get("concepts", {})

    old_concept_ids = set(old_concepts.keys())
    new_concept_ids = set(new_concepts.keys())

    added_concepts = new_concept_ids - old_concept_ids
    deleted_concepts = old_concept_ids - new_concept_ids

    modified_concepts = {}
    for cid in old_concept_ids & new_concept_ids:
        old_c = old_concepts[cid]
        new_c = new_concepts[cid]
        changes = {}
        for field in ["name", "concept_type", "difficulty", "summary"]:
            if old_c.get(field) != new_c.get(field):
                changes[field] = {"old": old_c.get(field), "new": new_c.get(field)}
        if changes:
            modified_concepts[cid] = changes

    def get_edge_set(graph):
        edges = set()
        for e in graph.get("edges", []):
            edges.add((e.get("from_id"), e.get("to_id"), e.get("edge_type"), e.get("relation")))
        return edges

    old_edges = get_edge_set(old_graph)
    new_edges = get_edge_set(new_graph)

    added_edges = new_edges - old_edges
    deleted_edges = old_edges - new_edges

    return {
        "concepts": {
            "added": list(added_concepts),
            "deleted": list(deleted_concepts),
            "modified": modified_concepts
        },
        "edges": {
            "added": [
                {"from_id": e[0], "to_id": e[1], "edge_type": e[2], "relation": e[3]}
                for e in added_edges
            ],
            "deleted": [
                {"from_id": e[0], "to_id": e[1], "edge_type": e[2], "relation": e[3]}
                for e in deleted_edges
            ]
        }
    }

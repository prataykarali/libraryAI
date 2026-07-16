"""Graph export payloads: visualization, GraphRAG index, and audits."""

from collections import Counter

from okf.canonicalize import is_same_concept_reference
from okf.cleanup import is_valid_concept_name
from okf.config import infer_source_category
from okf.util import _dedupe_dicts, _record_sources, create_concept_id


def build_visual_graph(okf_results: list, graph_export: dict) -> dict:
    """Build an Obsidian-style graph payload for local visualization."""
    nodes = {}

    for cid, concept in graph_export.get("concepts", {}).items():
        nodes[cid] = {
            "id": cid,
            "label": concept.get("name", cid),
            "concept_type": concept.get("concept_type", "definition"),
            "difficulty": concept.get("difficulty", "intermediate"),
            "summary": concept.get("summary", ""),
            "tags": [],
            "sources": [],
            "source_categories": [],
            "sections": [],
            "prerequisites": [],
            "unlocks": [],
            "related": [],
            "degree": 0,
            "source_count": 0,
        }

    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        cid = create_concept_id(name)
        node = nodes.setdefault(cid, {
            "id": cid,
            "label": name,
            "concept_type": result.get("concept_type", "definition"),
            "difficulty": result.get("difficulty", "intermediate"),
            "summary": result.get("summary", ""),
            "tags": [],
            "sources": [],
            "source_categories": [],
            "sections": [],
            "prerequisites": [],
            "unlocks": [],
            "related": [],
            "degree": 0,
            "source_count": 0,
        })
        if len(result.get("summary", "")) > len(node.get("summary", "")):
            node["summary"] = result.get("summary", "")
        node["concept_type"] = result.get("concept_type", node["concept_type"])
        node["difficulty"] = result.get("difficulty", node["difficulty"])
        node["tags"].extend(result.get("tags", []))
        # Use the full (possibly merged) provenance list so cross-document
        # evidence unioned during dedup is preserved, not collapsed to one row.
        node["sources"].extend(_record_sources(result))
        category = result.get("source_category") or infer_source_category(result.get("doc_id", ""))
        if category:
            node["source_categories"].append(category)
        if result.get("section_title"):
            node["sections"].append(result["section_title"])
        node["prerequisites"].extend(result.get("prerequisites", []))
        node["unlocks"].extend(result.get("unlocks", []))
        node["related"].extend(result.get("related_to", []))

    links = []
    degree_counts = Counter()
    for idx, edge in enumerate(graph_export.get("edges", []), 1):
        source = edge["from_id"]
        target = edge["to_id"]
        degree_counts[source] += 1
        degree_counts[target] += 1
        links.append({
            "id": f"edge_{idx:05d}",
            "source": source,
            "target": target,
            "source_label": edge.get("from_name", source),
            "target_label": edge.get("to_name", target),
            "edge_type": edge.get("edge_type", "RELATED"),
            "relation": edge.get("relation", "related"),
            "source_ref": edge.get("source", ""),
        })

    for node in nodes.values():
        node["tags"] = sorted(set(t for t in node["tags"] if t))
        node["sources"] = _dedupe_dicts(node["sources"])
        node["source_categories"] = sorted(set(node["source_categories"]))
        node["sections"] = sorted(set(node["sections"]))
        node["prerequisites"] = sorted(set(node["prerequisites"]))
        node["unlocks"] = sorted(set(node["unlocks"]))
        node["source_count"] = len(node["sources"])
        node["degree"] = degree_counts[node["id"]]

    return {
        "nodes": sorted(nodes.values(), key=lambda x: x["label"].lower()),
        "links": links,
        "clusters": {
            "by_type": dict(Counter(n["concept_type"] for n in nodes.values())),
            "by_difficulty": dict(Counter(n["difficulty"] for n in nodes.values())),
            "by_source_category": dict(Counter(
                category
                for n in nodes.values()
                for category in (n.get("source_categories") or ["unknown"])
            )),
        },
        "stats": {
            "node_count": len(nodes),
            "link_count": len(links),
            "max_degree": max(degree_counts.values()) if degree_counts else 0,
        }
    }


def build_graph_rag_index(okf_results: list, graph_export: dict) -> dict:
    """Create a compact concept-neighborhood index for GraphRAG retrieval."""
    visual = graph_export.get("visualization") or build_visual_graph(okf_results, graph_export)
    by_id = {node["id"]: node for node in visual["nodes"]}
    index = {}

    for node_id, node in by_id.items():
        requires = []
        unlocks = []
        related = []
        for link in visual["links"]:
            if link["source"] != node_id:
                continue
            target_name = by_id.get(link["target"], {}).get("label", link["target_label"])
            if link["edge_type"] == "REQUIRES":
                requires.append(target_name)
            elif link["edge_type"] == "UNLOCKS":
                unlocks.append(target_name)
            else:
                related.append({"concept": target_name, "relation": link["relation"]})

        retrieval_terms = sorted(set(
            [node["label"], node.get("concept_type", ""), node.get("difficulty", "")] +
            node.get("tags", []) + requires + unlocks +
            [r["concept"] for r in related]
        ))
        index[node_id] = {
            "name": node["label"],
            "summary": node.get("summary", ""),
            "concept_type": node.get("concept_type", "definition"),
            "difficulty": node.get("difficulty", "intermediate"),
            "requires": sorted(set(requires)),
            "unlocks": sorted(set(unlocks)),
            "related": related,
            "sources": node.get("sources", []),
            "retrieval_terms": retrieval_terms,
            "retrieval_text": " | ".join(t for t in retrieval_terms if t),
        }

    return {
        "version": "okf-graphrag-v1",
        "concepts": index,
        "stats": {
            "total_concepts": len(index),
            "total_links": len(visual.get("links", [])),
        }
    }


def audit_graph_export(graph_export: dict) -> dict:
    """Compute deterministic graph issues that fine-tuning cannot guarantee."""
    concepts = graph_export.get("concepts", {})
    edges = graph_export.get("edges", [])
    visual = graph_export.get("visualization", {})
    visual_nodes = {n.get("id"): n for n in visual.get("nodes", [])}

    invalid_nodes = [
        c.get("name", cid)
        for cid, c in concepts.items()
        if not is_valid_concept_name(c.get("name", ""))
    ]
    empty_summary_nodes = [
        c.get("name", cid)
        for cid, c in concepts.items()
        if not (c.get("summary") or "").strip()
    ]
    placeholder_nodes = [
        n.get("label", node_id)
        for node_id, n in visual_nodes.items()
        if not n.get("sources")
    ]

    self_edges = [
        e for e in edges
        if e.get("from_id") == e.get("to_id")
        or is_same_concept_reference(e.get("from_name", ""), e.get("to_name", ""))
    ]

    requires_pairs = {
        (e.get("from_id"), e.get("to_id"))
        for e in edges
        if e.get("edge_type") == "REQUIRES"
    }
    reciprocal_requires = []
    seen = set()
    for a, b in requires_pairs:
        if (b, a) in requires_pairs and tuple(sorted((a, b))) not in seen:
            seen.add(tuple(sorted((a, b))))
            reciprocal_requires.append({
                "a": concepts.get(a, {}).get("name", a),
                "b": concepts.get(b, {}).get("name", b),
            })

    return {
        "stats": {
            "invalid_nodes": len(invalid_nodes),
            "empty_summary_nodes": len(empty_summary_nodes),
            "placeholder_nodes": len(placeholder_nodes),
            "self_edges": len(self_edges),
            "reciprocal_requires": len(reciprocal_requires),
        },
        "examples": {
            "invalid_nodes": invalid_nodes[:25],
            "empty_summary_nodes": empty_summary_nodes[:25],
            "placeholder_nodes": placeholder_nodes[:25],
            "self_edges": [
                {
                    "from": e.get("from_name"),
                    "to": e.get("to_name"),
                    "edge_type": e.get("edge_type"),
                }
                for e in self_edges[:25]
            ],
            "reciprocal_requires": reciprocal_requires[:25],
        }
    }

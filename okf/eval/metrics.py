from okf.eval._base import *  # noqa: F403

def evaluate_extraction(okf_results: list, total_chunks: int, graph_export: dict,
                        raw_extraction_count: int = 0) -> dict:
    """Compute proxy accuracy metrics for the extraction pipeline."""
    if raw_extraction_count == 0:
        raw_extraction_count = len(okf_results)

    # 1. Extraction Rate (raw, before cleanup)
    extraction_rate = (raw_extraction_count / total_chunks * 100) if total_chunks > 0 else 0

    # 2. Schema Completeness — do all results have all required fields?
    required_fields = ["concept_name", "summary", "prerequisites", "unlocks"]
    expanded_fields = ["concept_type", "difficulty", "related_to", "tags"]
    complete_core = 0
    complete_expanded = 0
    for r in okf_results:
        if all(r.get(f) for f in required_fields):
            complete_core += 1
        if all(r.get(f) is not None for f in required_fields + expanded_fields):
            complete_expanded += 1

    schema_completeness_core = (complete_core / len(okf_results) * 100) if okf_results else 0
    schema_completeness_full = (complete_expanded / len(okf_results) * 100) if okf_results else 0

    # 3. Concept Quality — are names reasonable?
    good_names = 0
    for r in okf_results:
        name = r.get("concept_name", "")
        if name and len(name) < 60 and len(name.split()) <= 8 and not name.endswith("."):
            good_names += 1
    concept_quality = (good_names / len(okf_results) * 100) if okf_results else 0

    # 4. Relation Consistency — do prereqs/unlocks point to known graph nodes?
    all_concept_names = {r.get("concept_name", "").lower() for r in okf_results}
    total_refs = 0
    resolved_refs = 0
    for r in okf_results:
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p:
                total_refs += 1
                if p.lower() in all_concept_names:
                    resolved_refs += 1
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u:
                total_refs += 1
                if u.lower() in all_concept_names:
                    resolved_refs += 1

    relation_consistency = (resolved_refs / total_refs * 100) if total_refs > 0 else 0

    # 5. DAG Validity — check for self-loops (cycle detection is expensive)
    self_loops = 0
    for r in okf_results:
        name = r.get("concept_name", "").lower()
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p.lower() == name:
                self_loops += 1
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u.lower() == name:
                self_loops += 1
    dag_validity = 100.0 if self_loops == 0 else max(0, 100 - self_loops * 10)

    # 6. Type Distribution — how well does the model differentiate types?
    type_counts = Counter(r.get("concept_type", "unknown") for r in okf_results)
    type_diversity = len(type_counts)

    # 7. Difficulty Distribution
    diff_counts = Counter(r.get("difficulty", "unknown") for r in okf_results)

    # 8. Graph connectivity
    total_concepts = graph_export["stats"]["total_concepts"]
    total_edges = graph_export["stats"]["total_edges"]
    edge_density = (total_edges / total_concepts) if total_concepts > 0 else 0

    # Orphan nodes (no edges at all)
    connected_ids = set()
    for e in graph_export["edges"]:
        connected_ids.add(e["from_id"])
        connected_ids.add(e["to_id"])
    orphan_count = total_concepts - len(connected_ids)
    connectivity = ((total_concepts - orphan_count) / total_concepts * 100) if total_concepts > 0 else 0

    # Composite Score (weighted — uses extraction_rate, not post-cleanup count)
    composite = (
        extraction_rate * 0.15 +
        schema_completeness_core * 0.15 +
        concept_quality * 0.15 +
        relation_consistency * 0.15 +
        dag_validity * 0.15 +
        connectivity * 0.25
    )

    return {
        "overall_score": round(composite, 1),
        "breakdown": {
            "extraction_rate": round(extraction_rate, 1),
            "schema_completeness_core": round(schema_completeness_core, 1),
            "schema_completeness_expanded": round(schema_completeness_full, 1),
            "concept_quality": round(concept_quality, 1),
            "relation_consistency": round(relation_consistency, 1),
            "dag_validity": round(dag_validity, 1),
            "connectivity": round(connectivity, 1),
            "edge_density": round(edge_density, 2),
        },
        "distributions": {
            "concept_types": dict(type_counts),
            "difficulty_levels": dict(diff_counts),
        },
        "stats": {
            "total_chunks": total_chunks,
            "chunks_with_extractions": raw_extraction_count,
            "after_cleanup": len(okf_results),
            "failed_extractions": total_chunks - raw_extraction_count,
            "concepts_after_cleanup": len(okf_results),
            "total_concepts_in_graph": total_concepts,
            "total_edges_in_graph": total_edges,
            "orphan_nodes": orphan_count,
            "self_loops": self_loops,
        }
    }

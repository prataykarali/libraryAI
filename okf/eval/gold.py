import re
from okf.eval._base import *  # noqa: F403
from okf.eval.structural import structural_audit

def load_gold_graph(path: str):
    """Load gold graph from a JSON file path."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_names(input_data) -> set:
    if isinstance(input_data, dict):
        # Gold files / graph exports often wrap payloads as
        # {"concepts": [...|...], "edges": [...]}. Prefer the nested concepts
        # payload so top-level keys like "concepts"/"edges" are never treated
        # as concept names (historical FN=2 bug).
        if "concepts" in input_data and isinstance(input_data["concepts"], (list, dict, set)):
            nested = _extract_names(input_data["concepts"])
            if nested:
                return nested
        names = set()
        for k, v in input_data.items():
            if isinstance(v, dict):
                name = v.get("name") or v.get("concept_name")
                if name:
                    names.add(name.strip().lower())
            elif isinstance(v, str):
                names.add(v.strip().lower())
        if not names:
            # Last resort: keys as names only when values are not concept records
            # and this is not a concepts/edges wrapper (empty concepts list).
            if set(input_data.keys()) <= {"concepts", "edges", "stats", "visualization", "graph_rag_index"}:
                return set()
            names = {k.strip().lower() for k in input_data.keys()}
        return names
    elif isinstance(input_data, list):
        names = set()
        for c in input_data:
            if isinstance(c, dict):
                if "concepts" in c:
                    for sub_c in c["concepts"]:
                        name = sub_c.get("concept_name") or sub_c.get("name")
                        if name:
                            names.add(name.strip().lower())
                else:
                    name = c.get("concept_name") or c.get("name")
                    if name:
                        names.add(name.strip().lower())
            elif isinstance(c, str):
                names.add(c.strip().lower())
        return names
    elif isinstance(input_data, set):
        return {str(x).strip().lower() for x in input_data}
    return set()

def _canonical_key(name: str) -> str:
    """Gold-comparison key: alias resolution (ALIAS_MAP) + canonical form +
    concept-key normalization, so 'LoRA' and 'Low-Rank Adaptation' compare equal."""
    # _concept_key comes from okf.canonicalize via okf.eval._base
    return _concept_key(canonicalize_name(name))

def _extract_name_docs(input_data) -> dict:
    """Map lowered concept name → set of provenance doc_ids.

    Only records that carry a "sources" list with doc_id entries appear here;
    names without provenance are simply absent (callers treat them as
    unrestricted when doc-scoping)."""
    docs = {}

    def _record(rec):
        name = rec.get("name") or rec.get("concept_name")
        if not name:
            return
        doc_ids = {s.get("doc_id") for s in rec.get("sources", [])
                   if isinstance(s, dict) and s.get("doc_id")}
        if doc_ids:
            docs.setdefault(name.strip().lower(), set()).update(doc_ids)

    if isinstance(input_data, dict):
        payload = input_data.get("concepts", input_data)
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, dict):
                    _record(v)
        elif isinstance(payload, list):
            for c in payload:
                if isinstance(c, dict):
                    _record(c)
    elif isinstance(input_data, list):
        for c in input_data:
            if isinstance(c, dict):
                if isinstance(c.get("concepts"), list):
                    for sub_c in c["concepts"]:
                        if isinstance(sub_c, dict):
                            _record(sub_c)
                else:
                    _record(c)
    return docs

def compare_concepts(extracted, gold, doc_id=None) -> dict:
    """Calculate precision, recall, and F1 score of extracted concepts vs gold concepts.

    Names on BOTH sides are normalized through okf.canonicalize
    (canonicalize_name + _concept_key), so aliases like "LoRA" vs
    "Low-Rank Adaptation" count as matches. Names still unmatched after key
    normalization fall back to rapidfuzz token_set_ratio >= FUZZY_MATCH_THRESHOLD.

    If doc_id is given (gold sets are document-scoped), extracted concepts
    whose provenance names other documents only are excluded before
    precision/FP; concepts without provenance data are kept.
    """
    ext_set = _extract_names(extracted)
    gold_set = _extract_names(gold)

    if doc_id:
        name_docs = _extract_name_docs(extracted)
        ext_set = {n for n in ext_set
                   if n not in name_docs or doc_id in name_docs[n]}

    if not ext_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                "matched_pairs": [], "normalization": "canonical_v2"}
    if not ext_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0, "false_positives": 0, "false_negatives": len(gold_set),
                "matched_pairs": [], "normalization": "canonical_v2"}
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0, "false_positives": len(ext_set), "false_negatives": 0,
                "matched_pairs": [], "normalization": "canonical_v2"}

    # Group raw names by canonical key (near-duplicate raw names collapse).
    ext_keys = {}
    for n in ext_set:
        ext_keys.setdefault(_canonical_key(n), set()).add(n)
    gold_keys = {}
    for n in gold_set:
        gold_keys.setdefault(_canonical_key(n), set()).add(n)

    # 1. Exact match on canonical keys
    matched_keys = ext_keys.keys() & gold_keys.keys()
    matched_pairs = [
        [sorted(ext_keys[k])[0], sorted(gold_keys[k])[0]]
        for k in sorted(matched_keys)
    ]
    unmatched_ext = sorted(k for k in ext_keys if k not in matched_keys)
    unmatched_gold = sorted(k for k in gold_keys if k not in matched_keys)

    # 2. Fuzzy fallback: greedy best-score pairing of remaining keys
    if _rf_fuzz is not None and unmatched_ext and unmatched_gold:
        candidates = []
        for gk in unmatched_gold:
            for ek in unmatched_ext:
                score = _rf_fuzz.token_set_ratio(gk, ek)
                if score >= FUZZY_MATCH_THRESHOLD:
                    candidates.append((score, gk, ek))
        candidates.sort(reverse=True)
        used_gold, used_ext = set(), set()
        for score, gk, ek in candidates:
            if gk in used_gold or ek in used_ext:
                continue
            used_gold.add(gk)
            used_ext.add(ek)
            matched_pairs.append([sorted(ext_keys[ek])[0], sorted(gold_keys[gk])[0]])
        unmatched_ext = [k for k in unmatched_ext if k not in used_ext]
        unmatched_gold = [k for k in unmatched_gold if k not in used_gold]

    tp = len(matched_pairs)
    fp = len(unmatched_ext)
    fn = len(unmatched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "matched_pairs": matched_pairs,
        "normalization": "canonical_v2"
    }

def _extract_edges(input_data) -> set:
    edges = set()
    if isinstance(input_data, dict):
        if "edges" in input_data:
            for e in input_data["edges"]:
                # Support export schema (from_name/to_name) and gold files (source/target).
                src = e.get("from_name") or e.get("from_id") or e.get("source") or e.get("from")
                dst = e.get("to_name") or e.get("to_id") or e.get("target") or e.get("to")
                etype = str(e.get("edge_type") or e.get("type") or "").upper()
                if src and dst:
                    src_l = str(src).strip().lower()
                    dst_l = str(dst).strip().lower()
                    if etype == "REQUIRES":
                        # Gold files encode requires as source→target (learner path).
                        # Export edges use from_name REQUIRES to_name with inverted dependency storage elsewhere.
                        if e.get("source") or e.get("target"):
                            edges.add((src_l, dst_l))
                        else:
                            edges.add((dst_l, src_l))
                    elif etype == "UNLOCKS":
                        edges.add((src_l, dst_l))
                    else:
                        edges.add((min(src_l, dst_l), max(src_l, dst_l)))
        else:
            for cid, c in input_data.items():
                if isinstance(c, dict):
                    name = c.get("name") or c.get("concept_name")
                    if name:
                        name_l = name.strip().lower()
                        for p in c.get("prerequisites", []):
                            edges.add((p.strip().lower(), name_l))
                        for u in c.get("unlocks", []):
                            edges.add((name_l, u.strip().lower()))
                        for r in c.get("related", []):
                            if isinstance(r, dict) and r.get("concept"):
                                rc = r["concept"].strip().lower()
                                edges.add((min(name_l, rc), max(name_l, rc)))
    elif isinstance(input_data, list):
        if input_data and isinstance(input_data[0], dict) and ("from_name" in input_data[0] or "source" in input_data[0] or "from_id" in input_data[0] or "from" in input_data[0]):
            for e in input_data:
                src = e.get("source") or e.get("from_name") or e.get("from_id") or e.get("from")
                dst = e.get("target") or e.get("to_name") or e.get("to_id") or e.get("to")
                etype = str(e.get("edge_type") or e.get("type") or "").upper()
                if src and dst:
                    src_l = src.strip().lower()
                    dst_l = dst.strip().lower()
                    if etype in ("REQUIRES", "requires"):
                        if "from_name" in e:
                            edges.add((dst_l, src_l))
                        else:
                            edges.add((src_l, dst_l))
                    elif etype in ("UNLOCKS", "unlocks"):
                        edges.add((src_l, dst_l))
                    else:
                        edges.add((min(src_l, dst_l), max(src_l, dst_l)))
        else:
            for c in input_data:
                if isinstance(c, dict):
                    if "concepts" in c:
                        for sub_c in c["concepts"]:
                            name = sub_c.get("concept_name")
                            if name:
                                name_l = name.strip().lower()
                                for p in sub_c.get("prerequisites", []):
                                    edges.add((p.strip().lower(), name_l))
                                for u in sub_c.get("unlocks", []):
                                    edges.add((name_l, u.strip().lower()))
                                for r in sub_c.get("related_to", []):
                                    if isinstance(r, dict) and r.get("concept"):
                                        rc = r["concept"].strip().lower()
                                        edges.add((min(name_l, rc), max(name_l, rc)))
                    else:
                        name = c.get("concept_name") or c.get("name")
                        if name:
                            name_l = name.strip().lower()
                            for p in c.get("prerequisites", []):
                                edges.add((p.strip().lower(), name_l))
                            for u in c.get("unlocks", []):
                                edges.add((name_l, u.strip().lower()))
                            for r in c.get("related_to", []):
                                if isinstance(r, dict) and r.get("concept"):
                                    rc = r["concept"].strip().lower()
                                    edges.add((min(name_l, rc), max(name_l, rc)))
    return edges

def compare_edges(extracted, gold) -> dict:
    """Calculate precision, recall, and F1 of edges, and direction accuracy."""
    ext_edges = _extract_edges(extracted)
    gold_edges = _extract_edges(gold)

    # Convert edges to directed sets and undirected maps
    ext_undir = {}
    for u, v in ext_edges:
        ext_undir[frozenset([u, v])] = (u, v)

    gold_undir = {}
    for u, v in gold_edges:
        gold_undir[frozenset([u, v])] = (u, v)

    # Calculate metrics on directed edges
    if not ext_edges and not gold_edges:
        dir_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
    else:
        tp = len(ext_edges & gold_edges)
        fp = len(ext_edges - gold_edges)
        fn = len(gold_edges - ext_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        dir_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    # Calculate metrics on undirected edges
    if not ext_undir and not gold_undir:
        undir_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
    else:
        tp = len(ext_undir.keys() & gold_undir.keys())
        fp = len(ext_undir.keys() - gold_undir.keys())
        fn = len(gold_undir.keys() - ext_undir.keys())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        undir_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

    # Calculate direction accuracy for matched undirected edges
    matched_undir = ext_undir.keys() & gold_undir.keys()
    correct_directions = 0
    for edge in matched_undir:
        if ext_undir[edge] == gold_undir[edge]:
            correct_directions += 1

    dir_accuracy = correct_directions / len(matched_undir) if matched_undir else 1.0

    return {
        "directed": dir_metrics,
        "undirected": undir_metrics,
        "direction_accuracy": dir_accuracy
    }

def evaluate_pipeline(db, gold_path: str) -> dict:
    """Run the complete evaluation pipeline comparing the DB graph against gold data."""
    gold_data = load_gold_graph(gold_path)

    import kuzu
    if hasattr(db, 'execute'):
        conn = db
    else:
        conn = kuzu.Connection(db)

    from okf.graph_db import export_graph
    db_export = export_graph(conn)

    # Gold files use {"concepts": [...], "edges": [...]}; pass concepts list for name match.
    if isinstance(gold_data, dict) and "concepts" in gold_data:
        gold_concepts = gold_data["concepts"]
    else:
        gold_concepts = gold_data

    concept_metrics = compare_concepts(db_export["concepts"], gold_concepts)
    edge_metrics = compare_edges(db_export, gold_data)
    audit_metrics = structural_audit(db)

    return {
        "concept_comparison": concept_metrics,
        "edge_comparison": edge_metrics,
        "structural_audit": audit_metrics
    }

def print_report(report: dict):
    """Print a formatted report of the pipeline evaluation metrics."""
    print("\n" + "=" * 70)
    print("OKF PIPELINE EVALUATION REPORT")
    print("=" * 70)

    cc = report.get("concept_comparison", {})
    print("\nConcept Comparison Metrics:")
    print(f"  Precision: {cc.get('precision', 0.0):.2%}")
    print(f"  Recall:    {cc.get('recall', 0.0):.2%}")
    print(f"  F1 Score:  {cc.get('f1', 0.0):.2%}")
    print(f"  True Positives:  {cc.get('true_positives', 0)}")
    print(f"  False Positives: {cc.get('false_positives', 0)}")
    print(f"  False Negatives: {cc.get('false_negatives', 0)}")

    em = report.get("edge_comparison", {})
    dir_e = em.get("directed", {})
    undir_e = em.get("undirected", {})
    print("\nEdge Comparison Metrics (Directed):")
    print(f"  Precision: {dir_e.get('precision', 0.0):.2%}")
    print(f"  Recall:    {dir_e.get('recall', 0.0):.2%}")
    print(f"  F1 Score:  {dir_e.get('f1', 0.0):.2%}")

    print("\nEdge Comparison Metrics (Undirected):")
    print(f"  Precision: {undir_e.get('precision', 0.0):.2%}")
    print(f"  Recall:    {undir_e.get('recall', 0.0):.2%}")
    print(f"  F1 Score:  {undir_e.get('f1', 0.0):.2%}")
    print(f"  Direction Accuracy: {em.get('direction_accuracy', 0.0):.2%}")

    sa = report.get("structural_audit", {})
    print("\nStructural Audit:")
    print(f"  Total Self-Loops / Self-Edges: {len(sa.get('self_edges', []))}")
    print(f"  Total Cycles Detected:         {len(sa.get('cycles', []))}")
    print(f"  Orphan Count:                  {sa.get('orphan_count', 0)}")
    print(f"  Orphan Percentage:             {sa.get('orphan_percentage', 0.0):.2%}")
    print(f"  Edge Provenance Issues:        {len(sa.get('edge_provenance_issues', []))}")
    print(f"  Connected Components Count:    {sa.get('connected_components_count', 0)}")
    print("=" * 70 + "\n")

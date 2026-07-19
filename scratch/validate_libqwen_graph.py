#!/usr/bin/env python3
"""Post lib-qwen re-ingest validation gates (Session 1)."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

def main():
    results = json.loads((ROOT / "okf_results.json").read_text())
    graph = json.loads((ROOT / "okf_graph.json").read_text())
    concepts = graph.get("concepts") or graph.get("nodes") or []
    edges = graph.get("edges") or graph.get("relations") or []
    stats = graph.get("stats") or {}

    print("=== RECORDS ===")
    print(f"okf_results: {len(results)}")
    by_doc = Counter(r.get("doc_id", "?") for r in results)
    for d, c in by_doc.most_common():
        print(f"  {c:4d}  {d}")

    print("\n=== GRAPH ===")
    print(f"concepts: {len(concepts)}")
    print(f"edges: {len(edges)}")
    print(f"stats: {stats}")

    # Relation fill rates on raw results
    n = max(len(results), 1)
    has_p = sum(1 for r in results if r.get("prerequisites"))
    has_u = sum(1 for r in results if r.get("unlocks"))
    has_r = sum(1 for r in results if r.get("related_to"))
    print(f"\n=== RELATION FILL (raw records) ===")
    print(f"prereq nonempty: {has_p}/{len(results)} ({100*has_p/n:.1f}%)")
    print(f"unlock nonempty: {has_u}/{len(results)} ({100*has_u/n:.1f}%)")
    print(f"related nonempty: {has_r}/{len(results)} ({100*has_r/n:.1f}%)")

    # Weak concept probes
    names = {(c.get("name") or c.get("concept_name") or c.get("label") or "").lower() for c in concepts}
    probes = ["pca", "principal component", "linear algebra", "lora", "attention", "rag", "bert", "transformer"]
    print("\n=== WEAK/CORE CONCEPT PROBES ===")
    for p in probes:
        hits = [n for n in names if p in n]
        print(f"  {p!r}: {hits[:5] if hits else 'MISSING'}")

    # Edge type breakdown
    et = Counter()
    for e in edges:
        t = e.get("type") or e.get("relation") or e.get("edge_type") or "?"
        et[str(t)] += 1
    print("\n=== EDGE TYPES ===")
    for t, c in et.most_common():
        print(f"  {t}: {c}")

    # Orphans: concepts with degree 0
    linked = set()
    for e in edges:
        for k in ("source", "from", "src", "target", "to", "dst"):
            if e.get(k):
                linked.add(str(e[k]).lower())
    orphan = 0
    for c in concepts:
        cid = str(c.get("id") or c.get("name") or c.get("concept_name") or "").lower()
        cname = str(c.get("name") or c.get("concept_name") or c.get("label") or "").lower()
        if cid not in linked and cname not in linked:
            orphan += 1
    print(f"\norphans (no edge link by id/name): {orphan}/{len(concepts)} ({100*orphan/max(len(concepts),1):.1f}%)")

    # Expected stats gate
    exp_path = ROOT / "pilot_corpus" / "expected_stats.json"
    if exp_path.exists():
        exp = json.loads(exp_path.read_text())
        print("\n=== EXPECTED STATS GATE ===")
        print(json.dumps(exp, indent=2)[:800])
        nc, ne = len(concepts), len(edges)
        ok = True
        if "min_concepts" in exp and nc < exp["min_concepts"]:
            print(f"FAIL concepts {nc} < min {exp['min_concepts']}"); ok = False
        if "max_concepts" in exp and nc > exp["max_concepts"]:
            print(f"WARN concepts {nc} > max {exp['max_concepts']}")
        if "min_edges" in exp and ne < exp["min_edges"]:
            print(f"FAIL edges {ne} < min {exp['min_edges']}"); ok = False
        print("GATE:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()

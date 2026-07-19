"""AIML domain-scope filter: drop non-AIML entities that leak in from papers'
worked examples (e.g. GraphRAG's entertainment-news dataset) and known
SLM-hallucinated concept names.

This runs inside cleanup_and_canonicalize so every rebuild / re-ingest stays
clean without manual scrubbing.
"""
from okf.cleanup_parts._base import re  # noqa: F401

# Concept names that must never become graph nodes (lowercase, canonical-ish).
# Two families:
#  1) example-dataset entities from paper illustrations (not AIML concepts)
#  2) hallucinated / mode-collapse names observed from aura-qwen v3
NON_AIML_NAME_DENYLIST = {
    # GraphRAG paper example-dataset entities (entertainment/news podcast corpus)
    "taylor swift", "travis kelce", "britney spears", "public figures",
    "news article", "podcast", "answer 1", "health", "policy", "influence",
    "directness", "diversity", "comprehensiveness",
    "verdant oasis plaza", "unity march", "harmony assembly",
    "jeopardy question", "jeopardy question generation",
    "community answers", "global answer",
    # aura-qwen hallucinations / corrupted names
    "maximum empowerment", "bart: batchable arma", "convex convex hull",
    "negative log-log", "positive log-lag", "negative log-lag",
    "positive log-log", "expanding exponential",
    "the theorem",
}

# Records whose own source passage is a paper's worked example about
# news/entertainment content are illustrations, not AIML material.
EXAMPLE_PASSAGE_RE = re.compile(
    r"(?i)entertainment\s+articles?|public\s+figures\s+who|"
    r"taylor\s+swift|travis\s+kelce|britney\s+spears|justin\s+bieber|"
    r"verdant\s+oasis\s+plaza|unity\s+march|harmony\s+assembly"
)


def drop_non_aiml_records(okf_results: list) -> tuple[list, dict]:
    """Drop denylisted names and records grounded only in example passages.

    Returns (kept_records, stats). A concept extracted from an entertainment
    example passage loses that record; if it has other, legitimate sources the
    concept still survives through them.
    """
    stats = {"denylisted": 0, "example_passage": 0}
    dropped_names = []
    kept = []
    for r in okf_results:
        name = (r.get("concept_name") or "").strip().lower()
        if name in NON_AIML_NAME_DENYLIST:
            stats["denylisted"] += 1
            dropped_names.append(r.get("concept_name"))
            continue
        passage = r.get("source_passage") or ""
        if passage and EXAMPLE_PASSAGE_RE.search(passage):
            stats["example_passage"] += 1
            dropped_names.append(r.get("concept_name"))
            continue
        kept.append(r)

    # Also strip denylisted names out of surviving records' edge references
    pruned_refs = 0
    for r in kept:
        for key in ("prerequisites", "unlocks"):
            vals = r.get(key) or []
            new_vals = [v for v in vals
                        if not (isinstance(v, str) and v.strip().lower() in NON_AIML_NAME_DENYLIST)]
            pruned_refs += len(vals) - len(new_vals)
            r[key] = new_vals
        rel = r.get("related_to") or []
        new_rel = [x for x in rel
                   if not (isinstance(x, dict)
                           and str(x.get("concept", "")).strip().lower() in NON_AIML_NAME_DENYLIST)]
        pruned_refs += len(rel) - len(new_rel)
        r["related_to"] = new_rel
    stats["pruned_refs"] = pruned_refs

    total = stats["denylisted"] + stats["example_passage"]
    if total:
        uniq = sorted({str(n) for n in dropped_names if n})
        print(f"  Domain-scope filter dropped {total} non-AIML records "
              f"({stats['denylisted']} denylisted, {stats['example_passage']} example-passage): "
              f"{', '.join(uniq[:10])}{'...' if len(uniq) > 10 else ''}")
        if pruned_refs:
            print(f"    (also pruned {pruned_refs} edge references to denylisted names)")
    return kept, stats



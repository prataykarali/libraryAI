from okf.cleanup_parts._base import *  # noqa: F403

def is_valid_concept_name(name: str) -> bool:
    """Reject obvious metadata, numeric artifacts and non-concept names."""
    if not name or len(name.strip()) < 3:
        return False
    name = name.strip()
    if len(name) > 60 or len(name.split()) > 5:
        return False
    if _NUMERIC_NAME_RE.match(name):
        return False
    if _JUNK_NAME_RE.search(name):
        return False
    if _FORMULA_OR_VALUE_NAME_RE.search(name):
        return False
    return True

def prune_invalid_references(okf_results: list) -> dict:
    """Remove junk/self references from prerequisites, unlocks and related_to."""
    stats = {
        "invalid_prerequisites": 0,
        "invalid_unlocks": 0,
        "invalid_related": 0,
        "self_references": 0,
    }

    for r in okf_results:
        name = r.get("concept_name", "")

        clean_prereqs = []
        seen_prereqs = set()
        for p in r.get("prerequisites", []):
            if not isinstance(p, str) or not p.strip():
                stats["invalid_prerequisites"] += 1
                continue
            p = canonicalize_name(p)
            if is_same_concept_reference(p, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(p):
                stats["invalid_prerequisites"] += 1
                continue
            key = p.lower()
            if key not in seen_prereqs:
                clean_prereqs.append(p)
                seen_prereqs.add(key)

        clean_unlocks = []
        seen_unlocks = set()
        for u in r.get("unlocks", []):
            if not isinstance(u, str) or not u.strip():
                stats["invalid_unlocks"] += 1
                continue
            u = canonicalize_name(u)
            if is_same_concept_reference(u, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(u):
                stats["invalid_unlocks"] += 1
                continue
            key = u.lower()
            if key not in seen_unlocks:
                clean_unlocks.append(u)
                seen_unlocks.add(key)

        clean_related = []
        seen_related = set()
        for rel in r.get("related_to", []):
            if not isinstance(rel, dict):
                stats["invalid_related"] += 1
                continue
            concept = canonicalize_name(str(rel.get("concept", "")).strip())
            relation = str(rel.get("relation", "uses")).strip().lower()
            if is_same_concept_reference(concept, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(concept):
                stats["invalid_related"] += 1
                continue
            if relation not in VALID_RELATIONS:
                relation = "uses"
            key = (concept.lower(), relation)
            if key not in seen_related:
                clean_related.append({"concept": concept, "relation": relation})
                seen_related.add(key)

        r["prerequisites"] = clean_prereqs
        r["unlocks"] = clean_unlocks
        r["related_to"] = clean_related

    return stats

def _content_words(text: str) -> set:
    """Lowercase content words (len>2, stopword-filtered) of a text."""
    return {
        w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }

def apply_grounding_filter(okf_results: list) -> tuple[list, dict]:
    """Grounding filter: a record must be anchored in its own source passage.

    A record is DROPPED when:
      - neither its exact concept_name nor any content word of the name
        appears in source_passage (case-insensitive), UNLESS its summary has
        >=30% content-word overlap with the passage (an abstractive concept
        genuinely described by the chunk); or
      - its summary is empty AND the exact name never appears in the passage
        (word-fragment matches alone can't justify a summary-less record).

    This catches SLM hallucinations and mode-collapse boilerplate (e.g.
    "Graph RAG" extracted "from" a math-textbook paragraph) while keeping
    valid abstractive concepts, which carry a passage-grounded summary.
    """
    stats = {
        "dropped_ungrounded": 0,
        "dropped_empty_summary": 0,
        "rescued_by_summary_overlap": 0,
        "kept": 0,
    }
    dropped_examples = {"ungrounded": [], "empty_summary": []}
    kept = []
    for r in okf_results:
        passage = (r.get("source_passage") or "").lower()
        name = (r.get("concept_name") or "").strip().lower()
        summary = (r.get("summary") or "").strip()
        if not passage or not name:
            stats["kept"] += 1
            kept.append(r)
            continue

        exact_hit = name in passage
        word_hit = exact_hit or any(w in passage for w in _content_words(name))

        if not word_hit:
            # Rescue only if the summary is demonstrably about this passage.
            summary_words = _content_words(summary)
            passage_words = _content_words(passage)
            overlap = (len(summary_words & passage_words) / len(summary_words)
                       if summary_words else 0.0)
            if overlap >= 0.30:
                stats["rescued_by_summary_overlap"] += 1
                stats["kept"] += 1
                kept.append(r)
            else:
                stats["dropped_ungrounded"] += 1
                dropped_examples["ungrounded"].append(r.get("concept_name"))
            continue

        if not summary and not exact_hit:
            stats["dropped_empty_summary"] += 1
            dropped_examples["empty_summary"].append(r.get("concept_name"))
            continue

        stats["kept"] += 1
        kept.append(r)

    total_dropped = stats["dropped_ungrounded"] + stats["dropped_empty_summary"]
    if total_dropped:
        print(f"  Grounding filter dropped {total_dropped} records:")
        print(f"    {stats['dropped_ungrounded']} ungrounded (name not in passage, "
              f"summary overlap <30%): {', '.join(dropped_examples['ungrounded'][:6])}"
              f"{'...' if len(dropped_examples['ungrounded']) > 6 else ''}")
        print(f"    {stats['dropped_empty_summary']} empty-summary without exact "
              f"name match: {', '.join(dropped_examples['empty_summary'][:6])}"
              f"{'...' if len(dropped_examples['empty_summary']) > 6 else ''}")
        if stats["rescued_by_summary_overlap"]:
            print(f"    ({stats['rescued_by_summary_overlap']} ungrounded-name records "
                  f"rescued by >=30% summary/passage overlap)")
    return kept, stats

def prune_unresolved_references(okf_results: list) -> dict:
    """Prune truly invalid references (empty, non-string, self-refs) but keep
    cross-document references that become placeholder nodes in the graph.
    """
    stats = {
        "prerequisites": 0,
        "unlocks": 0,
        "related": 0,
        "self_references": 0,
    }

    for r in okf_results:
        concept_name = r.get("concept_name", "").strip().lower()

        clean_prereqs = []
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p.strip() and is_valid_concept_name(p):
                if p.strip().lower() == concept_name:
                    stats["self_references"] += 1
                else:
                    clean_prereqs.append(p)
            else:
                stats["prerequisites"] += 1

        clean_unlocks = []
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u.strip() and is_valid_concept_name(u):
                if u.strip().lower() == concept_name:
                    stats["self_references"] += 1
                else:
                    clean_unlocks.append(u)
            else:
                stats["unlocks"] += 1

        clean_related = []
        for rel in r.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                concept = rel.get("concept", "")
                if isinstance(concept, str) and concept.strip() and is_valid_concept_name(concept):
                    if concept.strip().lower() == concept_name:
                        stats["self_references"] += 1
                    else:
                        clean_related.append(rel)
                else:
                    stats["related"] += 1
            else:
                stats["related"] += 1

        r["prerequisites"] = clean_prereqs
        r["unlocks"] = clean_unlocks
        r["related_to"] = clean_related

    return stats

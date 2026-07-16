"""Post-extraction cleanup: junk filters, grounding, dedup, cycle breaking."""

import re

from okf.canonicalize import (
    apply_canonicalization,
    build_canonical_map,
    canonicalize_name,
    is_same_concept_reference,
)
from okf.config import VALID_RELATIONS
from okf.util import _dedupe_dicts, _record_sources

# ---------------------------------------------------------------------------
# Direction / cycle helpers
# ---------------------------------------------------------------------------
# Optional domain-specific partial ordering. Used only to resolve direct A↔B
# prerequisite cycles when both concepts are in the dict. Leave empty for a
# fully generic pipeline; populate via a domain config file if desired.
_FOUNDATIONAL_PRIORITY = {}


_DIFFICULTY_RANK = {"foundational": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def break_global_cycles(okf_results: list) -> int:
    """Break all cycles in the learning progression graph (prerequisites and unlocks).

    Enforces a strict Directed Acyclic Graph (DAG) for learning paths.
    """
    import sys
    # Increase recursion limit just in case
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    index = {r.get("concept_name", "").lower(): r for r in okf_results}
    total_removed = 0

    for _ in range(1000):  # limit iterations to prevent infinite loop
        # Build learning edges list (src -> dst means src must be learned before dst)
        edges = []
        for r in okf_results:
            u = r.get("concept_name", "")
            u_lower = u.lower()
            if not u_lower:
                continue

            # Prerequisites: p -> u (p is learned before u)
            for p in r.get("prerequisites", []):
                p_lower = p.lower()
                if p_lower and p_lower != u_lower:
                    edges.append((p_lower, u_lower, "prereq", p, u))

            # Unlocks: u -> val (u unlocks val, so u is learned before val)
            for val in r.get("unlocks", []):
                val_lower = val.lower()
                if val_lower and val_lower != u_lower:
                    edges.append((u_lower, val_lower, "unlock", u, val))

        # Build adjacency list
        adj = {}
        edge_lookup = {}
        for src, dst, etype, orig_src, orig_dst in edges:
            adj.setdefault(src, set()).add(dst)
            edge_lookup[(src, dst)] = (etype, orig_src, orig_dst)

        # DFS cycle detection
        state = {}  # node -> 0: unvisited, 1: visiting, 2: visited
        parent = {}
        cycle_found = None

        def dfs(node):
            nonlocal cycle_found
            if cycle_found:
                return
            state[node] = 1
            for neighbor in adj.get(node, []):
                if state.get(neighbor, 0) == 1:
                    # Cycle detected! Reconstruct cycle path
                    cycle = []
                    curr = node
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent.get(curr)
                    cycle.append(neighbor)
                    cycle.reverse()
                    cycle.append(neighbor)
                    cycle_found = cycle
                    return
                elif state.get(neighbor, 0) == 0:
                    parent[neighbor] = node
                    dfs(neighbor)
                    if cycle_found:
                        return
            state[node] = 2

        all_nodes = set(index.keys()) | set(adj.keys())
        for node in all_nodes:
            if state.get(node, 0) == 0:
                dfs(node)
                if cycle_found:
                    break

        if not cycle_found:
            break

        # We have a cycle path like [A, B, C, A]
        cycle_pairs = []
        for i in range(len(cycle_found) - 1):
            cycle_pairs.append((cycle_found[i], cycle_found[i+1]))

        # Find weakest edge in the cycle:
        # Score = diff(dst) - diff(src). Lower score = weaker/more likely incorrect.
        weakest_edge = None
        min_score = None

        for src, dst in cycle_pairs:
            src_res = index.get(src)
            dst_res = index.get(dst)

            src_diff = _DIFFICULTY_RANK.get(src_res.get("difficulty", "intermediate") if src_res else "intermediate", 2)
            dst_diff = _DIFFICULTY_RANK.get(dst_res.get("difficulty", "intermediate") if dst_res else "intermediate", 2)

            score = dst_diff - src_diff

            # Tie-breaker: score, -len(src), -len(dst)
            edge_score = (score, -len(src), -len(dst))

            if min_score is None or edge_score < min_score:
                min_score = edge_score
                weakest_edge = (src, dst)

        if weakest_edge:
            src, dst = weakest_edge
            etype, orig_src, orig_dst = edge_lookup[(src, dst)]

            if etype == "prereq":
                dst_res = index.get(dst)
                if dst_res:
                    dst_res["prerequisites"] = [x for x in dst_res.get("prerequisites", []) if x.lower() != src]
            else:
                src_res = index.get(src)
                if src_res:
                    src_res["unlocks"] = [x for x in src_res.get("unlocks", []) if x.lower() != dst]

            total_removed += 1

    return total_removed


# Noise filters used in both extraction normalization and post-cleanup.
_JUNK_NAME_RE = re.compile(
    r"(?i)\b(authors?|contributors?|chairs?|funding|acknowledg|thank|grants?|projects?|"
    r"universit|institute|canada\s+cifar|research\s+chair|cifar\s+ai|nserc|"
    r"phd\s+program|fellowship|scholarship|discovery\s+grant|ai\s+chairs?|"
    r"computational\s+resources\s+provided|table\s+\d+|caption)\b|"
    r"best\s+model\s+without|underlined"
)
_NUMERIC_NAME_RE = re.compile(r"^\d[\d\s%\.x\-/]*$")
_FORMULA_OR_VALUE_NAME_RE = re.compile(
    r"(?i)([{}=∑∆ΔΦ]|\.{2,}|"
    r"\bvs\.?\b|\bcomparison\b|\b\d+%|\b\d+\s*of\s+tokens\b|"
    r"\breplacement\s+token\s*\(\d|"
    r"\b(system|hyperparameter)\s+dev\b|"
    r"\b(dev|test)\s+(f1|acc|accuracy|score)\b|"
    r"^test\s+scores?$|"
    r"\bstate-of-the-art\b.*\b(bleu|f1|accuracy|score)\b|"
    r"\bbatch\b|\bwithout\s+gold\s+access\b)"
)


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


# Minimal stopword list for grounding-overlap checks (content words only).
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "at", "from", "which", "can", "we",
    "you", "they", "their", "our", "not", "but", "if", "then", "than",
    "also", "such", "each", "other", "into", "over", "under", "between",
    "through", "when", "where", "how", "what", "all", "any", "some", "more",
    "most", "used", "using", "use", "based", "one", "two", "may", "will",
    "would", "should", "could", "has", "have", "had", "do", "does", "done",
    "there", "about", "both", "very", "given", "well", "only", "called",
}


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


def dedupe_identical_records(okf_results: list) -> tuple[list, int]:
    """Collapse mode-collapse duplicates: records sharing an identical
    (doc_id, concept_name, summary) triple are one observation repeated by the
    model, not independent evidence. Union their provenance and relations into
    a single record (same source-union semantics as merge_duplicate_results).
    """
    seen = {}
    deduped = []
    for r in okf_results:
        key = (
            r.get("doc_id", ""),
            (r.get("concept_name") or "").strip().lower(),
            (r.get("summary") or "").strip(),
        )
        if key in seen:
            existing = seen[key]
            existing["sources"] = _dedupe_dicts(
                _record_sources(existing) + _record_sources(r))
            existing["source_count"] = len(existing["sources"])
            for field in ("prerequisites", "unlocks", "tags"):
                existing[field] = list(dict.fromkeys(
                    existing.get(field, []) + r.get(field, [])))
            existing_rels = {(x.get("concept", ""), x.get("relation", ""))
                             for x in existing.get("related_to", []) if isinstance(x, dict)}
            for rel in r.get("related_to", []):
                if isinstance(rel, dict):
                    key_rel = (rel.get("concept", ""), rel.get("relation", ""))
                    if key_rel not in existing_rels:
                        existing.setdefault("related_to", []).append(rel)
                        existing_rels.add(key_rel)
            # Union per-relation provenance; the first record's entries win.
            merged_prov = dict(r.get("relation_provenance") or {})
            merged_prov.update(existing.get("relation_provenance") or {})
            if merged_prov:
                existing["relation_provenance"] = merged_prov
        else:
            seen[key] = r
            r["sources"] = _dedupe_dicts(_record_sources(r))
            r["source_count"] = len(r["sources"])
            deduped.append(r)
    return deduped, len(okf_results) - len(deduped)


def merge_duplicate_results(okf_results: list) -> tuple[list, int]:
    """Merge repeated concept records, keeping the richest fields."""
    seen = {}
    merged_results = []
    for r in okf_results:
        key = r.get("concept_name", "").lower().replace("-", " ").replace("_", " ").strip()
        if key in seen:
            existing = seen[key]
            if len(r.get("summary", "")) > len(existing.get("summary", "")):
                existing["summary"] = r["summary"]
            existing["prerequisites"] = list(set(
                existing.get("prerequisites", []) + r.get("prerequisites", [])))
            existing["unlocks"] = list(set(
                existing.get("unlocks", []) + r.get("unlocks", [])))
            existing["tags"] = list(set(
                existing.get("tags", []) + r.get("tags", [])))
            existing_rels = {(x.get("concept", ""), x.get("relation", ""))
                             for x in existing.get("related_to", []) if isinstance(x, dict)}
            for rel in r.get("related_to", []):
                if isinstance(rel, dict):
                    key_rel = (rel.get("concept", ""), rel.get("relation", ""))
                    if key_rel not in existing_rels:
                        existing.setdefault("related_to", []).append(rel)
                        existing_rels.add(key_rel)
            # Union ALL provenance from every duplicate so cross-document
            # evidence is never undercounted. Deduplicate the source records
            # but keep one per distinct (doc/chunk/page/section) origin.
            existing["sources"] = _dedupe_dicts(
                _record_sources(existing) + _record_sources(r))
            existing["source_count"] = len(existing["sources"])
            # Per-relation provenance must survive the merge: relations copied
            # from r keep pointing at r's asserting chunk, not existing's.
            r_prov = dict(r.get("relation_provenance") or {})
            # Relations r asserted without explicit provenance default to r's
            # own (doc_id, chunk_id) — record that before it is lost.
            r_src = f"{r.get('doc_id', '')}:{r.get('chunk_id', '')}"
            if r_src != ":":
                for p in r.get("prerequisites", []):
                    r_prov.setdefault(f"prereq:{str(p).lower()}", r_src)
                for u in r.get("unlocks", []):
                    r_prov.setdefault(f"unlock:{str(u).lower()}", r_src)
                for rel in r.get("related_to", []):
                    if isinstance(rel, dict) and rel.get("concept"):
                        r_prov.setdefault(
                            f"related:{str(rel['concept']).lower()}", r_src)
            merged_prov = r_prov
            merged_prov.update(existing.get("relation_provenance") or {})
            if merged_prov:
                existing["relation_provenance"] = merged_prov
        else:
            seen[key] = r
            # Normalize the surviving record so it carries an explicit,
            # deduplicated provenance list from the outset.
            r["sources"] = _dedupe_dicts(_record_sources(r))
            r["source_count"] = len(r["sources"])
            merged_results.append(r)
    return merged_results, len(okf_results) - len(merged_results)


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


def cleanup_and_canonicalize(okf_results: list) -> list:
    """Stages 2b+3: post-extraction cleanup and entity canonicalization.

    Idempotent — shared by finalize_and_build and the --relations-only mode
    (which must canonicalize BEFORE the relation pass so candidate names match
    the final concept inventory).
    """
    print("\n[2b] POST-EXTRACTION CLEANUP")
    print("-" * 50)
    pre_cleanup = len(okf_results)

    # 0.5. Remove non-concept metadata artifacts (authors, grants, chairs, etc.)
    okf_results = [r for r in okf_results if is_valid_concept_name(r.get("concept_name", ""))]
    junk_removed = pre_cleanup - len(okf_results)
    print(f"  Removed {junk_removed} junk/non-concept records")

    # 1. Remove reference/bibliography section extractions (noise)
    okf_results = [r for r in okf_results
                   if not r.get("section_title", "").lower().startswith("reference")]
    ref_removed = pre_cleanup - junk_removed - len(okf_results)
    print(f"  Removed {ref_removed} reference-section concepts")

    # 1.5. Grounding filter: a record must be anchored in its own source
    # passage (exact name or name content-word present), with a >=30%
    # summary/passage content-word-overlap rescue for abstractive concepts.
    # See apply_grounding_filter for the full rules and drop-count printout.
    okf_results, _ground_stats = apply_grounding_filter(okf_results)

    # 1.7. Anti-attractor dedup: mode-collapsed extractions repeat the exact
    # same (doc_id, concept_name, summary) across many chunks ('Vector' x407).
    # Collapse them to one record each, unioning provenance.
    okf_results, attractor_removed = dedupe_identical_records(okf_results)
    if attractor_removed > 0:
        print(f"  Collapsed {attractor_removed} identical "
              f"(doc, name, summary) mode-collapse duplicates")

    # 2. Remove self-loops (concept listing itself as prerequisite/unlock)
    ref_stats = prune_invalid_references(okf_results)
    print(
        "  Pruned references: "
        f"{ref_stats['invalid_prerequisites']} invalid prerequisites, "
        f"{ref_stats['invalid_unlocks']} invalid unlocks, "
        f"{ref_stats['invalid_related']} invalid related targets, "
        f"{ref_stats['self_references']} self references"
    )

    # 3. Merge duplicate concept names (keep richest version)
    okf_results, dupe_removed = merge_duplicate_results(okf_results)
    print(f"  Merged {dupe_removed} duplicate concept entries")

    # 4. Remove concepts with very short names (likely noise)
    pre_filter = len(okf_results)
    okf_results = [r for r in okf_results
                   if len(r.get("concept_name", "")) >= 3]
    noise_removed = pre_filter - len(okf_results)
    if noise_removed > 0:
        print(f"  Removed {noise_removed} too-short concept names")

    print(f"  Final: {len(okf_results)} clean concepts (was {pre_cleanup})")

    # -- Stage 3: Canonicalization --
    print(f"\n[3] STAGE 3: Entity Canonicalization")
    print("-" * 50)

    canon_map = build_canonical_map(okf_results)
    okf_results = apply_canonicalization(okf_results, canon_map)
    okf_results, post_canon_dupes = merge_duplicate_results(okf_results)

    # Count dedup stats
    raw_concepts = len(canon_map)
    unique_concepts = len(set(canon_map.values()))
    print(f"  Raw concept mentions: {raw_concepts}")
    print(f"  Canonical concepts: {unique_concepts}")
    print(f"  Aliases resolved: {raw_concepts - unique_concepts}")
    if post_canon_dupes > 0:
        print(f"  Merged {post_canon_dupes} duplicate concepts after canonicalization")

    # Post-canonicalization reference cleanup catches aliases that collapsed
    # into invalid names or near-self references.
    post_ref_stats = prune_invalid_references(okf_results)
    post_ref_removed = sum(post_ref_stats.values())
    if post_ref_removed > 0:
        print(
            "  Post-canonicalization reference prune: "
            f"{post_ref_stats['invalid_prerequisites']} invalid prerequisites, "
            f"{post_ref_stats['invalid_unlocks']} invalid unlocks, "
            f"{post_ref_stats['invalid_related']} invalid related targets, "
            f"{post_ref_stats['self_references']} self references"
        )

    unresolved_stats = prune_unresolved_references(okf_results)
    unresolved_removed = sum(unresolved_stats.values())
    if unresolved_removed > 0:
        print(
            "  Removed unresolved refs that would create placeholder nodes: "
            f"{unresolved_stats['prerequisites']} prerequisites, "
            f"{unresolved_stats['unlocks']} unlocks, "
            f"{unresolved_stats['related']} related targets"
        )

    cycles_broken = break_global_cycles(okf_results)
    if cycles_broken > 0:
        print(f"  Removed {cycles_broken} prerequisite/unlock cycle edges to enforce hierarchy DAG")

    # Fill empty summaries from the richest available source
    summary_by_name = {}
    for r in okf_results:
        name = r.get("concept_name", "")
        s = r.get("summary", "")
        if s and len(s) > len(summary_by_name.get(name, "")):
            summary_by_name[name] = s
    filled = 0
    for r in okf_results:
        name = r.get("concept_name", "")
        if not r.get("summary") and name in summary_by_name:
            r["summary"] = summary_by_name[name]
            filled += 1
    if filled > 0:
        print(f"  Filled {filled} empty summaries from sibling records")

    return okf_results

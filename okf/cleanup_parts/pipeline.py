from okf.cleanup_parts._base import *  # noqa: F403
from okf.cleanup_parts.grounding import is_valid_concept_name, apply_grounding_filter, prune_invalid_references, prune_unresolved_references, _content_words
from okf.cleanup_parts.dedupe import dedupe_identical_records, merge_duplicate_results, deduplicate_concepts
from okf.cleanup_parts.cycles import break_global_cycles
from okf.cleanup_parts.directionality import (
    is_bibliography_section,
    prune_inverted_prerequisites,
)
from okf.cleanup_parts.domain_scope import drop_non_aiml_records
from okf.canonicalize import apply_canonicalization

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
    # Broad match: "References", "Bibliography", "Works Cited", etc.
    okf_results = [
        r for r in okf_results
        if not is_bibliography_section(r.get("section_title", ""))
    ]
    ref_removed = pre_cleanup - junk_removed - len(okf_results)
    print(f"  Removed {ref_removed} reference/bibliography-section concepts")

    # 1.2. Domain-scope filter: drop non-AIML entities from papers' worked
    # examples (GraphRAG's entertainment dataset) and known hallucinated names.
    okf_results, _domain_stats = drop_non_aiml_records(okf_results)

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

    # Directionality: drop advanced→foundational inverted prereqs + known nonsense
    inverted_dropped = prune_inverted_prerequisites(okf_results)
    if inverted_dropped > 0:
        print(f"  Removed {inverted_dropped} inverted/blocked prerequisite edges")

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

def reject_weak_concepts(concepts: list, chunks: list, min_overlap: float = 0.3) -> list:
    """Reject weak concepts using grounding overlaps.

    Rejects concepts whose name/summary does not meet grounding thresholds
    against their source passage.
    """
    chunk_lookup = {}
    for c in chunks:
        doc_id = c.get("doc_id")
        chunk_id = c.get("chunk_id")
        if doc_id and chunk_id:
            chunk_lookup[(doc_id, chunk_id)] = c

    kept = []
    for r in concepts:
        passage = r.get("source_passage")
        if not passage:
            doc_id = r.get("doc_id")
            chunk_id = r.get("chunk_id")
            if doc_id and chunk_id and (doc_id, chunk_id) in chunk_lookup:
                chunk = chunk_lookup[(doc_id, chunk_id)]
                passage = chunk.get("text") or chunk.get("text_passage") or ""
        passage = (passage or "").lower()
        name = (r.get("concept_name") or "").strip().lower()
        summary = (r.get("summary") or "").strip()

        if not passage or not name:
            kept.append(r)
            continue

        exact_hit = name in passage
        word_hit = exact_hit or any(w in passage for w in _content_words(name))

        if not word_hit:
            summary_words = _content_words(summary)
            passage_words = _content_words(passage)
            overlap = (len(summary_words & passage_words) / len(summary_words)
                       if summary_words else 0.0)
            if overlap >= min_overlap:
                kept.append(r)
            continue

        if not summary and not exact_hit:
            continue

        kept.append(r)

    return kept

def filter_out_of_domain(concepts: list, domain_terms: list) -> list:
    """Filter out-of-domain concepts.

    Removes concepts whose names do not match any of the provided domain terms
    (case-insensitive substring match).
    """
    if not domain_terms:
        return concepts
    domain_terms_lower = [t.lower() for t in domain_terms if t]
    if not domain_terms_lower:
        return concepts

    kept = []
    for r in concepts:
        name = (r.get("concept_name") or "").lower()
        if any(dt in name for dt in domain_terms_lower):
            kept.append(r)
    return kept

def prune_orphans(db, threshold: float = 0.20) -> list:
    """Identifies, logs, and deletes orphan nodes in the database.

    Orphan nodes are Concept nodes that have no REQUIRES, UNLOCKS, or RELATED
    relationships to/from other Concept nodes.
    """
    import kuzu
    if hasattr(db, 'execute'):
        conn = db
    else:
        conn = kuzu.Connection(db)

    # 1. Total Concept Count
    res = conn.execute("MATCH (c:Concept) RETURN count(c)")
    total = res.get_next()[0] if res.has_next() else 0
    if total == 0:
        print("No concepts in database to prune.")
        return []

    # 2. Find orphans
    res = conn.execute("MATCH (c:Concept) WHERE NOT (c)-[:REQUIRES|UNLOCKS|RELATED]-(:Concept) RETURN c.id, c.name")
    orphans = []
    while res.has_next():
        row = res.get_next()
        orphans.append((row[0], row[1]))

    orphan_count = len(orphans)
    orphan_ratio = orphan_count / total

    print(f"Identified {orphan_count} orphan concept nodes out of {total} total concepts ({orphan_ratio:.2%}).")

    pruned = []
    if orphan_ratio >= threshold:
        # Detach and delete orphans
        for cid, name in orphans:
            print(f"  Deleting orphan concept: {name} (ID: {cid})")
            escaped_id = cid.replace("\\", "\\\\").replace("'", "\\'")
            conn.execute(f"MATCH (c:Concept {{id: '{escaped_id}'}}) DETACH DELETE c")
            pruned.append({"id": cid, "name": name})
    else:
        print(f"Orphan ratio ({orphan_ratio:.2%}) is below threshold ({threshold:.2%}). Orphans kept.")

    return pruned

def clean_pipeline(concepts: list, chunks: list, domain_terms: list = None) -> list:
    """Run the clean pipeline stages: weak concept rejection, deduplication, and domain filtering."""
    concepts = reject_weak_concepts(concepts, chunks)
    concepts = deduplicate_concepts(concepts)
    if domain_terms:
        concepts = filter_out_of_domain(concepts, domain_terms)
    return concepts

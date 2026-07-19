from okf.util import _record_sources, _dedupe_dicts, _source_record
from okf.cleanup_parts._base import *  # noqa: F403

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

def deduplicate_concepts(concepts: list, threshold: int = 85) -> list:
    """Deduplicate concepts using fuzzy string matching (thefuzz) with threshold."""
    if not concepts:
        return []
    try:
        from thefuzz import fuzz
    except ImportError:
        import sys
        print("Warning: thefuzz not available, falling back to exact matching", file=sys.stderr)
        fuzz = None

    canon_map = {}
    seen_names = []  # list of (lower_name, original_name)

    for c in concepts:
        name = c.get("concept_name", "").strip()
        if not name:
            continue
        name_lower = name.lower()

        matched_name = None
        if fuzz is not None:
            for sl, orig in seen_names:
                if fuzz.ratio(name_lower, sl) >= threshold:
                    matched_name = orig
                    break
        else:
            for sl, orig in seen_names:
                if name_lower == sl:
                    matched_name = orig
                    break

        if matched_name:
            canon_map[name] = matched_name
        else:
            seen_names.append((name_lower, name))
            canon_map[name] = name

    merged_by_name = {}
    for r in concepts:
        raw_name = r.get("concept_name", "")
        if not raw_name:
            continue
        canon_name = canon_map.get(raw_name, raw_name)

        canon_key = canon_name.lower()
        if canon_key not in merged_by_name:
            merged_by_name[canon_key] = {
                "concept_name": canon_name,
                "concept_type": r.get("concept_type", "definition"),
                "difficulty": r.get("difficulty", "intermediate"),
                "summary": r.get("summary", ""),
                "prerequisites": list(r.get("prerequisites", [])),
                "unlocks": list(r.get("unlocks", [])),
                "tags": list(r.get("tags", [])),
                "related_to": list(r.get("related_to", [])),
                "sources": list(_record_sources(r)),
                "relation_provenance": dict(r.get("relation_provenance") or {})
            }
            if r.get("source_passage"):
                merged_by_name[canon_key]["source_passage"] = r.get("source_passage")
            if r.get("doc_id"):
                merged_by_name[canon_key]["doc_id"] = r.get("doc_id")
            if r.get("chunk_id"):
                merged_by_name[canon_key]["chunk_id"] = r.get("chunk_id")
            if r.get("page_number") is not None:
                merged_by_name[canon_key]["page_number"] = r.get("page_number")
            if r.get("section_title"):
                merged_by_name[canon_key]["section_title"] = r.get("section_title")
        else:
            existing = merged_by_name[canon_key]
            if len(r.get("summary", "")) > len(existing.get("summary", "")):
                existing["summary"] = r["summary"]

            existing["prerequisites"] = list(dict.fromkeys(existing["prerequisites"] + r.get("prerequisites", [])))
            existing["unlocks"] = list(dict.fromkeys(existing["unlocks"] + r.get("unlocks", [])))
            existing["tags"] = list(dict.fromkeys(existing["tags"] + r.get("tags", [])))

            existing_rels = {(x.get("concept", ""), x.get("relation", ""))
                             for x in existing.get("related_to", []) if isinstance(x, dict)}
            for rel in r.get("related_to", []):
                if isinstance(rel, dict):
                    key_rel = (rel.get("concept", ""), rel.get("relation", ""))
                    if key_rel not in existing_rels:
                        existing["related_to"].append(rel)
                        existing_rels.add(key_rel)

            existing["sources"] = _dedupe_dicts(existing["sources"] + _record_sources(r))
            if "source_count" in existing or "source_count" in r:
                existing["source_count"] = len(existing["sources"])

            r_prov = dict(r.get("relation_provenance") or {})
            r_src = f"{r.get('doc_id', '')}:{r.get('chunk_id', '')}"
            if r_src != ":":
                for p in r.get("prerequisites", []):
                    r_prov.setdefault(f"prereq:{str(p).lower()}", r_src)
                for u in r.get("unlocks", []):
                    r_prov.setdefault(f"unlock:{str(u).lower()}", r_src)
                for rel in r.get("related_to", []):
                    if isinstance(rel, dict) and rel.get("concept"):
                        r_prov.setdefault(f"related:{str(rel['concept']).lower()}", r_src)
            existing["relation_provenance"].update(r_prov)

    results = list(merged_by_name.values())
    for r in results:
        r["prerequisites"] = [canon_map.get(p, p) for p in r["prerequisites"] if isinstance(p, str)]
        r["unlocks"] = [canon_map.get(u, u) for u in r["unlocks"] if isinstance(u, str)]
        new_related = []
        for rel in r["related_to"]:
            if isinstance(rel, dict) and rel.get("concept"):
                rel["concept"] = canon_map.get(rel["concept"], rel["concept"])
                new_related.append(rel)
        r["related_to"] = new_related

    return results

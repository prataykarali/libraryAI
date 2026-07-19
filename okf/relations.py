"""Second-pass relation extraction over cleaned+canonicalized records."""

import json
import sys
import time

from okf import extraction
from okf.cleanup import cleanup_and_canonicalize
from okf.config import BASE_DIR, MAX_CHARS_TO_SLM, RELATION_PROMPT
from okf.extraction import (
    _extract_json_payload,
    _generate_local,
    _normalize_related,
    _string_list,
    load_local_model,
)


def _passage_candidates(record: dict, all_names: list, cap: int = 8) -> list:
    """Canonical concept names (other than the record's own) that appear
    verbatim, case-insensitively, in the record's source passage."""
    passage_lower = (record.get("source_passage") or "").lower()
    own = (record.get("concept_name") or "").strip().lower()
    candidates = []
    for name in all_names:
        if name.lower() == own:
            continue
        if name.lower() in passage_lower:
            candidates.append(name)
            if len(candidates) >= cap:
                break
    return candidates


def extract_relations_for_record(record: dict, candidates: list) -> dict:
    """Prompt the local model for relations between a record's concept and the
    candidate concepts found in its passage. Returns accepted relations only
    (targets restricted to the candidate list); defensive parse like
    extract_okf_v15. Empty dict-of-empty-lists on any failure."""
    empty = {"prerequisites": [], "unlocks": [], "related_to": []}
    passage = record.get("source_passage") or ""
    if len(passage) > MAX_CHARS_TO_SLM:
        passage = passage[:MAX_CHARS_TO_SLM]
    prompt = RELATION_PROMPT.format(
        passage=passage,
        concept=record.get("concept_name", ""),
        candidates=json.dumps(candidates),
    )
    try:
        raw = _generate_local(prompt)
        cleaned = _extract_json_payload(raw)
        data = json.loads(cleaned)
    except Exception:
        return empty
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return empty

    own_lower = (record.get("concept_name") or "").strip().lower()
    cand_lookup = {c.lower(): c for c in candidates}

    def _accept_names(values):
        accepted = []
        for v in _string_list(values):
            key = v.strip().lower()
            if key != own_lower and key in cand_lookup:
                accepted.append(cand_lookup[key])
        return list(dict.fromkeys(accepted))

    accepted_related = []
    seen_rel = set()
    for rel in _normalize_related(data.get("related_to")):
        key = rel["concept"].strip().lower()
        if key == own_lower or key not in cand_lookup:
            continue
        rel["concept"] = cand_lookup[key]
        rel_key = (key, rel["relation"])
        if rel_key not in seen_rel:
            accepted_related.append(rel)
            seen_rel.add(rel_key)

    return {
        "prerequisites": _accept_names(data.get("prerequisites")),
        "unlocks": _accept_names(data.get("unlocks")),
        "related_to": accepted_related,
    }


def relation_pass(okf_results: list) -> dict:
    """Second-pass relation extraction over cleaned+canonicalized records.

    Mutates records in place: accepted relations are unioned onto the
    asserting record's prerequisites/unlocks/related_to, so edge provenance is
    that record's (doc_id, chunk_id) by construction. Targets not in the
    passage's candidate list are rejected before they can mint placeholders.
    """
    if extraction.LOCAL_MODEL is None:
        load_local_model()
    if not extraction.LOCAL_MODE or extraction.LOCAL_MODEL is None:
        print("ERROR: local model unavailable — relation pass never falls back to Ollama.")
        return {"processed": 0, "skipped": 0, "records_with_new_relations": 0,
                "relations_added": 0}

    all_names = sorted(
        {r.get("concept_name", "").strip() for r in okf_results
         if r.get("concept_name", "").strip()},
        key=len, reverse=True)  # longest first so 'Gradient Descent' beats 'Gradient'

    eligible = []
    skipped = 0
    for r in okf_results:
        if len(r.get("source_passage") or "") <= 200:
            skipped += 1
            continue
        candidates = _passage_candidates(r, all_names)
        if not candidates:
            skipped += 1
            continue
        eligible.append((r, candidates))

    print(f"\n[RELATION PASS] {len(eligible)} records with candidates "
          f"({skipped} skipped: trivial passage or no co-occurring concepts)")

    stats = {"processed": 0, "skipped": skipped,
             "records_with_new_relations": 0, "relations_added": 0}
    for i, (r, candidates) in enumerate(eligible):
        name = r.get("concept_name", "")
        print(f"  [{i+1}/{len(eligible)}] {name[:40]:40s} "
              f"({len(candidates)} candidates)", end="")
        sys.stdout.flush()
        start_time = time.time()
        rels = extract_relations_for_record(r, candidates)
        elapsed = time.time() - start_time
        stats["processed"] += 1

        added = 0
        for p in rels["prerequisites"]:
            if p not in r.setdefault("prerequisites", []):
                r["prerequisites"].append(p)
                added += 1
        for u in rels["unlocks"]:
            if u not in r.setdefault("unlocks", []):
                r["unlocks"].append(u)
                added += 1
        existing_rel = {(x.get("concept", "").lower(), x.get("relation", ""))
                        for x in r.get("related_to", []) if isinstance(x, dict)}
        for rel in rels["related_to"]:
            key = (rel["concept"].lower(), rel["relation"])
            if key not in existing_rel:
                r.setdefault("related_to", []).append(rel)
                existing_rel.add(key)
                added += 1

        if added:
            stats["records_with_new_relations"] += 1
            stats["relations_added"] += added
            summary = "; ".join(
                [f"req:{p}" for p in rels["prerequisites"]] +
                [f"unl:{u}" for u in rels["unlocks"]] +
                [f"{x['relation']}:{x['concept']}" for x in rels["related_to"]])
            print(f" -> +{added} [{summary[:60]}] ({elapsed:.1f}s)")
        else:
            print(f" -> 0 ({elapsed:.1f}s)")

    print(f"\n  Relation pass: {stats['relations_added']} relations added to "
          f"{stats['records_with_new_relations']}/{stats['processed']} records")
    return stats


def run_relations_only():
    """--relations-only mode: load saved results, clean+canonicalize, run the
    second-pass relation extraction, save, and rebuild the graph."""
    # Deferred import: pipeline imports this module, so importing it at the
    # top level would be circular.
    from okf.pipeline import finalize_and_build

    saved_file = BASE_DIR / "okf_results.json"
    if not saved_file.exists():
        print("ERROR: okf_results.json not found — run extraction first.")
        return
    with open(saved_file, "r", encoding="utf-8") as f:
        okf_results = json.load(f)
    print(f"Loaded {len(okf_results)} records from okf_results.json")

    # Cleanup/canonicalize FIRST so relation candidates are canonical names
    # from the final inventory (relation_pass runs after cleanup by design).
    okf_results = cleanup_and_canonicalize(okf_results)

    relation_pass(okf_results)

    chunk_count = len({(r.get("doc_id", ""), r.get("chunk_id", ""))
                       for r in okf_results if r.get("chunk_id")})
    return finalize_and_build(okf_results, chunk_count, chunk_count)


def validate_relation(relation, chunks: list, alias_index: dict = None) -> bool:
    """Check if both concepts of the relation (or their aliases) co-occur in at least one chunk."""
    if isinstance(relation, dict):
        concept_a = relation.get("source") or relation.get("concept_a") or relation.get("from") or relation.get("from_name") or relation.get("from_id")
        concept_b = relation.get("target") or relation.get("concept_b") or relation.get("to") or relation.get("to_name") or relation.get("to_id") or relation.get("concept")
    elif isinstance(relation, (tuple, list)) and len(relation) >= 2:
        concept_a = relation[0]
        concept_b = relation[1]
    else:
        return False

    if not concept_a or not concept_b:
        return False

    from okf.alias_index import generate_aliases_for_name, resolve_concept_name
    
    a_canon = resolve_concept_name(str(concept_a), alias_index) if alias_index else str(concept_a)
    b_canon = resolve_concept_name(str(concept_b), alias_index) if alias_index else str(concept_b)
    
    aliases_a = generate_aliases_for_name(a_canon)
    aliases_b = generate_aliases_for_name(b_canon)

    for chunk in chunks:
        text = (chunk.get("text") or chunk.get("text_passage") or "").lower()
        has_a = any(al in text for al in aliases_a)
        has_b = any(al in text for al in aliases_b)
        if has_a and has_b:
            return True
    return False


def filter_relations(relations, chunks: list, concepts: list, alias_index: dict = None):
    """Filter a list/dict of relations against chunks and concepts using alias-aware matching."""
    from okf.alias_index import build_alias_index, resolve_concept_name
    
    if alias_index is None:
        raw_names = []
        for c in concepts:
            if isinstance(c, dict):
                name = c.get("concept_name") or c.get("name")
                if name:
                    raw_names.append(str(name))
            elif isinstance(c, str):
                raw_names.append(c)
        alias_index = build_alias_index(raw_names)

    concept_canon_names = set()
    for c in concepts:
        if isinstance(c, dict):
            name = c.get("concept_name") or c.get("name")
            if name:
                concept_canon_names.add(resolve_concept_name(str(name), alias_index).lower().strip())
        elif isinstance(c, str):
            concept_canon_names.add(resolve_concept_name(c, alias_index).lower().strip())

    filtered = []
    is_dict = isinstance(relations, dict)
    items = relations.values() if is_dict else relations

    for rel in items:
        if isinstance(rel, dict):
            concept_a = rel.get("source") or rel.get("concept_a") or rel.get("from") or rel.get("from_name") or rel.get("from_id")
            concept_b = rel.get("target") or rel.get("concept_b") or rel.get("to") or rel.get("to_name") or rel.get("to_id") or rel.get("concept")
        elif isinstance(rel, (tuple, list)) and len(rel) >= 2:
            concept_a = rel[0]
            concept_b = rel[1]
        else:
            continue

        if not concept_a or not concept_b:
            continue

        a_canon = resolve_concept_name(str(concept_a), alias_index).lower().strip()
        b_canon = resolve_concept_name(str(concept_b), alias_index).lower().strip()

        if a_canon in concept_canon_names and b_canon in concept_canon_names:
            if validate_relation(rel, chunks, alias_index):
                filtered.append(rel)

    if is_dict:
        filtered_dict = {}
        for k, rel in relations.items():
            if rel in filtered:
                filtered_dict[k] = rel
        return filtered_dict
    return filtered


def infer_prerequisite_direction(concept_a, concept_b, chunks: list) -> tuple:
    """Infer prerequisite direction between concept_a and concept_b.

    Uses order of appearance in chunks (first occurrence or co-occurrence position).
    Falls back to difficulty ranks (foundational < intermediate < advanced < expert).
    """
    diff_rank = {"foundational": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    diff_a = "intermediate"
    diff_b = "intermediate"

    if isinstance(concept_a, dict):
        diff_a = concept_a.get("difficulty", "intermediate")
        name_a = concept_a.get("concept_name") or concept_a.get("name") or ""
    else:
        name_a = str(concept_a)

    if isinstance(concept_b, dict):
        diff_b = concept_b.get("difficulty", "intermediate")
        name_b = concept_b.get("concept_name") or concept_b.get("name") or ""
    else:
        name_b = str(concept_b)

    a_lower = name_a.strip().lower()
    b_lower = name_b.strip().lower()

    co_occur_score = 0
    first_chunk_a = None
    first_chunk_b = None

    for idx, chunk in enumerate(chunks):
        text = (chunk.get("text") or chunk.get("text_passage") or "").lower()
        has_a = a_lower in text
        has_b = b_lower in text

        if has_a and first_chunk_a is None:
            first_chunk_a = idx
        if has_b and first_chunk_b is None:
            first_chunk_b = idx

        if has_a and has_b:
            try:
                idx_a = text.index(a_lower)
                idx_b = text.index(b_lower)
                if idx_a < idx_b:
                    co_occur_score += 1
                elif idx_b < idx_a:
                    co_occur_score -= 1
            except ValueError:
                pass

    if co_occur_score > 0:
        return (name_a, name_b)
    elif co_occur_score < 0:
        return (name_b, name_a)

    if first_chunk_a is not None and first_chunk_b is not None and first_chunk_a != first_chunk_b:
        if first_chunk_a < first_chunk_b:
            return (name_a, name_b)
        else:
            return (name_b, name_a)

    # Fallback to difficulty rank
    rank_a = diff_rank.get(str(diff_a).lower(), 2)
    rank_b = diff_rank.get(str(diff_b).lower(), 2)

    if rank_a < rank_b:
        return (name_a, name_b)
    elif rank_b < rank_a:
        return (name_b, name_a)

    # Final fallback: alphabetical
    if name_a < name_b:
        return (name_a, name_b)
    else:
        return (name_b, name_a)


def build_validated_edges(concepts: list, chunks: list, model_name: str = None) -> list:
    """Build and validate relationships from concepts against chunks using alias-aware matching."""
    from okf.alias_index import build_alias_index, resolve_concept_name

    concept_raw_names = []
    concept_diffs = {}
    for c in concepts:
        if isinstance(c, dict):
            name = c.get("concept_name") or c.get("name")
            if name:
                name_str = str(name).strip()
                concept_raw_names.append(name_str)
                concept_diffs[name_str.lower()] = c.get("difficulty", "intermediate")
        elif isinstance(c, str):
            concept_raw_names.append(c.strip())
            concept_diffs[c.strip().lower()] = "intermediate"

    alias_index = build_alias_index(concept_raw_names)
    canonical_concept_names = {
        resolve_concept_name(name, alias_index).lower().strip() for name in concept_raw_names
    }

    edges = []
    seen_edges = set()  # (from_lower, to_lower, edge_type)

    def add_edge(from_name, to_name, edge_type, relation, source):
        from_canon = resolve_concept_name(from_name, alias_index)
        to_canon = resolve_concept_name(to_name, alias_index)
        
        fl = from_canon.lower().strip()
        tl = to_canon.lower().strip()
        if fl == tl:
            return
        key = (fl, tl, edge_type)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({
            "from_name": from_canon,
            "to_name": to_canon,
            "relation": relation,
            "edge_type": edge_type,
            "source": source
        })

    for c in concepts:
        if not isinstance(c, dict):
            continue
        raw_name = c.get("concept_name", "").strip()
        name = resolve_concept_name(raw_name, alias_index)
        if not name or name.lower().strip() not in canonical_concept_names:
            continue

        doc_id = c.get("doc_id", "")
        chunk_id = c.get("chunk_id", "")
        default_source = f"{doc_id}:{chunk_id}"
        rel_prov = c.get("relation_provenance") or {}

        # Prerequisites
        for p in c.get("prerequisites", []):
            if not isinstance(p, str) or not p.strip():
                continue
            p = p.strip()
            p_canon = resolve_concept_name(p, alias_index)
            if p_canon.lower().strip() not in canonical_concept_names:
                continue

            rel_dict = {"source": name, "target": p_canon}
            if not validate_relation(rel_dict, chunks, alias_index):
                continue

            diff_a = concept_diffs.get(name.lower(), "intermediate")
            diff_b = concept_diffs.get(p_canon.lower(), "intermediate")
            obj_a = {"concept_name": name, "difficulty": diff_a}
            obj_b = {"concept_name": p_canon, "difficulty": diff_b}
            dir_first, dir_second = infer_prerequisite_direction(obj_a, obj_b, chunks)

            src = rel_prov.get(f"prereq:{p.lower()}", default_source)
            if dir_first.lower() == p_canon.lower():
                add_edge(name, p_canon, "REQUIRES", "requires", src)
            else:
                add_edge(p_canon, name, "REQUIRES", "requires", src)

        # Unlocks
        for u in c.get("unlocks", []):
            if not isinstance(u, str) or not u.strip():
                continue
            u = u.strip()
            u_canon = resolve_concept_name(u, alias_index)
            if u_canon.lower().strip() not in canonical_concept_names:
                continue

            rel_dict = {"source": name, "target": u_canon}
            if not validate_relation(rel_dict, chunks, alias_index):
                continue

            diff_a = concept_diffs.get(name.lower(), "intermediate")
            diff_b = concept_diffs.get(u_canon.lower(), "intermediate")
            obj_a = {"concept_name": name, "difficulty": diff_a}
            obj_b = {"concept_name": u_canon, "difficulty": diff_b}
            dir_first, dir_second = infer_prerequisite_direction(obj_a, obj_b, chunks)

            src = rel_prov.get(f"unlock:{u.lower()}", default_source)
            if dir_first.lower() == name.lower():
                add_edge(name, u_canon, "UNLOCKS", "enables", src)
            else:
                add_edge(u_canon, name, "UNLOCKS", "enables", src)

        # Related
        for rel in c.get("related_to", []):
            if not isinstance(rel, dict):
                continue
            target = rel.get("concept", "").strip()
            if not target:
                continue
            target_canon = resolve_concept_name(target, alias_index)
            if target_canon.lower().strip() not in canonical_concept_names:
                continue

            rel_dict = {"source": name, "target": target_canon}
            if not validate_relation(rel_dict, chunks, alias_index):
                continue

            rel_type = rel.get("relation", "related")
            src = rel_prov.get(f"related:{target.lower()}", default_source)

            if name.lower() < target_canon.lower():
                add_edge(name, target_canon, "RELATED", rel_type, src)
            else:
                add_edge(target_canon, name, "RELATED", rel_type, src)

    return edges

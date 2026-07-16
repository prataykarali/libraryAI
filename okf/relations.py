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

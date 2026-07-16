#!/usr/bin/env python
"""Import relation fields from okf_results.json.bak2 (old-model extractions)
into the current okf_results.json (aura-qwen extractions).

The aura-qwen fine-tune almost never emits prerequisites/unlocks/related_to
(13/1024 records) and cannot perform relation selection even when prompted
with constrained candidates (verified 2026-07-16). The old model's records
carry 651 relation mentions averaging 4.23/record. This script copies those
relations onto matching current records, subject to grounding rules:

- A bak2 record contributes only if its canonicalized concept_name matches a
  canonical concept in the CURRENT inventory.
- Each relation target is kept only if its canonical form is also in the
  current inventory (no placeholder nodes can be minted downstream — the
  edge-builder in okf_pipeline.py additionally gates on the inventory).
- Relations attach to the current record with the same (doc_id, chunk_id)
  when one exists, else the doc's first record for that concept, else the
  concept's first record anywhere. relation_provenance records the bak2
  record's doc_id:chunk_id so edges cite the asserting chunk.
"""
import json
from collections import Counter

import okf

VALID_RELATIONS = okf.VALID_RELATIONS

with open("okf_results.json", encoding="utf-8") as f:
    current = json.load(f)
with open("okf_results.json.bak2", encoding="utf-8") as f:
    bak2 = json.load(f)

# Canonical inventory of the CURRENT results (post-canonicalization names may
# differ from raw; use the pipeline's canonicalizer for both sides).
canon_map = okf.build_canonical_map(current)


def canon(name: str) -> str:
    return canon_map.get(name, okf.canonicalize_name(name or ""))


inventory = {canon(r.get("concept_name", "")).lower()
             for r in current if r.get("concept_name")}

# Lookups into current records
by_doc_chunk_name = {}
by_doc_name = {}
by_name = {}
for r in current:
    key_name = canon(r.get("concept_name", "")).lower()
    by_doc_chunk_name.setdefault(
        (r.get("doc_id"), r.get("chunk_id"), key_name), r)
    by_doc_name.setdefault((r.get("doc_id"), key_name), r)
    by_name.setdefault(key_name, r)

stats = Counter()
for b in bak2:
    rels_present = (b.get("prerequisites") or b.get("unlocks")
                    or b.get("related_to"))
    if not rels_present:
        continue
    stats["bak2_records_with_relations"] += 1

    cname = canon(b.get("concept_name", "")).lower()
    if cname not in inventory:
        stats["skipped_concept_not_in_inventory"] += 1
        continue

    target = (by_doc_chunk_name.get((b.get("doc_id"), b.get("chunk_id"), cname))
              or by_doc_name.get((b.get("doc_id"), cname))
              or by_name.get(cname))
    if target is None:
        stats["skipped_no_target_record"] += 1
        continue

    prov = f"{b.get('doc_id', '')}:{b.get('chunk_id', '')}"
    rel_prov = target.setdefault("relation_provenance", {})

    for p in b.get("prerequisites") or []:
        if not isinstance(p, str) or not p.strip():
            continue
        cp = canon(p)
        if cp.lower() == cname:
            stats["skipped_self_loop"] += 1
            continue
        if cp.lower() not in inventory:
            stats["skipped_target_not_in_inventory"] += 1
            continue
        if cp not in target.setdefault("prerequisites", []):
            target["prerequisites"].append(cp)
            rel_prov[f"prereq:{cp.lower()}"] = prov
            stats["imported_prerequisites"] += 1

    for u in b.get("unlocks") or []:
        if not isinstance(u, str) or not u.strip():
            continue
        cu = canon(u)
        if cu.lower() == cname:
            stats["skipped_self_loop"] += 1
            continue
        if cu.lower() not in inventory:
            stats["skipped_target_not_in_inventory"] += 1
            continue
        if cu not in target.setdefault("unlocks", []):
            target["unlocks"].append(cu)
            rel_prov[f"unlock:{cu.lower()}"] = prov
            stats["imported_unlocks"] += 1

    existing_rel = {(x.get("concept", "").lower(), x.get("relation", ""))
                    for x in target.get("related_to", []) if isinstance(x, dict)}
    for rel in b.get("related_to") or []:
        if not isinstance(rel, dict) or not rel.get("concept"):
            continue
        cr = canon(rel["concept"])
        rtype = rel.get("relation", "")
        if rtype not in VALID_RELATIONS:
            stats["skipped_invalid_relation_type"] += 1
            continue
        if cr.lower() == cname:
            stats["skipped_self_loop"] += 1
            continue
        if cr.lower() not in inventory:
            stats["skipped_target_not_in_inventory"] += 1
            continue
        key = (cr.lower(), rtype)
        if key not in existing_rel:
            target.setdefault("related_to", []).append(
                {"concept": cr, "relation": rtype})
            rel_prov[f"related:{cr.lower()}"] = prov
            existing_rel.add(key)
            stats["imported_related"] += 1

imported = (stats["imported_prerequisites"] + stats["imported_unlocks"]
            + stats["imported_related"])
print("bak2 relation import summary:")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")
print(f"  TOTAL imported relation mentions: {imported}")

if imported:
    with open("okf_results.json", "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    print("Saved okf_results.json")
else:
    print("Nothing imported — okf_results.json left untouched.")

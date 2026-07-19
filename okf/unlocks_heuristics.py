"""unlocks_heuristics.py — Heuristics to infer UNLOCKS edges from text and relations."""
import re
import collections
from okf.alias_index import build_alias_index, resolve_concept_name, generate_aliases_for_name

def add_heuristic_unlocks(okf_results: list, chunks: list) -> list:
    """Scan relations and text passages to promote/infer UNLOCKS edges.
    
    Mutates okf_results in-place.
    UNLOCKS target is > 20 edges (currently 6).
    1. Promotes RELATED edges with unlock-like labels (enables, unlocks, leads_to, part_of, etc.) to UNLOCKS.
    2. Scans text passages containing concept pairs for unlock triggers (e.g. 'A enables B', 'B builds on A') and adds them.
    """
    concept_raw_names = [r.get("concept_name") for r in okf_results if r.get("concept_name")]
    alias_index = build_alias_index(concept_raw_names)
    
    # Map from canonical lowercase -> original canonical name
    canon_lookup = {}
    concept_by_canon = {}
    for name in concept_raw_names:
        canon = resolve_concept_name(name, alias_index)
        canon_lookup[canon.lower().strip()] = canon
        # Find the first record for this canon name to mutate
        for r in okf_results:
            if r.get("concept_name") == name:
                concept_by_canon[canon.lower().strip()] = r
                break

    # 1. Promote related_to edges with unlocks/enables semantics
    promoted_count = 0
    unlock_labels = {"enables", "unlocks", "leads_to", "leads to", "allows", "prerequisite_for", "pre-requisite", "extends"}
    
    for r in okf_results:
        name = r.get("concept_name")
        if not name:
            continue
        c_canon = resolve_concept_name(name, alias_index)
        
        kept_related = []
        for rel in r.get("related_to", []):
            if not isinstance(rel, dict) or not rel.get("concept"):
                kept_related.append(rel)
                continue
                
            rel_concept = rel["concept"]
            rel_type = rel.get("relation", "").lower().strip()
            
            # If relation type indicates forward unlocking
            if rel_type in unlock_labels:
                target_canon = resolve_concept_name(rel_concept, alias_index)
                unlocks_list = r.setdefault("unlocks", [])
                if target_canon not in unlocks_list:
                    unlocks_list.append(target_canon)
                    # Copy provenance if exists
                    prov = r.setdefault("relation_provenance", {})
                    src_prov = prov.get(f"related:{rel_concept.lower()}")
                    if src_prov:
                        prov[f"unlock:{target_canon.lower()}"] = src_prov
                    promoted_count += 1
            else:
                kept_related.append(rel)
                
        r["related_to"] = kept_related

    print(f"  [UNLOCKS HEURISTICS] Promoted {promoted_count} RELATED edges to UNLOCKS.")

    # 2. Linguistic text scan on co-occurring concepts (highly optimized)
    concept_aliases = {}
    for canon_low, canon_name in canon_lookup.items():
        concept_aliases[canon_low] = [al.lower() for al in generate_aliases_for_name(canon_name)]

    inferred_count = 0
    # Scan chunks for co-occurrences and check trigger patterns
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        if not doc_id or not chunk_id:
            continue
            
        text = (chunk.get("text") or chunk.get("text_passage") or "").lower()
        
        # Fast check: does the chunk contain at least two concepts?
        active_concepts = []
        for canon_low, aliases in concept_aliases.items():
            if any(al in text for al in aliases):
                active_concepts.append(canon_low)
                
        if len(active_concepts) < 2:
            continue
            
        sentences = re.split(r'[.!?\n]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
                
            # Find which concepts are in this sentence
            present_concepts = []
            for canon_low in active_concepts:
                for alias in concept_aliases[canon_low]:
                    if alias in sentence:
                        if re.search(r'\b' + re.escape(alias) + r'\b', sentence):
                            present_concepts.append((canon_low, alias))
                            break
            
            if len(present_concepts) < 2:
                continue
                
            # Pairwise analysis within the sentence
            for i in range(len(present_concepts)):
                for j in range(len(present_concepts)):
                    if i == j:
                        continue
                    (c1_low, alias1) = present_concepts[i]
                    (c2_low, alias2) = present_concepts[j]
                    
                    # Try to locate order in sentence
                    idx1 = sentence.find(alias1)
                    idx2 = sentence.find(alias2)
                    if idx1 == -1 or idx2 == -1 or idx1 >= idx2:
                        continue
                        
                    # Text segment between concept 1 and concept 2
                    middle = sentence[idx1 + len(alias1):idx2].strip()
                    
                    # Heuristics:
                    # 1. "c1 enables/unlocks/leads to/allows c2" -> c1 UNLOCKS c2
                    if re.search(r'\b(enables|unlocks|leads to|allows|is a prerequisite for|key to)\b', middle):
                        rec1 = concept_by_canon.get(c1_low)
                        c2_name = canon_lookup.get(c2_low)
                        if rec1 is not None and c2_name:
                            unlocks_list = rec1.setdefault("unlocks", [])
                            if c2_name not in unlocks_list:
                                unlocks_list.append(c2_name)
                                prov = rec1.setdefault("relation_provenance", {})
                                prov[f"unlock:{c2_name.lower()}"] = f"{doc_id}:{chunk_id}"
                                inferred_count += 1
                                
                    # 2. "c1 requires/builds on/depends on/is based on c2" -> c2 UNLOCKS c1
                    elif re.search(r'\b(requires|builds on|depends on|is based on|needs|uses)\b', middle):
                        rec2 = concept_by_canon.get(c2_low)
                        c1_name = canon_lookup.get(c1_low)
                        if rec2 is not None and c1_name:
                            unlocks_list = rec2.setdefault("unlocks", [])
                            if c1_name not in unlocks_list:
                                unlocks_list.append(c1_name)
                                prov = rec2.setdefault("relation_provenance", {})
                                prov[f"unlock:{c1_name.lower()}"] = f"{doc_id}:{chunk_id}"
                                inferred_count += 1

    print(f"  [UNLOCKS HEURISTICS] Inferred {inferred_count} forward UNLOCKS edges from sentence context.")
    return okf_results

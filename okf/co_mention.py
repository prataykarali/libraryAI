"""co_mention.py — Build RELATED edges based on concept co-occurrences in chunks."""
import collections
from okf.alias_index import build_alias_index, resolve_concept_name, generate_aliases_for_name

def build_co_mention_edges(okf_results: list, chunks: list, max_co_mentions_per_concept: int = 5) -> list:
    """Scan chunks for concept co-occurrences and add RELATED edges of type 'co_mention' to okf_results.
    
    Mutates okf_results in-place by updating 'related_to' lists and 'relation_provenance' maps.
    Only creates co-mention edges between concepts that do not already have a REQUIRES, UNLOCKS, or RELATED edge.
    Prioritizes cross-document bridges (concepts primarily originating from different documents).
    """
    # 1. Build alias index from existing concepts
    concept_raw_names = []
    concept_docs = collections.defaultdict(set)
    concept_by_canon = {}
    
    for r in okf_results:
        name = r.get("concept_name")
        if name:
            name_str = str(name).strip()
            concept_raw_names.append(name_str)
            doc_id = r.get("doc_id")
            if doc_id:
                concept_docs[name_str.lower()].add(doc_id)
            concept_by_canon[name_str.lower()] = r

    alias_index = build_alias_index(concept_raw_names)
    
    # Map from canonical lowercase -> original canonical name
    canon_lookup = {}
    for name in concept_raw_names:
        canon = resolve_concept_name(name, alias_index)
        canon_lookup[canon.lower().strip()] = canon

    # 2. Map chunks to the concepts they mention
    chunk_mentions = collections.defaultdict(set)
    chunk_docs = {}
    
    # Pre-generate alias sets for all canonical concepts for fast matching
    concept_aliases = {}
    for canon_low, canon_name in canon_lookup.items():
        concept_aliases[canon_low] = [al.lower() for al in generate_aliases_for_name(canon_name)]

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        if not doc_id or not chunk_id:
            continue
            
        chunk_key = (doc_id, chunk_id)
        chunk_docs[chunk_key] = doc_id
        text = (chunk.get("text") or chunk.get("text_passage") or "").lower()
        
        # Fast alias scan
        for canon_low, aliases in concept_aliases.items():
            if any(al in text for al in aliases):
                chunk_mentions[chunk_key].add(canon_low)

    # 3. Track existing relationship pairs to avoid duplicates
    existing_pairs = set()
    for r in okf_results:
        name = r.get("concept_name")
        if not name:
            continue
        c_canon = resolve_concept_name(name, alias_index).lower().strip()
        
        # Prerequisites
        for p in r.get("prerequisites", []):
            p_canon = resolve_concept_name(p, alias_index).lower().strip()
            existing_pairs.add((min(c_canon, p_canon), max(c_canon, p_canon)))
            
        # Unlocks
        for u in r.get("unlocks", []):
            u_canon = resolve_concept_name(u, alias_index).lower().strip()
            existing_pairs.add((min(c_canon, u_canon), max(c_canon, u_canon)))
            
        # Related
        for rel in r.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                r_canon = resolve_concept_name(rel["concept"], alias_index).lower().strip()
                existing_pairs.add((min(c_canon, r_canon), max(c_canon, r_canon)))

    # 4. Generate candidate co-mention pairs
    candidates = collections.defaultdict(list) # (c1_low, c2_low) -> list of (doc_id, chunk_id)
    
    for chunk_key, mentions in chunk_mentions.items():
        if len(mentions) < 2:
            continue
        # Generate pairs of co-occurring concepts
        mention_list = sorted(list(mentions))
        for i in range(len(mention_list)):
            for j in range(i + 1, len(mention_list)):
                c1 = mention_list[i]
                c2 = mention_list[j]
                pair = (c1, c2)
                if pair not in existing_pairs:
                    candidates[pair].append(chunk_key)

    # 5. Score and sort candidates to prioritize cross-document bridges
    scored_candidates = []
    for pair, occurrences in candidates.items():
        c1, c2 = pair
        # Determine if they primarily originate from different documents (cross-doc bridge)
        docs_c1 = concept_docs.get(c1, set())
        docs_c2 = concept_docs.get(c2, set())
        
        is_cross_doc = len(docs_c1 & docs_c2) == 0 and len(docs_c1) > 0 and len(docs_c2) > 0
        
        # Score: cross-doc gets a large boost, then order by frequency of co-occurrences
        score = 100.0 if is_cross_doc else 0.0
        score += len(occurrences)
        
        # Pick the best occurrence chunk for provenance
        best_chunk = occurrences[0]
        scored_candidates.append((score, pair, best_chunk))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # 6. Apply co-mention RELATED edges up to the cap per concept
    co_mention_counts = collections.defaultdict(int)
    edges_added = 0
    
    for score, pair, chunk_key in scored_candidates:
        c1_low, c2_low = pair
        
        # Check concept caps
        if co_mention_counts[c1_low] >= max_co_mentions_per_concept or co_mention_counts[c2_low] >= max_co_mentions_per_concept:
            continue
            
        c1_name = canon_lookup.get(c1_low)
        c2_name = canon_lookup.get(c2_low)
        if not c1_name or not c2_name:
            continue
            
        doc_id, chunk_id = chunk_key
        provenance = f"{doc_id}:{chunk_id}"
        
        # Add to Concept 1
        rec1 = concept_by_canon.get(c1_low)
        if rec1 is not None:
            related_list = rec1.setdefault("related_to", [])
            related_list.append({"concept": c2_name, "relation": "co_mention"})
            prov = rec1.setdefault("relation_provenance", {})
            prov[f"related:{c2_name.lower()}"] = provenance
            co_mention_counts[c1_low] += 1
            
        # Add to Concept 2 (since related is symmetric, we can represent it on both or just one)
        rec2 = concept_by_canon.get(c2_low)
        if rec2 is not None:
            related_list = rec2.setdefault("related_to", [])
            related_list.append({"concept": c1_name, "relation": "co_mention"})
            prov = rec2.setdefault("relation_provenance", {})
            prov[f"related:{c1_name.lower()}"] = provenance
            co_mention_counts[c2_low] += 1
            
        edges_added += 1

    print(f"  [CO-MENTION RELATED] Added {edges_added} co-mention RELATED edges to graph inventory.")
    return okf_results

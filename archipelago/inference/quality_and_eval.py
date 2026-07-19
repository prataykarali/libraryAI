import re

def is_math_concept(name: str) -> bool:
    """Helper to detect typical mathematical/statistical terms."""
    if not name:
        return False
    keywords = [
        r"\blinear\s+regression\b",
        r"\bmatrix\b",
        r"\bvector\b",
        r"\bcalculus\b",
        r"\bgradient\b",
        r"\bprobability\b",
        r"\bstatistics\b",
        r"\blinear\s+algebra\b",
        r"\bleast\s+squares\b",
        r"\beigenvalue\b",
        r"\bsvd\b",
        r"\bsingular\s+value\b",
        r"\boptimization\b",
        r"\bderivative\b",
        r"\bintegral\b",
        r"\balgebra\b"
    ]
    pattern = "|".join(keywords)
    return bool(re.search(pattern, name, re.IGNORECASE))

def enforce_dag_relations(relations_list, concepts_dict):
    """
    Detect if any mathematical/statistical concept is designated as an unlock
    for an advanced/expert level concept and reverse the direction.
    """
    diff_lookup = {}
    if isinstance(concepts_dict, dict):
        for k, v in concepts_dict.items():
            if isinstance(v, dict):
                diff_lookup[k.lower()] = v.get("difficulty") or v.get("concept_difficulty") or "intermediate"
            elif isinstance(v, str):
                diff_lookup[k.lower()] = v
    elif isinstance(concepts_dict, list):
        for c in concepts_dict:
            if isinstance(c, dict) and "concept_name" in c:
                diff_lookup[c["concept_name"].lower()] = c.get("difficulty") or "intermediate"

    def get_diff(name):
        if not name:
            return "intermediate"
        return diff_lookup.get(name.lower(), "intermediate")

    def is_advanced(name):
        diff = get_diff(name)
        return diff.lower() in ("advanced", "expert")

    modified_relations = []
    for rel in relations_list:
        if isinstance(rel, dict):
            r = dict(rel)
            
            src_key = None
            for k in ["source", "from_name", "from", "concept_a", "from_id"]:
                if k in r:
                    src_key = k
                    break
            
            tgt_key = None
            for k in ["target", "to_name", "to", "concept_b", "to_id", "concept"]:
                if k in r:
                    tgt_key = k
                    break
            
            rel_key = None
            for k in ["relation", "relation_type", "edge_type"]:
                if k in r:
                    rel_key = k
                    break

            if src_key and tgt_key:
                src_val = r[src_key]
                tgt_val = r[tgt_key]
                rel_val = r.get(rel_key, "") if rel_key else ""

                # Case 1: Advanced non-math concept unlocks math concept
                if (is_math_concept(tgt_val) and not is_math_concept(src_val) and is_advanced(src_val) and
                    str(rel_val).lower() in ("unlocks", "enables")):
                    # Reverse so that math concept is a prerequisite
                    if rel_key:
                        r[rel_key] = "requires"
                    r[src_key] = tgt_val
                    r[tgt_key] = src_val

                # Case 2: Math concept is unlocked by Advanced concept (requires form reversed)
                elif (is_math_concept(src_val) and not is_math_concept(tgt_val) and is_advanced(tgt_val) and
                      str(rel_val).lower() in ("requires", "prerequisite")):
                    # Swap source and target to correct the direction
                    r[src_key] = tgt_val
                    r[tgt_key] = src_val

            modified_relations.append(r)
        elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
            r = list(rel)
            src_val, tgt_val, rel_val = r[0], r[1], r[2]
            if (is_math_concept(tgt_val) and not is_math_concept(src_val) and is_advanced(src_val) and
                str(rel_val).lower() in ("unlocks", "enables")):
                r[0], r[1] = tgt_val, src_val
            elif (is_math_concept(src_val) and not is_math_concept(tgt_val) and is_advanced(tgt_val) and
                  str(rel_val).lower() in ("requires", "prerequisite")):
                r[0], r[1] = tgt_val, src_val
            
            modified_relations.append(tuple(r) if isinstance(rel, tuple) else r)
        else:
            modified_relations.append(rel)
    return modified_relations

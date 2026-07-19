# Session 3 prep — after Session 2 curriculum/ranking

**Date:** 2026-07-17  
**Depends on:** Session 2 multi-hop curriculum + alias ranking (this wave)

## Done in Session 2 (baseline for S3)

- Root `okf_graph.json` ≡ `graph_ui/okf_graph.json` (138 concepts / 125 edges)
- Kuzu grafted: REQUIRES=50, UNLOCKS=6, RELATED=69
- `find_curriculum_chains` + in-bubble `#page=N` markdown links
- Alias/acronym ranking (`extract_acronym`, `generate_aliases`, core-concept bonus)
- Chat UI keeps PDF links (no longer strips `<a>`)
- Tests: `tests/unit/test_session2_curriculum_ranking.py`, `tests/integration/test_session2_multihop.py`

## Session 3 goals

| Goal | Target |
|------|--------|
| Alias-aware relation validation | `validate_relation` treats LoRA ≡ Low-Rank Adaptation |
| Co-mention RELATED builder | Chunks that co-mention A+B → RELATED edge |
| UNLOCKS direction heuristics | enables/unlocks from polarity of relation text |
| Orphan reduction | **&lt;35%** orphans (now ~54%) |
| UNLOCKS count | **&gt;20** (now 6) |
| Cross-document bridges | denser paper↔paper REQUIRES/UNLOCKS |

## Code to inspect first

1. **`okf/relations.py`** — second-pass relation extraction  
2. **`okf/cleanup.py`** — relation filters, name validation  
3. **`okf/canonicalize.py`** — alias map + fuzzy merge (extend for validation)  
4. **`okf/graph_db.py`** — `create_edge`, inventory gate  
5. Search for `validate_relation` / relation name checks in cleanup + relations

## Planned implementation sketch

### 1. Alias-aware `validate_relation`

```python
def names_equivalent(a, b, alias_index) -> bool:
    # normalize, acronym, tags, fuzzy threshold
```

- Build alias index from concept inventory + generate_aliases  
- When checking prereq/unlock target exists, resolve via alias index  
- Unit tests: "LoRA" edge target matches `low_rank_adaptation`

### 2. Co-mention RELATED builder

- After ingest / rebuild: for each chunk, concepts mentioned together  
- If pair lacks REQUIRES/UNLOCKS, emit RELATED with `relation_type=co_mention`  
- Cap edges per concept; prefer cross-doc pairs  
- Provenance: `doc_id:chunk_id`

### 3. UNLOCKS heuristics

- Map relation labels: enables / unlocks / leads_to → UNLOCKS  
- Invert REQUIRES when text says "X enables Y" (X UNLOCKS Y)  
- Break cycles with existing `enforce_dag`

### 4. Orphan + bridge metrics

- Re-run `evaluate` structural audit after builders  
- Track orphan_ratio, unlocks_count, cross_doc_edge_ratio in accuracy.json  

## Success criteria (Session 3)

- [ ] `validate_relation` accepts alias targets  
- [ ] Co-mention RELATED increases connectivity; orphans &lt; 35%  
- [ ] UNLOCKS &gt; 20 after heuristic pass  
- [ ] Cross-doc bridges increased vs ~7/125  
- [ ] pytest unit + integration green; no dual-export drift  

## Non-goals (later)

- v4 fine-tune train/deploy  
- Auth / delete-doc product path  
- Full agentic Cypher loop  

# Handoff — 2026-07-16 (post bulk-ingest + defect-fix session)

## State as of now
- ALL 8 docs ingested with aura-qwen on GPU (BERT, LoRA, RAG, probable, syllabi,
  Math-for-ML 417p, + GraphRAG/Attention from yesterday). ~60 min total.
- Graph: 138 concepts / 125 edges / 54% orphans (was 105/15/82% before fixes).
  Zero phantom nodes, zero hallucinated placeholder nodes ('Grande Jura' gone).
- Pipeline fixes LANDED in okf_pipeline.py (all tested, py_compile + pytest pass):
  * _kuzu_escape: Kuzu wants \' not SQL '' — silent MERGE failures fixed (D7 root cause).
  * apply_grounding_filter: name-in-passage required, >=30% summary-overlap rescue.
    Dropped 324 hallucinated records on rebuild.
  * dedupe_identical_records: collapsed 509 mode-collapse duplicates (Vector x407).
  * Placeholder gating in ingest_to_kuzu: relation targets must exist in the
    extracted inventory; 342 bogus edges skipped.
  * relation_provenance: edges cite the asserting record's (doc_id, chunk_id).
  * --relations-only CLI + relation_pass() second-pass extraction (code works,
    but see MODEL VERDICT below).
- import_bak2_relations.py: grafts old-model relation layer. Direct graft only
  matched 1 relation (canonical names diverged), so instead the 75 relation-rich
  bak2 RECORDS were appended to okf_results.json as a relation layer (targets
  gated by inventory). That's where most of the 125 edges come from.
- UI fixes: BOOKS entry for Math-for-ML (D11), 4 missing concept_type colors (D10).
- Backups: okf_results.json.bak2 (old-model), .bak3 (post-bulk-ingest), .bak4
  (pre-relation-graft).

## MODEL VERDICT (tester audit 2026-07-16, confirmed by inline smoke tests)
aura-qwen v3 fine-tune is NOT fit for relation extraction or textbook ingest:
- Emits relations on 13/1024 records; returns "[] [] []" even for constrained
  candidate-selection prompts (3 prompt formats tried on GPU — all failed).
- Mode-collapses on textbooks: 'Vector' x407 with 2 distinct summaries.
- ROOT CAUSE = training data (training_data/okf_train_pairs_v3.jsonl):
  'Vector' is 37% of all concept objects; 51% of pairs have empty [] output;
  only 13% of concept objects have any relation; 31/852 pairs have 3+ concepts.
  The model faithfully reproduces these statistics.

## TODO after usage-limit reset (~2:17 PM UTC+8 / 11:47 AM IST)
1. Relaunch DATASET agent → build training_data/okf_train_pairs_v4.jsonl:
   label cap ~3%/concept, <=15% empty outputs, >=60% relation coverage
   (source: bak2 relation records + hand-authored gold pairs from
   pdf_chunks.json), >=25% multi-concept, >=25% textbook passages.
   Deliver okf_dataset_report_v4.json + V4_CHANGES.md. (Agent died at 402
   before writing anything — start fresh.)
2. Tester FREEZE-AUDIT of current 138/125 graph (task: verify the 125 edges
   against passages, near-dup merges, then freeze ingestion architecture).
3. Fine-tune v4 (user GPU, no API quota needed): continue_finetune.py pointed
   at v4 files. Then re-ingest all 8 docs (~1h GPU) and re-audit.
4. THEN inference phase: okf_graph.json graph_rag_index + okf_graph.db (KùzuDB)
   + chat.py / graph_server.py.

## Ingestion usage (works today)
.venv/bin/python okf_pipeline.py --add <file.pdf|.md|.txt> --local [--limit N]
[--max-pages N]. NEVER system python3 (CPU torch 20x slower). NEVER Ollama.
Graph UI: graph_server.py :5050, reads graph_ui/okf_graph.json per request.
Caveat until v4 model: new docs get grounded concepts but few model-emitted
relations; relations come from the bak2 layer + (post-v4) the relation pass.

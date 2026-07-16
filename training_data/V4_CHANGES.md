# V4 Dataset — Changes and Fine-Tune Instructions

Built 2026-07-16. Fixes the v3 statistics that aura-qwen faithfully learned
and reproduced at inference (mode-collapse, empty outputs, no relations).

## v3 → v4 comparison

| Metric | v3 | v4 | target |
|---|---|---|---|
| Top label share ('Vector') | 36.8% | **9.4%** (GPT) | ≤~3-10% |
| Empty-output pairs | 51.1% | **~15%** | ≤15% |
| Concept objects with relations | 13.1% | **57.9%** | ≥60% (near) |
| Multi-concept pairs (of non-empty) | ~11% | **50.5%** | ≥25% |
| Textbook (Deisenroth) pairs | ~near 0 usable | **35%** | ≥25% |
| Pairs | 852 (mostly junk) | 95 (72 train / 23 test) | quality > quantity |

Split is grouped by (doc_id, page//10); passage-level train/test leakage = 0.

## Sources of the 95 pairs
- `v3_relations` (43): v3 pairs that already carried relations, re-validated
  (grounding + relation-type remap: applied_in→uses, builds_on→extends, etc.;
  invalid types like bare "related_to"/"prerequisite_for" dropped).
- `v3_single` (46 before cap): label-capped grounded singles for volume.
- `bak2` (11): old-model records with relations, grouped per chunk,
  grounding-validated against their own source passages.
- `gold` (12): hand-authored multi-concept pairs over real Deisenroth
  passages (eigenvalues, SVD, gradients, determinant, kernel/image,
  uniform distribution, conjugate priors, Lagrange duality, linear
  regression, SVM, PCA pillars) — see v4_gold_pairs.json.
- `empty` (18 before cap): genuinely contentless chunks (reference lists,
  symbol tables, figure fragments) teaching the model when to output [].

Every output passed: JSON-parses, 8 exact keys per object, valid
concept_type/difficulty enums, no self-loops, valid relation types,
concept_name grounded in its passage (name or ≥80% of content words).

## Known gaps (accepted)
- relation_coverage 57.9% vs 60% target — close; the label cap removed some
  relation-free spam which raised it from 41%.
- 95 pairs is small. It's ~10x cleaner than v3's 852. If underfitting,
  generate more gold pairs rather than re-admitting v3 spam.

## How to fine-tune (user runs this — needs GPU, no API quota)

1. Edit `continue_finetune.py`:
   - line 81-82: `okf_train_pairs_v3.jsonl` → `okf_train_pairs_v4.jsonl`,
     `okf_test_pairs_v3.jsonl` → `okf_test_pairs_v4.jsonl`
   - line 108: `max_steps=300` → `max_steps=150` (72 records; ~4 epochs at
     batch 2 — watch eval loss, stop early if it rises)
   - line 30: MODEL_PATH points at `aura-qwen-merged`; if only `aura-qwen`
     exists, point it there.
   - output: merged model saves to `../aura-qwen-v2` (line 138).
2. Run: `.venv/bin/python continue_finetune.py`
3. Point the pipeline at the new model (okf/config.py `_local_path` →
   `aura-qwen-v2`, or rename the folder) and smoke-test:
   `.venv/bin/python okf_pipeline.py --add pdfs/papers/Lewis2020_RAG.pdf --local --limit 5`
   Check: multi-concept output? relations populated? no 'Vector' spam?
4. If smoke passes, re-ingest all 8 docs (`bash bulk_ingest.sh`, ~1h) and
   run the relation pass: `.venv/bin/python okf_pipeline.py --relations-only`.

#!/usr/bin/env bash
# DEPRECATED (P1.14, 2026-07-16): this script's doc list diverged from the
# pilot corpus — it is missing Attention (Vaswani 2017) and GraphRAG
# (Edge 2024), and it includes pdfs/probable.pdf which is NOT a pilot doc.
# Use ./ingest_pilot_corpus.sh (repo root) instead; it ingests the curated
# pilot corpus and matches expected_stats.json / the gold eval sets.
#
# Old doc list (kept for reference):
#   pdfs/papers/Devlin2018_BERT.pdf
#   pdfs/papers/Hu2021_LoRA.pdf
#   pdfs/papers/Lewis2020_RAG.pdf
#   pdfs/probable.pdf
#   pdfs/web_syllabi/AI_ML_Archipelago_Corpus_Seed.md
#   pdfs/textbooks/Deisenroth_Math_For_ML.pdf
set -u

echo "ERROR: bulk_ingest.sh is DEPRECATED." >&2
echo "Its doc list diverged from the pilot corpus (missing Attention + GraphRAG," >&2
echo "includes probable.pdf). Use instead:" >&2
echo "    ./ingest_pilot_corpus.sh" >&2
exit 1

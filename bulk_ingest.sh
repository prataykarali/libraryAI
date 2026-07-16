#!/usr/bin/env bash
# Sequential GPU bulk-ingest of the 6 remaining Archipelago docs.
# Runs the aura-qwen local model via the CUDA venv (never system python3).
# Math-for-ML (417p) runs LAST because it is the biggest.
set -u

PY=.venv/bin/python
LOGDIR=ingest_logs
mkdir -p "$LOGDIR"

DOCS=(
  "pdfs/papers/Devlin2018_BERT.pdf"
  "pdfs/papers/Hu2021_LoRA.pdf"
  "pdfs/papers/Lewis2020_RAG.pdf"
  "pdfs/probable.pdf"
  "pdfs/web_syllabi/AI_ML_Archipelago_Corpus_Seed.md"
  "pdfs/textbooks/Deisenroth_Math_For_ML.pdf"
)

for d in "${DOCS[@]}"; do
  base=$(basename "$d")
  log="$LOGDIR/${base%.*}.log"
  echo "=== [$(date '+%H:%M:%S')] START $d -> $log ==="
  "$PY" okf_pipeline.py --add "$d" --local > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] FAILED $d (rc=$rc), see $log ==="
  else
    echo "=== [$(date '+%H:%M:%S')] DONE  $d ==="
  fi
done
echo "=== [$(date '+%H:%M:%S')] ALL DONE ==="

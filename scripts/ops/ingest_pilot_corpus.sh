#!/usr/bin/env bash
# Ingest the curated CS/ML pilot corpus into the local Archipelago graph.
# Uses the project venv + local model path (never system python for GPU work).
set -euo pipefail

# Resolve to app root (libraryAI/), not scripts/ops/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected venv at .venv/bin/python — create/activate the project venv first." >&2
  exit 1
fi

LOGDIR="${ROOT}/ingest_logs"
mkdir -p "$LOGDIR"

# Default: core 6 documents. Set INCLUDE_MATH_ML=1 to also ingest the textbook.
CORE_DOCS=(
  "pilot_corpus/pdfs/Vaswani2017_Attention_Is_All_You_Need.pdf"
  "pilot_corpus/pdfs/Devlin2018_BERT.pdf"
  "pilot_corpus/pdfs/Hu2021_LoRA.pdf"
  "pilot_corpus/pdfs/Lewis2020_RAG.pdf"
  "pilot_corpus/pdfs/Edge2024_GraphRAG.pdf"
  "pilot_corpus/pdfs/AI_ML_Archipelago_Corpus_Seed.md"
)

DOCS=("${CORE_DOCS[@]}")
if [[ "${INCLUDE_MATH_ML:-0}" == "1" ]]; then
  DOCS+=("pilot_corpus/pdfs/Deisenroth_Math_For_ML.pdf")
fi

echo "=== Archipelago pilot corpus ingest ==="
echo "Root: $ROOT"
echo "Documents: ${#DOCS[@]}"
echo "Tip: stop inference_server first if it holds the okf_graph.db lock."
echo

failed=0
for d in "${DOCS[@]}"; do
  if [[ ! -e "$d" ]]; then
    echo "SKIP missing: $d"
    failed=$((failed + 1))
    continue
  fi
  base="$(basename "$d")"
  log="$LOGDIR/pilot_${base%.*}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START $d -> $log ==="
  if "$PY" okf_pipeline.py --add "$d" --local >"$log" 2>&1; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DONE  $d ==="
  else
    rc=$?
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] FAILED $d (rc=$rc), see $log ==="
    failed=$((failed + 1))
  fi
done

echo
if [[ "$failed" -gt 0 ]]; then
  echo "=== FINISHED WITH $failed failure(s)/skip(s) ==="
  exit 1
fi
echo "=== ALL PILOT DOCS INGESTED ==="
echo "Next: restart graph_server / inference_server / chat_server, then ./pilot_readiness.sh"

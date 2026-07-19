#!/usr/bin/env bash
# Final pilot readiness gate script (Session 8).
#
# Full gates (default):
#   1. Unit tests          — tests/unit/
#   2. Integration tests   — tests/integration/
#   3. Live E2E            — tests/e2e/ with RUN_LIVE_E2E=1
#   4. Graph quality       — structural_audit + gold via scripts/run_gold_eval.py
#                            fails on self-loops; fails if concept_count <
#                            pilot_corpus/expected_stats.json min_concepts
#                            (clear message: re-run ./ingest_pilot_corpus.sh)
#   5. Latency             — tests/e2e/test_latency.py with RUN_LIVE_E2E=1
#
# Offline CI partial check (no live services required):
#   SKIP_LIVE=1 ./pilot_readiness.sh
#   Runs steps 1–2 + graph quality (gold eval) only; skips live E2E and latency.
#
# Prefer project venv:
#   Uses .venv/bin/python when present; otherwise falls back to python3.
#
# Exit non-zero on any failed gate.
set -euo pipefail

# Resolve to app root (libraryAI/), not scripts/ops/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
  export PATH="${ROOT}/.venv/bin:${PATH}"
else
  PY="python3"
fi

SKIP_LIVE="${SKIP_LIVE:-0}"

echo "=== Archipelago Pilot Readiness Check ==="
echo "Using: $PY"
echo "CWD:   $ROOT"
if [[ "$SKIP_LIVE" == "1" ]]; then
  echo "Mode:  SKIP_LIVE=1 (offline partial: unit + integration + graph quality)"
else
  echo "Mode:  full gates (unit + integration + live e2e + graph quality + latency)"
fi
echo

echo "1. Unit tests..."
"$PY" -m pytest tests/unit/ -q || exit 1

echo "2. Integration tests..."
"$PY" -m pytest tests/integration/ -q || exit 1

if [[ "$SKIP_LIVE" == "1" ]]; then
  echo "3. Live E2E tests... SKIPPED (SKIP_LIVE=1)"
else
  echo "3. Live E2E tests..."
  if [[ "${RUN_LIVE_E2E:-}" != "1" ]]; then
    echo "   (setting RUN_LIVE_E2E=1 for this gate)"
  fi
  RUN_LIVE_E2E=1 "$PY" -m pytest tests/e2e/ -q || exit 1
fi

echo "4. Graph quality (structural audit + gold curriculum)..."
# Prefer scripts/run_gold_eval.py: structural_audit, self-loop fail,
# thin-graph fail vs expected_stats.json, evaluate_pipeline on pilot gold sets.
"$PY" "${ROOT}/scripts/run_gold_eval.py" || exit 1

if [[ "$SKIP_LIVE" == "1" ]]; then
  echo "5. Latency... SKIPPED (SKIP_LIVE=1)"
  echo
  echo "=== PARTIAL GATES PASSED (SKIP_LIVE=1) ==="
  echo "Offline unit/integration + graph quality OK."
  echo "For full pilot sign-off, re-run without SKIP_LIVE (services up):"
  echo "  RUN_LIVE_E2E=1 ./pilot_readiness.sh"
  exit 0
fi

echo "5. Latency..."
RUN_LIVE_E2E=1 "$PY" -m pytest tests/e2e/test_latency.py -v || exit 1

echo
echo "=== ALL GATES PASSED ==="
echo "Pilot may proceed to departmental deployment checklist sign-off."
echo "See PILOT_READINESS_REPORT.md and PRIVACY_POLICY.md."

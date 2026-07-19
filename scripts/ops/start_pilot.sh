#!/usr/bin/env bash
# One-command pilot launch for Archipelago.
#
#   ./scripts/ops/start_pilot.sh              # local dev (open librarian)
#   ARCHIPELAGO_LIBRARIAN_TOKEN='...' ./scripts/ops/start_pilot.sh   # shared machine
#
# What it does:
#   1. Starts the 3 services via serve.sh (inference :5051, graph :5050, chat :5052)
#   2. Runs the 7-query demo gate (scripts/ops/demo_check.sh)
#   3. Prints the pilot URL table
#
# Exits non-zero if any service fails to come up or any demo query fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== Archipelago pilot launch ==="

if [[ -z "${ARCHIPELAGO_LIBRARIAN_TOKEN:-}" && -z "${ARCHIPELAGO_TOKEN:-}" ]]; then
  echo ""
  echo "  NOTE: no ARCHIPELAGO_LIBRARIAN_TOKEN set — librarian upload/delete is OPEN."
  echo "  Fine for local dev; on a shared machine set a token first:"
  echo "      export ARCHIPELAGO_LIBRARIAN_TOKEN='change-me'"
  echo ""
fi

./scripts/ops/serve.sh start || { echo "ERROR: services failed to start"; exit 1; }

echo ""
echo "Running 7-query demo gate (proves routing is live, not stale) ..."
if ./scripts/ops/demo_check.sh; then
  DEMO=OK
else
  DEMO=FAILED
fi

cat <<'EOF'

Pilot URLs
  Student    http://localhost:5052            query only (no token, no upload)
  Librarian  http://localhost:5050            Librarian tab: upload / delete / manual nodes
  API        http://localhost:5051            backend

After a restart, hard-refresh browsers (Ctrl+Shift+R).
Ollama (localhost:11434) is optional — without it, chat uses template answers.

Pilot scope: chat over the pilot PDFs + seed concepts; prereqs / related /
books for topics in the graph. Not in scope: perfect textbooks for every
topic, or "any PDF becomes a perfect knowledge graph". See PILOT_LAUNCH.md.
EOF

if [[ "$DEMO" == "FAILED" ]]; then
  echo "WARNING: demo gate FAILED — a running process is stale or answering from"
  echo "the wrong tree. Try: ./scripts/ops/serve.sh restart && ./scripts/ops/demo_check.sh"
  exit 1
fi

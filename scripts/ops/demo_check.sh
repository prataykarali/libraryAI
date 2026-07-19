#!/usr/bin/env bash
# 7-query pilot demo gate ("prove the patch").
#
# Sends the launch demo script's queries to the live chat API (:5051) and
# checks the routing/anchor each one must produce. Run after every restart:
#
#   ./scripts/ops/demo_check.sh
#
# Any FAIL means the running process is stale or answering from the wrong
# tree — restart with scripts/ops/serve.sh and re-run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

API="${ARCHIPELAGO_API:-http://localhost:5051}"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

PASS=0
FAIL=0

# ask <query> → prints the first line (JSON payload) of the streamed response
ask() {
  curl -s -X POST "${API}/api/chat" \
    -H 'Content-Type: application/json' \
    ${ARCHIPELAGO_TOKEN:+-H "X-Archipelago-Token: ${ARCHIPELAGO_TOKEN}"} \
    --max-time 240 \
    -d "$(printf '{"query":"%s"}' "$1")" | head -1
}

# body <query> → prints the streamed text after [STREAM_START]
body() {
  curl -s -X POST "${API}/api/chat" \
    -H 'Content-Type: application/json' \
    ${ARCHIPELAGO_TOKEN:+-H "X-Archipelago-Token: ${ARCHIPELAGO_TOKEN}"} \
    --max-time 240 \
    -d "$(printf '{"query":"%s"}' "$1")" | sed -n '/\[STREAM_START\]/,$p'
}

# check <label> <query> <python-expr over payload dict d> [body-grep] [body-antigrep]
check() {
  local label="$1" query="$2" expr="$3" grep_re="${4:-}" anti_re="${5:-}"
  local payload ok=1
  payload="$(ask "$query")"
  if [[ -z "$payload" ]]; then
    echo "FAIL  ${label}: no response from ${API} (server down?)"
    FAIL=$((FAIL+1)); return
  fi
  if ! echo "$payload" | "$PY" -c "
import sys, json
d = json.loads(sys.stdin.readline())
sys.exit(0 if (${expr}) else 1)
" 2>/dev/null; then
    ok=0
  fi
  if [[ $ok -eq 1 && ( -n "$grep_re" || -n "$anti_re" ) ]]; then
    local text
    text="$(body "$query")"
    if [[ -n "$grep_re" ]] && ! echo "$text" | grep -qiE "$grep_re"; then ok=0; fi
    if [[ -n "$anti_re" ]] && echo "$text" | grep -qiE "$anti_re"; then ok=0; fi
  fi
  if [[ $ok -eq 1 ]]; then
    echo "PASS  ${label}"
    PASS=$((PASS+1))
  else
    echo "FAIL  ${label}  (query: ${query})"
    echo "      payload: $(echo "$payload" | head -c 300)"
    FAIL=$((FAIL+1))
  fi
}

route()  { echo "d.get('routing',{}).get('route') == '$1'"; }

echo "=== Archipelago 7-query demo gate → ${API} ==="

# 1. Onboarding
check "1 onboarding"  "hi i wanna start learning AIML"  "$(route onboarding)"

# 2. Identity
check "2 identity"    "who are you"                     "$(route identity)"

# 3. RAG family — must NOT be rejected as "not related"
check "3 rag family"  "various sorts of RAGs" \
  "d.get('routing',{}).get('route') in ('graph_strong','graph_soft') and (d.get('anchor_concept') or {}).get('id','').startswith('rag')"

# 4. AI agent — agent/ReAct-ish anchor
check "4 ai agent"    "i wanna build an AI agent" \
  "d.get('routing',{}).get('route') in ('graph_strong','graph_soft') and 'agent' in ((d.get('anchor_concept') or {}).get('id') or '')"

# 5. Neural networks — no GNN / dimensionality-reduction offered as "learn first"
check "5 neural nets" "neural networks" \
  "d.get('routing',{}).get('route') in ('graph_strong','graph_soft') and not any('graph_neural' in (p.get('id') or '') or 'dimensionality' in (p.get('id') or '') for p in d.get('prerequisites',[]) if isinstance(p, dict))"

# 6. Books on deep learning — library route, textbook ranked first
check "6 dl books"    "books on deep learning"          "$(route library_books)" \
  "textbook"

# 7. Out-of-scope stays out of scope
check "7 oos stars"   "suggest books about stars"       "$(route out_of_scope)"

echo ""
echo "=== ${PASS} passed, ${FAIL} failed ==="
[[ $FAIL -eq 0 ]] || exit 1

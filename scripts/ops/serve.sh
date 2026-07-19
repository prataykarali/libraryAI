#!/usr/bin/env bash
# Archipelago process launcher (P1-12).
#
# Starts/stops the three Flask services with the CUDA venv python, logging to
# logs/<service>.log and tracking PIDs in logs/<service>.pid:
#
#   inference  — python -m archipelago.apps.inference_app  (port 5051, chat API)
#   graph      — graph_server.py                           (port 5050, graph API + graph UI)
#   chat       — chat_server.py                            (port 5052, chat UI)
#
# Usage: serve.sh start|stop|status|restart
#
# Environment passthrough: ARCHIPELAGO_BIND / ARCHIPELAGO_TOKEN are honored by
# the services themselves; this script just inherits and forwards the env.
# Ollama (localhost:11434) is optional — chat falls back to template answers.
set -euo pipefail

# Resolve to app root (libraryAI/), not scripts/ops/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: ${PY} not found — create the venv first (see README)." >&2
  exit 1
fi

LOGS="${ROOT}/logs"
mkdir -p "$LOGS"

HOST="${ARCHIPELAGO_BIND:-127.0.0.1}"
# URLs a browser should open; 0.0.0.0 binds are reachable via localhost
URL_HOST="$HOST"
[[ "$URL_HOST" == "0.0.0.0" ]] && URL_HOST="localhost"

SERVICES=(inference graph chat)

svc_port()    { case "$1" in inference) echo 5051;; graph) echo 5050;; chat) echo 5052;; esac; }
svc_cmd()     { case "$1" in
                  inference) echo "-m archipelago.apps.inference_app";;
                  graph)     echo "graph_server.py";;
                  chat)      echo "chat_server.py";;
                esac; }
svc_desc()    { case "$1" in
                  inference) echo "inference/chat API";;
                  graph)     echo "graph API + graph UI";;
                  chat)      echo "chat UI";;
                esac; }

is_running() {  # $1 = service name; returns 0 if pidfile points at a live process
  local pidfile="${LOGS}/$1.pid"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_up() {  # $1 = port
  # Pass the shared token when set so authed services don't read as DOWN (401)
  curl -sf -o /dev/null --max-time 2 \
    ${ARCHIPELAGO_TOKEN:+-H "X-Archipelago-Token: ${ARCHIPELAGO_TOKEN}"} \
    "http://${URL_HOST}:$1/" 2>/dev/null
}

start_one() {
  local name="$1" port cmd
  port="$(svc_port "$name")"
  cmd="$(svc_cmd "$name")"
  if is_running "$name"; then
    echo "  ${name}: already running (pid $(cat "${LOGS}/${name}.pid"))"
    return 0
  fi
  # A foreign process (e.g. a manually launched python3 server) holding the
  # port makes our nohup'd start die with "Address already in use" while the
  # stale code keeps serving — fail loudly instead of silently shadowing.
  if port_up "$port"; then
    echo "  ${name}: ERROR — port ${port} already in use by another process." >&2
    echo "           Find it with: ss -tlnp | grep ${port}   then kill it and retry." >&2
    return 1
  fi
  # shellcheck disable=SC2086  # cmd is intentionally word-split (-m module form)
  nohup "$PY" $cmd >> "${LOGS}/${name}.log" 2>&1 &
  echo $! > "${LOGS}/${name}.pid"
  echo "  ${name}: started pid $! → port ${port} ($(svc_desc "$name")), log: logs/${name}.log"
}

stop_one() {
  local name="$1" pidfile="${LOGS}/$1.pid"
  if ! is_running "$name"; then
    echo "  ${name}: not running"
    rm -f "$pidfile"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
    echo "  ${name}: force-killed pid ${pid}"
  else
    echo "  ${name}: stopped pid ${pid}"
  fi
  rm -f "$pidfile"
}

status_all() {
  local name port
  for name in "${SERVICES[@]}"; do
    port="$(svc_port "$name")"
    if port_up "$port"; then
      echo "  ${name} (:${port}, $(svc_desc "$name")): UP"
    elif is_running "$name"; then
      echo "  ${name} (:${port}, $(svc_desc "$name")): STARTING (process alive, port not answering yet)"
    else
      echo "  ${name} (:${port}, $(svc_desc "$name")): DOWN"
    fi
  done
  # Ollama is optional — warn, don't fail
  if curl -sf -o /dev/null --max-time 2 "http://localhost:11434/" 2>/dev/null; then
    echo "  ollama (:11434, NL synthesis): UP"
  else
    echo "  WARNING: Ollama DOWN — chat will use template fallback; start with: ollama serve"
  fi
}

wait_for_inference() {
  # Inference loads embedding models on startup — allow up to ~60s
  local port deadline
  port="$(svc_port inference)"
  deadline=$((SECONDS + 60))
  echo -n "Waiting for inference server on :${port} (loads embedding models, up to 60s) "
  while (( SECONDS < deadline )); do
    if port_up "$port"; then
      echo " UP"
      return 0
    fi
    if ! is_running inference; then
      echo " FAILED"
      echo "ERROR: inference process died — see logs/inference.log (last 20 lines):" >&2
      tail -20 "${LOGS}/inference.log" >&2 || true
      return 1
    fi
    echo -n "."
    sleep 2
  done
  echo " TIMEOUT"
  echo "WARNING: inference not answering after 60s — check logs/inference.log" >&2
  return 1
}

print_urls() {
  echo ""
  echo "Open in your browser:"
  echo "  Chat UI:  http://${URL_HOST}:$(svc_port chat)/"
  echo "  Graph UI: http://${URL_HOST}:$(svc_port graph)/"
}

cmd="${1:-}"
case "$cmd" in
  start)
    echo "Starting Archipelago services (bind: ${HOST}) ..."
    for s in "${SERVICES[@]}"; do start_one "$s"; done
    wait_for_inference || true
    echo ""
    status_all
    print_urls
    ;;
  stop)
    echo "Stopping Archipelago services ..."
    for s in "${SERVICES[@]}"; do stop_one "$s"; done
    ;;
  status)
    status_all
    ;;
  restart)
    "$0" stop
    exec "$0" start
    ;;
  *)
    echo "Usage: $0 start|stop|status|restart" >&2
    exit 2
    ;;
esac

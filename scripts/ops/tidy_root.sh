#!/usr/bin/env bash
# Move safe root clutter (logs / pid files) into data/logs/.
# Does NOT touch okf_*.json, okf_graph.db, backups, or training data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${ROOT}/data/logs"
mkdir -p "${LOG_DIR}"

moved=0
for name in \
  graph_server.log \
  graph_server.pid \
  ingest_attention.log \
  ingest_graphrag.log \
  graph_output.txt
do
  src="${ROOT}/${name}"
  if [[ -f "${src}" ]]; then
    dest="${LOG_DIR}/${name}"
    # Avoid clobbering an existing archive; timestamp if needed.
    if [[ -e "${dest}" ]]; then
      dest="${LOG_DIR}/${name}.$(date +%Y%m%d_%H%M%S)"
    fi
    mv "${src}" "${dest}"
    echo "moved ${name} -> ${dest#${ROOT}/}"
    moved=$((moved + 1))
  fi
done

echo "done: ${moved} file(s) moved into data/logs/"
echo "note: *.bak* and *.thin.* artifacts are intentional backups — leave them,"
echo "      or archive under data/backups/ only after checking scripts/docs."

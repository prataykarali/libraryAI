"""
Archipelago Graph Server
Serves real-time OKF graph data from okf_graph.json via REST API and hosts the UI.
Runs on port 5050.
"""

import json
import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "okf_graph.json"
STATIC_DIR = BASE_DIR / "graph_ui"

app = Flask(__name__, static_folder=str(STATIC_DIR))

# ── Optional shared-token auth ────────────────────────────────────────────────
@app.before_request
def check_token():
    """If ARCHIPELAGO_TOKEN is set, require a matching X-Archipelago-Token
    header on every request; when unset this is a no-op (pilot default)."""
    token = os.environ.get("ARCHIPELAGO_TOKEN")
    if not token:
        return None
    if request.method == "OPTIONS":  # CORS preflight carries no custom headers
        return None
    if request.headers.get("X-Archipelago-Token") != token:
        return jsonify({"error": "unauthorized"}), 401
    return None

# ── CORS Configuration ────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

# ── Load data (live-reload every request in dev) ──────────────────────────────
def load_graph():
    """Read okf_graph.json fresh on every call — real-time, no cache."""
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/graph")
def api_graph():
    """
    Returns the full raw graph as extracted by the model.
    Source: okf_graph.json → visualization block (richest OKF fields)
    """
    data = load_graph()
    viz  = data["visualization"]
    raw_edges = data["edges"]

    nodes = viz["nodes"]
    links = raw_edges

    # Normalise edges to source/target for D3
    edges = [
        {
            "id":       e.get("id", f"e{i}"),
            "source":   e["from_id"],
            "target":   e["to_id"],
            "relation": e["relation"],
            "edge_type": e["edge_type"],
            "source_ref": e.get("source", ""),
        }
        for i, e in enumerate(links)
    ]

    return jsonify({
        "nodes":  nodes,
        "edges":  edges,
        "stats":  data.get("stats", {}),
        "clusters": viz.get("clusters", {}),
    })

@app.route("/api/schema")
def api_schema():
    """Returns the OKF v1.6 schema spec as JSON for the UI info panel."""
    return jsonify({
        "version": "OKF v1.6",
        "description": "Open Knowledge Format — contract the local SLM follows to turn text chunks into graph-ready concepts.",
        "node_fields": {
            "concept_name":  "short noun phrase ≤5 words, Title Case, stable across docs",
            "concept_type":  "method | metric | technique | theory | tool | dataset | result | definition",
            "difficulty":    "foundational | intermediate | advanced | expert",
            "summary":       "1–2 sentences — what it IS",
            "prerequisites": "concepts you must know BEFORE this → REQUIRES edges",
            "unlocks":       "concepts this ENABLES downstream → UNLOCKS edges",
            "related_to":    "uses | extends | contrasts_with | evaluated_by | variant_of | part_of",
            "tags":          "lowercase-hyphenated keywords",
        },
        "edge_types": {
            "REQUIRES":  "prerequisite relationship — foundational dependency",
            "UNLOCKS":   "enables downstream learning — forward progression",
            "RELATED":   "secondary relationships: uses / extends / contrasts_with / evaluated_by / part_of",
        },
        "extraction_model": "Local SLM via Ollama",
        "pipeline_file": "okf_pipeline.py",
        "data_file": "okf_graph.json",
    })

@app.route("/api/node/<node_id>")
def api_node(node_id):
    """Returns full OKF data for a single node including all its edges."""
    data  = load_graph()
    viz   = data["visualization"]
    nodes = {n["id"]: n for n in viz["nodes"]}

    if node_id not in nodes:
        return jsonify({"error": "Node not found"}), 404

    node  = nodes[node_id]
    edges = data["edges"]

    outgoing = [e for e in edges if e["from_id"] == node_id]
    incoming = [e for e in edges if e["to_id"]   == node_id]

    return jsonify({
        "node":     node,
        "outgoing": outgoing,
        "incoming": incoming,
    })

@app.route("/api/stats")
def api_stats():
    data = load_graph()
    edges = data["edges"]
    from collections import Counter
    rel_counts = Counter(e["relation"] for e in edges)
    type_counts = Counter(n.get("concept_type","?") for n in data["visualization"]["nodes"])
    diff_counts = Counter(n.get("difficulty","?")   for n in data["visualization"]["nodes"])
    return jsonify({
        "total_concepts": data["stats"]["total_concepts"],
        "total_edges":    data["stats"]["total_edges"],
        "relation_breakdown": dict(rel_counts),
        "type_breakdown":     dict(type_counts),
        "difficulty_breakdown": dict(diff_counts),
        "data_file": str(DATA_FILE),
        "last_modified": os.path.getmtime(DATA_FILE),
    })

# ── Serve the UI ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/archipelago_graph.html")
def archipelago_graph():
    """Original graph URL kept alive, but served from the live data viewer."""
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    if filename.lower() in ("neon.html", "neon"):
        return "Holographic / animated graph is disabled because it crashes the browser. Use / instead.", 410
    return send_from_directory(str(STATIC_DIR), filename)

if __name__ == "__main__":
    STATIC_DIR.mkdir(exist_ok=True)
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Archipelago Graph UI Server                     ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Data: {DATA_FILE}     ║")
    print("║  Open: http://localhost:5050                      ║")
    print("╚══════════════════════════════════════════════════╝\n")
    app.run(host=os.environ.get("ARCHIPELAGO_BIND", "127.0.0.1"), port=5050, debug=False)

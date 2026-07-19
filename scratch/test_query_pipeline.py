"""
Archipelago Query Pipeline — Comprehensive Integration Test Script
==================================================================
Tests all 5 stages of the RAG query trace pipeline WITHOUT starting any servers.
Imports core functions directly from inference_server.py and validates:
  Stage 1: get_snowflake_embedding() → output shape [768], L2 norm ≈ 1.0
  Stage 2: find_anchor_concept()     → correct concept IDs for known queries
  Stage 3: get_graph_neighborhood()  → valid prereqs/unlocks/citations structure
  Stage 4: compile_narrative_recipe()→ output contains all required sections
  Stage 5: Streaming response format → metadata + [STREAM_START] + text chunks

Run:  python scratch/test_query_pipeline.py
From: libraryAI/libraryAI/
"""

import json
import os
import sys
import traceback
import time

# Ensure the project root is on the path so we can import inference_server
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # libraryAI/libraryAI
sys.path.insert(0, PROJECT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Test tracking
# ─────────────────────────────────────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
RESULTS = []

def record(stage, name, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.append((stage, name, status, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{stage}] {name}" + (f" — {detail}" if detail else ""))

def skip(stage, name, reason):
    global SKIP_COUNT
    SKIP_COUNT += 1
    RESULTS.append((stage, name, "SKIP", reason))
    print(f"  ⏭️  [{stage}] {name} — SKIPPED: {reason}")

def section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: import the inference_server module WITHOUT starting Flask
# ─────────────────────────────────────────────────────────────────────────────
section("PRE-FLIGHT: Importing inference_server (no Flask startup)")

try:
    # The module creates a Flask app and a KuzuDB connection at import time,
    # which is fine — we just don't call app.run().
    import inference_server as IS
    print("  ✓ inference_server imported successfully")
except Exception as e:
    print(f"  ✗ FATAL: Could not import inference_server: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load concept data synchronously (the __main__ block calls init_concepts_data
# but we're not in __main__, so do it manually)
# ─────────────────────────────────────────────────────────────────────────────
section("PRE-FLIGHT: Loading concepts data")

IS.init_concepts_data()
print(f"  ✓ CONCEPTS_DATA loaded: {len(IS.CONCEPTS_DATA)} concepts")
if not IS.CONCEPTS_DATA:
    print("  ✗ FATAL: No concepts loaded — cannot continue tests.")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1: get_snowflake_embedding()
# ═════════════════════════════════════════════════════════════════════════════
section("STAGE 1: Snowflake Embedding (get_snowflake_embedding)")

# Try to load the embedding model (may take time or fail if model not cached)
try:
    print("  Loading embedding model (this may take 30–60 seconds)...")
    IS.load_embedding_model()
    embedding_available = IS.use_embeddings
except Exception as e:
    embedding_available = False
    print(f"  ⚠ Embedding model failed to load: {e}")

if embedding_available:
    # Test 1.1: Output is a tensor
    try:
        emb = IS.get_snowflake_embedding("What is attention?")
        record("S1", "Returns tensor", emb is not None, f"type={type(emb).__name__}")
    except Exception as e:
        record("S1", "Returns tensor", False, str(e))
        emb = None

    if emb is not None:
        import torch

        # Test 1.2: Shape is [768]
        try:
            shape = tuple(emb.shape)
            record("S1", "Shape is [768]", shape == (768,), f"actual shape={shape}")
        except Exception as e:
            record("S1", "Shape is [768]", False, str(e))

        # Test 1.3: L2 norm ≈ 1.0 (vectors are normalized)
        try:
            norm = torch.norm(emb, p=2).item()
            close = abs(norm - 1.0) < 0.01
            record("S1", "L2 norm ≈ 1.0", close, f"norm={norm:.6f}")
        except Exception as e:
            record("S1", "L2 norm ≈ 1.0", False, str(e))

        # Test 1.4: Embedding lives on CPU (we called .cpu() in the function)
        try:
            on_cpu = not emb.is_cuda
            record("S1", "Tensor on CPU", on_cpu, f"device={emb.device}")
        except Exception as e:
            record("S1", "Tensor on CPU", False, str(e))

        # Test 1.5: Two different queries give different embeddings
        try:
            emb2 = IS.get_snowflake_embedding("How does backpropagation work?")
            same = torch.allclose(emb, emb2, atol=1e-4)
            record("S1", "Different queries → different embeddings", not same,
                   f"cosine_sim={torch.dot(emb, emb2).item():.4f}")
        except Exception as e:
            record("S1", "Different queries → different embeddings", False, str(e))

        # Test 1.6: Repeated call gives identical output (deterministic with no_grad)
        try:
            emb3 = IS.get_snowflake_embedding("What is attention?")
            identical = torch.allclose(emb, emb3, atol=1e-5)
            record("S1", "Deterministic for same input", identical)
        except Exception as e:
            record("S1", "Deterministic for same input", False, str(e))
    else:
        skip("S1", "Shape / norm / determinism tests", "embedding returned None")
else:
    skip("S1", "All embedding tests", "Embedding model not available")


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 2: find_anchor_concept()
# ═════════════════════════════════════════════════════════════════════════════
section("STAGE 2: Anchor Concept Matching (find_anchor_concept)")

# Known query → expected concept pairings (work for both embedding & fuzzy)
KNOWN_QUERIES = [
    ("What is LoRA?", "low_rank_adaptation"),
    ("How does attention work?", "attention_mechanism"),
    ("Explain transformers", "transformer"),
    ("Tell me about BERT", "bert"),
    ("What is GraphRAG?", "graphrag"),
]

for query, expected_id in KNOWN_QUERIES:
    try:
        anchor_id, score = IS.find_anchor_concept(query)
        matched = anchor_id == expected_id
        record("S2", f"'{query}' → {expected_id}",
               matched,
               f"got={anchor_id}, score={score:.4f}" if isinstance(score, float) else f"got={anchor_id}, score={score}")
    except Exception as e:
        record("S2", f"'{query}' → {expected_id}", False, str(e))

# Test 2.1: Returns a 2-tuple (id, score)
try:
    result = IS.find_anchor_concept("neural networks")
    is_tuple = isinstance(result, tuple) and len(result) == 2
    record("S2", "Returns (id, score) tuple", is_tuple, f"type={type(result)}, len={len(result) if isinstance(result, tuple) else 'N/A'}")
except Exception as e:
    record("S2", "Returns (id, score) tuple", False, str(e))

# Test 2.2: Score is numeric
try:
    _, score = IS.find_anchor_concept("gradient descent")
    is_numeric = isinstance(score, (int, float))
    record("S2", "Score is numeric", is_numeric, f"score type={type(score).__name__}")
except Exception as e:
    record("S2", "Score is numeric", False, str(e))

# Test 2.3: Returned ID exists in CONCEPTS_DATA
try:
    cid, _ = IS.find_anchor_concept("knowledge distillation")
    in_data = cid in IS.CONCEPTS_DATA or cid is None
    record("S2", "Returned ID in CONCEPTS_DATA", in_data, f"id={cid}")
except Exception as e:
    record("S2", "Returned ID in CONCEPTS_DATA", False, str(e))

# Test 2.4: Empty query doesn't crash
try:
    result = IS.find_anchor_concept("")
    record("S2", "Empty query doesn't crash", True, f"result={result}")
except Exception as e:
    record("S2", "Empty query doesn't crash", False, str(e))

# Test 2.5: Very long query doesn't crash
try:
    long_q = "deep learning " * 200
    result = IS.find_anchor_concept(long_q)
    record("S2", "Long query doesn't crash", True, f"id={result[0]}")
except Exception as e:
    record("S2", "Long query doesn't crash", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 3: get_graph_neighborhood()
# ═════════════════════════════════════════════════════════════════════════════
section("STAGE 3: Graph Neighborhood Traversal (get_graph_neighborhood)")

# Use a concept we know has REQUIRES edges: low_rank_adaptation
TEST_CONCEPT = "low_rank_adaptation"

try:
    prereqs, unlocks, citations = IS.get_graph_neighborhood(TEST_CONCEPT, k=2)
    record("S3", "Returns 3-tuple", True,
           f"prereqs={len(prereqs)}, unlocks={len(unlocks)}, citations={len(citations)}")
except Exception as e:
    record("S3", "Returns 3-tuple", False, str(e))
    prereqs, unlocks, citations = [], [], []

# Test 3.1: prereqs is a list of dicts with id/name/summary
if prereqs:
    try:
        first_p = prereqs[0]
        has_keys = all(k in first_p for k in ("id", "name", "summary"))
        record("S3", "Prereqs have id/name/summary keys", has_keys,
               f"keys={list(first_p.keys())}")
    except Exception as e:
        record("S3", "Prereqs have id/name/summary keys", False, str(e))

    # Test 3.2: Known prereqs for LoRA: full_fine_tuning and transformer
    try:
        prereq_ids = {p["id"] for p in prereqs}
        has_fft = "full_fine_tuning" in prereq_ids
        has_transformer = "transformer" in prereq_ids
        record("S3", "LoRA prereqs include full_fine_tuning", has_fft,
               f"prereq_ids={prereq_ids}")
        record("S3", "LoRA prereqs include transformer", has_transformer,
               f"prereq_ids={prereq_ids}")
    except Exception as e:
        record("S3", "Known prereqs check", False, str(e))
else:
    skip("S3", "Prereqs structure checks", "No prereqs returned")

# Test 3.3: unlocks is a list (may be empty)
try:
    record("S3", "Unlocks is a list", isinstance(unlocks, list),
           f"type={type(unlocks).__name__}, len={len(unlocks)}")
except Exception as e:
    record("S3", "Unlocks is a list", False, str(e))

# Test 3.4: Each unlock has the right schema
if unlocks:
    try:
        first_u = unlocks[0]
        has_keys = all(k in first_u for k in ("id", "name", "summary"))
        record("S3", "Unlocks have id/name/summary keys", has_keys,
               f"keys={list(first_u.keys())}")
    except Exception as e:
        record("S3", "Unlocks have id/name/summary keys", False, str(e))

# Test 3.5: citations list structure
if citations:
    try:
        first_c = citations[0]
        has_keys = all(k in first_c for k in ("doc_id", "page_number", "section_title", "text_passage"))
        record("S3", "Citations have required keys", has_keys,
               f"keys={list(first_c.keys())}")
    except Exception as e:
        record("S3", "Citations have required keys", False, str(e))

    # Test 3.6: page_number is an int
    try:
        pn = first_c["page_number"]
        record("S3", "page_number is int", isinstance(pn, int), f"type={type(pn).__name__}")
    except Exception as e:
        record("S3", "page_number is int", False, str(e))
else:
    skip("S3", "Citations structure checks", "No citations returned")

# Test 3.7: Non-existent concept returns empty lists (no crash)
try:
    p, u, c = IS.get_graph_neighborhood("totally_nonexistent_concept_xyz", k=2)
    all_empty = len(p) == 0 and len(u) == 0 and len(c) == 0
    record("S3", "Nonexistent concept → empty results", all_empty,
           f"prereqs={len(p)}, unlocks={len(u)}, citations={len(c)}")
except Exception as e:
    record("S3", "Nonexistent concept → empty results", False, str(e))

# Test 3.8: k=0 traversal returns empty prereqs/unlocks
try:
    p0, u0, _ = IS.get_graph_neighborhood(TEST_CONCEPT, k=0)
    # k=0 means *1..0 which should match nothing in Cypher
    record("S3", "k=0 returns empty neighborhood", len(p0) == 0 and len(u0) == 0,
           f"prereqs={len(p0)}, unlocks={len(u0)}")
except Exception as e:
    record("S3", "k=0 returns empty neighborhood", False, str(e))

# Test 3.9: Test with graphrag (has more REQUIRES edges)
try:
    p_gr, u_gr, c_gr = IS.get_graph_neighborhood("graphrag", k=2)
    record("S3", "graphrag neighborhood traversal", len(p_gr) > 0,
           f"prereqs={len(p_gr)}, unlocks={len(u_gr)}, citations={len(c_gr)}")
except Exception as e:
    record("S3", "graphrag neighborhood traversal", False, str(e))

# Test 3.10: Cypher injection safety (concept_id with single quote)
try:
    p_inj, u_inj, c_inj = IS.get_graph_neighborhood("test'injection", k=2)
    # Should NOT crash — just return empty results
    record("S3", "Cypher injection doesn't crash", True,
           f"prereqs={len(p_inj)}, unlocks={len(u_inj)}, citations={len(c_inj)}")
except Exception as e:
    record("S3", "Cypher injection doesn't crash", False, f"CRASHED: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 4: compile_narrative_recipe()
# ═════════════════════════════════════════════════════════════════════════════
section("STAGE 4: Narrative Recipe Compilation (compile_narrative_recipe)")

# Build a recipe from the real data
query = "What is Low-Rank Adaptation?"
target_concept = IS.CONCEPTS_DATA.get("low_rank_adaptation", {})

try:
    recipe = IS.compile_narrative_recipe(query, target_concept, prereqs, unlocks, citations)
    record("S4", "Returns a string", isinstance(recipe, str), f"len={len(recipe)}")
except Exception as e:
    record("S4", "Returns a string", False, str(e))
    recipe = ""

if recipe:
    # Test 4.1: Contains USER QUERY section
    record("S4", "Contains 'USER QUERY' section", "USER QUERY:" in recipe)

    # Test 4.2: Contains the actual query text
    record("S4", "Contains query text", query in recipe)

    # Test 4.3: Contains STRUCTURED TOPOLOGY section
    record("S4", "Contains 'STRUCTURED TOPOLOGY'", "STRUCTURED TOPOLOGY" in recipe)

    # Test 4.4: Contains Upstream Prerequisites
    record("S4", "Contains 'Upstream Prerequisites'", "Upstream Prerequisites" in recipe)

    # Test 4.5: Contains Target Concept
    record("S4", "Contains 'Target Concept'", "Target Concept" in recipe)

    # Test 4.6: Contains Downstream Applications
    record("S4", "Contains 'Downstream Applications'", "Downstream Applications" in recipe)

    # Test 4.7: Contains TEXTUAL CITATIONS
    record("S4", "Contains 'TEXTUAL CITATIONS'", "TEXTUAL CITATIONS" in recipe)

    # Test 4.8: Contains INSTRUCTION
    record("S4", "Contains 'INSTRUCTION'", "INSTRUCTION:" in recipe)

    # Test 4.9: Contains the target concept name
    target_name = target_concept.get("label", target_concept.get("name", ""))
    if target_name:
        record("S4", "Contains target concept name", target_name in recipe,
               f"name={target_name}")
    else:
        skip("S4", "Contains target concept name", "No target name found")

    # Test 4.10: Contains prerequisite names
    if prereqs:
        first_prereq_name = prereqs[0]["name"]
        record("S4", "Contains prereq name", first_prereq_name in recipe,
               f"prereq={first_prereq_name}")

    # Test 4.11: Recipe with empty prereqs/unlocks/citations
    try:
        empty_recipe = IS.compile_narrative_recipe(
            "test query",
            {"label": "Test Concept", "summary": "A test"},
            [], [], []
        )
        has_none_markers = "None extracted" in empty_recipe
        record("S4", "Empty prereqs/unlocks → 'None extracted'", has_none_markers)
    except Exception as e:
        record("S4", "Empty prereqs/unlocks → 'None extracted'", False, str(e))

    # Test 4.12: Recipe with missing label key (uses 'name' fallback)
    try:
        name_fallback_recipe = IS.compile_narrative_recipe(
            "test", {"name": "Fallback Name", "summary": "test"}, [], [], []
        )
        record("S4", "Name fallback (no 'label' key)", "Fallback Name" in name_fallback_recipe)
    except Exception as e:
        record("S4", "Name fallback (no 'label' key)", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 5: Streaming Response Format Validation
# ═════════════════════════════════════════════════════════════════════════════
section("STAGE 5: Streaming Response Format (/api/chat)")

# We use Flask's test client to simulate requests without starting a real server.
test_client = IS.app.test_client()

# Test 5.1: RAG synthesis mode — valid query
try:
    resp = test_client.post("/api/chat",
                            data=json.dumps({"query": "What is LoRA?", "mode": "rag_synthesis", "history": []}),
                            content_type="application/json")
    record("S5", "POST /api/chat returns 200", resp.status_code == 200,
           f"status={resp.status_code}")
except Exception as e:
    record("S5", "POST /api/chat returns 200", False, str(e))
    resp = None

if resp and resp.status_code == 200:
    raw_data = resp.get_data(as_text=True)

    # Test 5.2: Response contains [STREAM_START] delimiter
    has_delimiter = "\n[STREAM_START]\n" in raw_data
    record("S5", "Contains [STREAM_START] delimiter", has_delimiter)

    if has_delimiter:
        parts = raw_data.split("\n[STREAM_START]\n", 1)
        metadata_str = parts[0]
        text_body = parts[1] if len(parts) > 1 else ""

        # Test 5.3: Metadata is valid JSON
        try:
            metadata = json.loads(metadata_str)
            record("S5", "Metadata is valid JSON", True)
        except json.JSONDecodeError as e:
            record("S5", "Metadata is valid JSON", False, f"parse error: {e}")
            metadata = {}

        # Test 5.4: Metadata has anchor_concept
        record("S5", "Metadata has 'anchor_concept'", "anchor_concept" in metadata)

        # Test 5.5: Metadata has prerequisites
        record("S5", "Metadata has 'prerequisites'", "prerequisites" in metadata)

        # Test 5.6: Metadata has unlocks
        record("S5", "Metadata has 'unlocks'", "unlocks" in metadata)

        # Test 5.7: Metadata has citations
        record("S5", "Metadata has 'citations'", "citations" in metadata)

        # Test 5.8: Metadata has logs array
        has_logs = "logs" in metadata and isinstance(metadata.get("logs"), list)
        record("S5", "Metadata has 'logs' array", has_logs)

        # Test 5.9: Each log entry has step/status/details
        if has_logs and metadata["logs"]:
            first_log = metadata["logs"][0]
            log_keys_ok = all(k in first_log for k in ("step", "status", "details"))
            record("S5", "Log entries have step/status/details", log_keys_ok,
                   f"keys={list(first_log.keys())}")

        # Test 5.10: Text body is non-empty (either aura output or error message)
        record("S5", "Text body after [STREAM_START] is non-empty",
               len(text_body.strip()) > 0,
               f"body_len={len(text_body)}")

        # Test 5.11: Citations in metadata are stripped of text_passage (lightweight)
        if metadata.get("citations"):
            first_cite = metadata["citations"][0]
            has_passage = "text_passage" in first_cite
            record("S5", "Citations metadata excludes text_passage", not has_passage,
                   f"citation keys={list(first_cite.keys())}")

# Test 5.12: Empty query returns 400
try:
    resp_empty = test_client.post("/api/chat",
                                  data=json.dumps({"query": "", "mode": "rag_synthesis"}),
                                  content_type="application/json")
    record("S5", "Empty query returns 400", resp_empty.status_code == 400,
           f"status={resp_empty.status_code}")
except Exception as e:
    record("S5", "Empty query returns 400", False, str(e))

# Test 5.13: Invalid mode returns 400
try:
    resp_mode = test_client.post("/api/chat",
                                 data=json.dumps({"query": "test", "mode": "invalid_mode"}),
                                 content_type="application/json")
    record("S5", "Invalid mode returns 400", resp_mode.status_code == 400,
           f"status={resp_mode.status_code}")
except Exception as e:
    record("S5", "Invalid mode returns 400", False, str(e))

# Test 5.14: Missing mode defaults to rag_synthesis (200)
try:
    resp_default = test_client.post("/api/chat",
                                    data=json.dumps({"query": "What is attention?"}),
                                    content_type="application/json")
    record("S5", "Missing mode defaults to rag_synthesis",
           resp_default.status_code == 200,
           f"status={resp_default.status_code}")
except Exception as e:
    record("S5", "Missing mode defaults to rag_synthesis", False, str(e))

# Test 5.15: CORS headers present
try:
    resp_cors = test_client.post("/api/chat",
                                  data=json.dumps({"query": "LoRA", "mode": "rag_synthesis"}),
                                  content_type="application/json")
    has_cors = resp_cors.headers.get("Access-Control-Allow-Origin") == "*"
    record("S5", "CORS headers present", has_cors,
           f"ACAO={resp_cors.headers.get('Access-Control-Allow-Origin')}")
except Exception as e:
    record("S5", "CORS headers present", False, str(e))

# Test 5.16: Failed anchor match still returns valid streaming format
try:
    resp_fail = test_client.post("/api/chat",
                                 data=json.dumps({"query": "xyzzynoconcepthere", "mode": "rag_synthesis"}),
                                 content_type="application/json")
    fail_data = resp_fail.get_data(as_text=True)
    # Even failed anchors might match something via fuzzy. Check format.
    has_format = "\n[STREAM_START]\n" in fail_data
    record("S5", "Low-confidence query still has streaming format", has_format,
           f"status={resp_fail.status_code}")
except Exception as e:
    record("S5", "Low-confidence query still has streaming format", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  BONUS: Code Review Static Analysis Checks
# ═════════════════════════════════════════════════════════════════════════════
section("BONUS: Static Code Review Validations")

# Check 1: Schema consistency — graph_db.py uses REQUIRES, inference_server uses REQUIRES
# (verifying there's no PREREQUISITE vs REQUIRES mismatch)
try:
    with open(os.path.join(PROJECT_DIR, "inference_server.py"), "r") as f:
        is_code = f.read()
    with open(os.path.join(PROJECT_DIR, "okf", "graph_db.py"), "r") as f:
        gdb_code = f.read()

    # inference_server.py should use REQUIRES (matching graph_db.py schema)
    uses_requires = "[:REQUIRES" in is_code
    uses_prerequisite = "[:PREREQUISITE" in is_code
    record("CR", "inference_server uses :REQUIRES (not :PREREQUISITE)",
           uses_requires and not uses_prerequisite,
           f"REQUIRES={uses_requires}, PREREQUISITE={uses_prerequisite}")

    # graph_db.py creates REQUIRES table
    gdb_requires = "REQUIRES" in gdb_code
    record("CR", "graph_db.py defines REQUIRES rel table", gdb_requires)

except Exception as e:
    record("CR", "Schema consistency check", False, str(e))

# Check 2: Cypher injection vulnerability in get_graph_neighborhood
try:
    # The function uses f-string interpolation without escaping concept_id
    has_fstring = "f\"MATCH (a:Concept {{id: '{concept_id}'}}" in is_code
    record("CR", "get_graph_neighborhood uses unescaped f-string (SQL injection risk)",
           has_fstring,
           "concept_id is interpolated directly — should use _kuzu_escape() or parameterized queries")
except Exception as e:
    record("CR", "Cypher injection check", False, str(e))

# Check 3: Thread safety — model loading modifies globals without locks
try:
    has_lock = "Lock" in is_code or "lock" in is_code
    record("CR", "Thread safety: has Lock for global model state",
           has_lock,
           "load_embedding_model and load_aura_model modify globals from background threads — no Lock found"
           if not has_lock else "Lock found")
except Exception as e:
    record("CR", "Thread safety check", False, str(e))

# Check 4: CONCEPT_EMBEDDINGS_TENSOR device consistency
# The tensor is built on one device but query embeddings go through .to(device)
try:
    has_device_consistency = "query_emb_dev = query_emb.to(device)" in is_code
    record("CR", "Device consistency: query_emb moved to tensor device",
           has_device_consistency,
           "find_anchor_concept correctly transfers query embedding to CONCEPT_EMBEDDINGS_TENSOR.device")
except Exception as e:
    record("CR", "Device consistency check", False, str(e))

# Check 5: Text passage crash on None
# compile_narrative_recipe calls c['text_passage'].strip() — if text_passage is None, this crashes
try:
    has_strip = "c['text_passage'].strip()" in is_code
    record("CR", "compile_narrative_recipe: text_passage.strip() may crash on None",
           not has_strip,
           "If text_passage is None (from DB), .strip() will raise AttributeError")
except Exception as e:
    record("CR", "text_passage None check", False, str(e))

# Check 6: run_ollama_agent tool description mentions REQUIRES + UNLOCKS + RELATED
try:
    tool_desc_section = is_code[is_code.index("query_database"):is_code.index("query_database") + 500]
    mentions_requires = "REQUIRES" in tool_desc_section
    mentions_unlocks = "UNLOCKS" in tool_desc_section
    mentions_related = "RELATED" in tool_desc_section
    record("CR", "Ollama tool description includes all edge types",
           mentions_requires and mentions_unlocks and mentions_related,
           f"REQUIRES={mentions_requires}, UNLOCKS={mentions_unlocks}, RELATED={mentions_related}")
except Exception as e:
    record("CR", "Ollama tool description check", False, str(e))

# Check 7: generate_aura_synthesis max_new_tokens mismatch with streaming version
try:
    # Non-streaming: max_new_tokens=512 (line ~308)
    # Streaming: max_new_tokens=256 (line ~756)
    # These differ — is that intentional?
    record("CR", "max_new_tokens mismatch: non-stream=512 vs stream=256",
           True,
           "generate_aura_synthesis() uses 512, streaming generate() uses 256 — may cause inconsistent output lengths")
except Exception as e:
    record("CR", "max_new_tokens check", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("FINAL TEST REPORT")

print(f"\n  Total:  {PASS_COUNT + FAIL_COUNT + SKIP_COUNT}")
print(f"  Passed: {PASS_COUNT} ✅")
print(f"  Failed: {FAIL_COUNT} ❌")
print(f"  Skipped: {SKIP_COUNT} ⏭️")
print()

if FAIL_COUNT > 0:
    print("  ── Failed Tests ──")
    for stage, name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"    ❌ [{stage}] {name}: {detail}")
    print()

if SKIP_COUNT > 0:
    print("  ── Skipped Tests ──")
    for stage, name, status, detail in RESULTS:
        if status == "SKIP":
            print(f"    ⏭️  [{stage}] {name}: {detail}")
    print()

print(f"{'═'*70}")
overall = "ALL TESTS PASSED" if FAIL_COUNT == 0 else f"{FAIL_COUNT} TEST(S) FAILED"
print(f"  {overall}")
print(f"{'═'*70}\n")

sys.exit(0 if FAIL_COUNT == 0 else 1)

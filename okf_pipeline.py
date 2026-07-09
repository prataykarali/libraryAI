#!/usr/bin/env python3
"""
Full Archipelago Pipeline: PDF → OKF v1.5 → Canonicalize → KùzuDB Graph RAG

Stages:
  1. Section-aware PDF chunking (pdf_ingestion.py)
  2. OKF v1.5 extraction via SLM (expanded schema)
  3. Entity canonicalization (alias resolution)
  4. KùzuDB MERGE ingestion (no duplicate nodes across documents)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

import ollama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "qwen3.5:0.8b"
BASE_DIR = Path(__file__).resolve().parent
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# OKF v1.5 Extraction Prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT_V15 = """You are a knowledge extraction engine. Extract structured knowledge from the text below.

Return ONLY valid JSON with these exact fields:
- concept_name: The main concept (short noun phrase, max 5 words)
- concept_type: One of: method, metric, technique, theory, tool, dataset, result, definition
- difficulty: One of: foundational, intermediate, advanced, expert
- summary: 1-2 sentence summary of what this concept is
- prerequisites: List of concepts needed BEFORE this one (list of short strings)
- unlocks: List of concepts enabled AFTER learning this (list of short strings)
- related_to: List of objects like {{"concept": "name", "relation": "type"}} where relation is one of: contrasts_with, uses, extends, evaluated_by, variant_of, part_of
- tags: List of keyword tags (lowercase, hyphenated)

Text:
{text}

Return ONLY the JSON object, no other text:"""


# ---------------------------------------------------------------------------
# SLM Extraction
# ---------------------------------------------------------------------------
def extract_okf_v15(text: str, doc_id: str = "", chunk_id: str = "",
                    page_number: int = 0, section_title: str = "") -> dict | None:
    """Extract OKF v1.5 data from a text chunk using Ollama."""
    # Truncate text to fit in context window (leave room for prompt + output)
    max_text_len = 2000
    if len(text) > max_text_len:
        text = text[:max_text_len]

    prompt = EXTRACTION_PROMPT_V15.format(text=text)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response_stream = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                think=False,
            )
            full_response = ""
            for chunk in response_stream:
                token = chunk["message"]["content"]
                full_response += token

            # Parse JSON from response
            cleaned = full_response.strip()
            # Strip markdown code fences
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Extract JSON object
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                cleaned = cleaned[first_brace:last_brace + 1]

            data = json.loads(cleaned)

            # Inject provenance metadata
            data["doc_id"] = doc_id
            data["chunk_id"] = chunk_id
            data["page_number"] = page_number
            data["section_title"] = section_title

            # Validate required fields exist
            if not data.get("concept_name"):
                if attempt < MAX_RETRIES:
                    continue
                return None

            # If concept_name is a list, take first element or join
            if isinstance(data.get("concept_name"), list):
                names = [n for n in data["concept_name"] if isinstance(n, str)]
                data["concept_name"] = names[0] if names else ""
                if not data["concept_name"]:
                    if attempt < MAX_RETRIES:
                        continue
                    return None

            # Normalize fields that might be wrong type
            if isinstance(data.get("prerequisites"), str):
                data["prerequisites"] = [data["prerequisites"]]
            if isinstance(data.get("unlocks"), str):
                data["unlocks"] = [data["unlocks"]]
            if not isinstance(data.get("prerequisites"), list):
                data["prerequisites"] = []
            if not isinstance(data.get("unlocks"), list):
                data["unlocks"] = []
            if not isinstance(data.get("related_to"), list):
                data["related_to"] = []
            if not isinstance(data.get("tags"), list):
                data["tags"] = []

            # Ensure concept_type and difficulty have valid values
            valid_types = {"method", "metric", "technique", "theory", "tool", "dataset", "result", "definition"}
            ctype = data.get("concept_type", "")
            if isinstance(ctype, list) and ctype:
                ctype = ctype[0]
            if not isinstance(ctype, str) or ctype.lower() not in valid_types:
                data["concept_type"] = "definition"
            else:
                data["concept_type"] = ctype.lower()

            valid_diff = {"foundational", "intermediate", "advanced", "expert"}
            diff = data.get("difficulty", "")
            if isinstance(diff, list) and diff:
                diff = diff[0]
            if not isinstance(diff, str) or diff.lower() not in valid_diff:
                data["difficulty"] = "intermediate"
            else:
                data["difficulty"] = diff.lower()

            return data

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                continue
            return None
        except Exception as exc:
            print(f"    Error: {exc}")
            if attempt < MAX_RETRIES:
                continue
            return None

    return None


# ---------------------------------------------------------------------------
# Entity Canonicalization
# ---------------------------------------------------------------------------

# Common aliases to collapse
ALIAS_MAP = {
    "3rd normal form": "3NF",
    "third normal form": "3NF",
    "natural language processing": "NLP",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "AI",
    "convolutional neural network": "CNN",
    "recurrent neural network": "RNN",
    "long short-term memory": "LSTM",
    "kl divergence": "KL Divergence",
    "kullback-leibler divergence": "KL Divergence",
    "kullback leibler divergence": "KL Divergence",
    "cross entropy": "Cross-Entropy",
    "cross-entropy loss": "Cross-Entropy",
    "lora": "LoRA",
    "low-rank adaptation": "LoRA",
    "low rank adaptation": "LoRA",
    "wasserstein distance": "Wasserstein Distance",
    "wasserstein-1": "Wasserstein Distance",
    "w1 distance": "Wasserstein Distance",
    "fine tuning": "Fine-Tuning",
    "fine-tuning": "Fine-Tuning",
    "finetuning": "Fine-Tuning",
    "calibration fine-tuning": "Calibration Fine-Tuning",
    "calibration fine tuning": "Calibration Fine-Tuning",
}


def canonicalize_name(name: str) -> str:
    """Normalize a concept name to a canonical form."""
    if not name:
        return ""

    # Strip whitespace
    name = name.strip()

    # Remove trailing periods
    name = name.rstrip(".")

    # Check alias map
    name_lower = name.lower()
    if name_lower in ALIAS_MAP:
        return ALIAS_MAP[name_lower]

    # Remove parenthetical abbreviations: "Natural Language Processing (NLP)" → "Natural Language Processing"
    name = re.sub(r'\s*\([^)]{1,10}\)\s*$', '', name)

    # If it's a full sentence (has a verb-like pattern), truncate
    if len(name) > 60:
        # Try to keep just the first noun phrase
        parts = name.split(",")
        name = parts[0].strip()
    if len(name) > 60:
        parts = name.split(" - ")
        name = parts[0].strip()

    # Title case
    name = name.strip()
    if name == name.lower() or name == name.upper():
        name = name.title()

    return name


def build_canonical_map(okf_results: list) -> dict:
    """
    Build a mapping from raw concept names → canonical names.
    Deduplicates similar concepts via fuzzy matching.
    """
    raw_names = set()
    for result in okf_results:
        cn = result.get("concept_name", "")
        if isinstance(cn, str) and cn:
            raw_names.add(cn)
        for p in result.get("prerequisites", []):
            if isinstance(p, str) and p:
                raw_names.add(p)
        for u in result.get("unlocks", []):
            if isinstance(u, str) and u:
                raw_names.add(u)
        for r in result.get("related_to", []):
            if isinstance(r, dict) and isinstance(r.get("concept"), str):
                raw_names.add(r.get("concept", ""))

    # Canonicalize all names
    canon_map = {}
    canonical_set = {}  # canonical_lower → canonical

    for raw in raw_names:
        if not raw:
            continue
        canon = canonicalize_name(raw)
        canon_lower = canon.lower()

        # Check for fuzzy matches in existing canonicals
        matched = False
        for existing_lower, existing_canon in canonical_set.items():
            # Substring match
            if (canon_lower in existing_lower or existing_lower in canon_lower) and \
               len(canon_lower) > 3 and len(existing_lower) > 3:
                # Use the shorter name as canonical
                if len(existing_canon) <= len(canon):
                    canon_map[raw] = existing_canon
                else:
                    # Update to use the shorter name
                    canon_map[raw] = canon
                    canonical_set[canon_lower] = canon
                    # Re-map anything that pointed to the old longer name
                    for k, v in canon_map.items():
                        if v == existing_canon:
                            canon_map[k] = canon
                matched = True
                break

        if not matched:
            canon_map[raw] = canon
            canonical_set[canon_lower] = canon

    return canon_map


def apply_canonicalization(okf_results: list, canon_map: dict) -> list:
    """Apply canonical name mapping to all concept references in OKF results."""
    for result in okf_results:
        raw_name = result.get("concept_name", "")
        result["concept_name"] = canon_map.get(raw_name, canonicalize_name(raw_name))

        result["prerequisites"] = [
            canon_map.get(p, canonicalize_name(p))
            for p in result.get("prerequisites", [])
            if isinstance(p, str) and p.strip()
        ]
        result["unlocks"] = [
            canon_map.get(u, canonicalize_name(u))
            for u in result.get("unlocks", [])
            if isinstance(u, str) and u.strip()
        ]

        new_related = []
        for r in result.get("related_to", []):
            if isinstance(r, dict) and r.get("concept"):
                r["concept"] = canon_map.get(r["concept"], canonicalize_name(r["concept"]))
                new_related.append(r)
        result["related_to"] = new_related

    return okf_results


# ---------------------------------------------------------------------------
# KùzuDB Graph Ingestion (MERGE semantics)
# ---------------------------------------------------------------------------
def create_concept_id(name: str) -> str:
    """Generate a stable, deterministic ID from a concept name."""
    cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
    cid = re.sub(r'_+', '_', cid).strip('_')
    return cid or 'concept'


def ingest_to_kuzu(okf_results: list, db_path: str = "okf_graph.db"):
    """Ingest OKF results into KùzuDB with MERGE semantics (no duplicate nodes)."""
    import kuzu
    import shutil

    # Clean existing DB
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            shutil.rmtree(db_path, ignore_errors=True)

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Create schema
    print("  Creating graph schema...")
    try:
        conn.execute("""
            CREATE NODE TABLE Concept (
                id STRING PRIMARY KEY,
                name STRING,
                concept_type STRING,
                difficulty STRING,
                summary STRING,
                sources STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE REQUIRES (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE UNLOCKS (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE RELATED (
                FROM Concept TO Concept,
                relation_type STRING,
                source STRING
            )
        """)
    except Exception:
        pass

    # Track all known concept IDs for MERGE semantics
    known_concepts = {}  # id → True

    def ensure_concept(name, concept_type="definition", difficulty="intermediate",
                       summary="", source_info=""):
        """MERGE-style: create node if it doesn't exist."""
        cid = create_concept_id(name)
        if cid in known_concepts:
            return cid

        # Escape single quotes
        safe_name = name.replace("'", "''")
        safe_summary = (summary or "").replace("'", "''")[:500]
        safe_source = (source_info or "").replace("'", "''")

        try:
            conn.execute(f"""
                CREATE (c:Concept {{
                    id: '{cid}',
                    name: '{safe_name}',
                    concept_type: '{concept_type}',
                    difficulty: '{difficulty}',
                    summary: '{safe_summary}',
                    sources: '{safe_source}'
                }})
            """)
            known_concepts[cid] = True
        except Exception:
            known_concepts[cid] = True  # Already exists
        return cid

    def create_edge(from_id, to_id, rel_table, rel_type, source):
        """Create a relationship edge."""
        if from_id == to_id:
            return False
        safe_source = source.replace("'", "''")
        try:
            conn.execute(f"""
                MATCH (a:Concept {{id: '{from_id}'}}),
                      (b:Concept {{id: '{to_id}'}})
                CREATE (a)-[:{rel_table} {{relation_type: '{rel_type}', source: '{safe_source}'}}]->(b)
            """)
            return True
        except Exception:
            return False

    # Ingest all concepts
    print("  Ingesting concept nodes...")
    node_count = 0
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        source_info = f"{result.get('doc_id', '')}:{result.get('chunk_id', '')}:p{result.get('page_number', 0)}"
        ensure_concept(
            name,
            result.get("concept_type", "definition"),
            result.get("difficulty", "intermediate"),
            result.get("summary", ""),
            source_info
        )
        node_count += 1

        # Also ensure prerequisite and unlock nodes exist (they might come from other chunks)
        for prereq in result.get("prerequisites", []):
            if isinstance(prereq, str) and prereq.strip():
                ensure_concept(prereq, source_info=source_info)
        for unlock in result.get("unlocks", []):
            if isinstance(unlock, str) and unlock.strip():
                ensure_concept(unlock, source_info=source_info)
        for rel in result.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                ensure_concept(rel["concept"], source_info=source_info)

    print(f"    -> {len(known_concepts)} unique concept nodes")

    # Create edges
    print("  Creating relationship edges...")
    edge_count = 0
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        concept_id = create_concept_id(name)
        source_info = f"{result.get('doc_id', '')}:{result.get('chunk_id', '')}"

        for prereq in result.get("prerequisites", []):
            if isinstance(prereq, str) and prereq.strip():
                prereq_id = create_concept_id(prereq)
                if create_edge(concept_id, prereq_id, "REQUIRES", "requires", source_info):
                    edge_count += 1

        for unlock in result.get("unlocks", []):
            if isinstance(unlock, str) and unlock.strip():
                unlock_id = create_concept_id(unlock)
                if create_edge(concept_id, unlock_id, "UNLOCKS", "enables", source_info):
                    edge_count += 1

        for rel in result.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                rel_id = create_concept_id(rel["concept"])
                rel_type = rel.get("relation", "related")
                if create_edge(concept_id, rel_id, "RELATED", rel_type, source_info):
                    edge_count += 1

    print(f"    -> {edge_count} edges created")

    # Export graph to JSON
    print("  Exporting graph...")
    export = export_graph(conn)

    return conn, db, export


def export_graph(conn) -> dict:
    """Export the full graph structure to a dict."""
    # Get concepts
    result = conn.execute("""
        MATCH (c:Concept)
        RETURN c.id, c.name, c.concept_type, c.difficulty, c.summary, c.sources
        ORDER BY c.name
    """)
    concepts = {}
    while result.has_next():
        row = result.get_next()
        concepts[row[0]] = {
            "name": row[1],
            "concept_type": row[2],
            "difficulty": row[3],
            "summary": row[4],
            "sources": row[5]
        }

    # Get all edges
    edges = []
    for rel_table in ["REQUIRES", "UNLOCKS", "RELATED"]:
        try:
            result = conn.execute(f"""
                MATCH (a:Concept)-[r:{rel_table}]->(b:Concept)
                RETURN a.id, a.name, r.relation_type, b.id, b.name
            """)
            while result.has_next():
                row = result.get_next()
                edges.append({
                    "from_id": row[0],
                    "from_name": row[1],
                    "relation": row[2],
                    "to_id": row[3],
                    "to_name": row[4],
                    "edge_type": rel_table
                })
        except Exception:
            pass

    return {
        "concepts": concepts,
        "edges": edges,
        "stats": {
            "total_concepts": len(concepts),
            "total_edges": len(edges),
            "requires_edges": sum(1 for e in edges if e["edge_type"] == "REQUIRES"),
            "unlocks_edges": sum(1 for e in edges if e["edge_type"] == "UNLOCKS"),
            "related_edges": sum(1 for e in edges if e["edge_type"] == "RELATED"),
        }
    }


# ---------------------------------------------------------------------------
# Accuracy / Evaluation
# ---------------------------------------------------------------------------
def evaluate_extraction(okf_results: list, total_chunks: int, graph_export: dict,
                        raw_extraction_count: int = 0) -> dict:
    """Compute proxy accuracy metrics for the extraction pipeline."""
    if raw_extraction_count == 0:
        raw_extraction_count = len(okf_results)

    # 1. Extraction Rate (raw, before cleanup)
    extraction_rate = (raw_extraction_count / total_chunks * 100) if total_chunks > 0 else 0

    # 2. Schema Completeness — do all results have all required fields?
    required_fields = ["concept_name", "summary", "prerequisites", "unlocks"]
    expanded_fields = ["concept_type", "difficulty", "related_to", "tags"]
    complete_core = 0
    complete_expanded = 0
    for r in okf_results:
        if all(r.get(f) for f in required_fields):
            complete_core += 1
        if all(r.get(f) is not None for f in required_fields + expanded_fields):
            complete_expanded += 1

    schema_completeness_core = (complete_core / len(okf_results) * 100) if okf_results else 0
    schema_completeness_full = (complete_expanded / len(okf_results) * 100) if okf_results else 0

    # 3. Concept Quality — are names reasonable?
    good_names = 0
    for r in okf_results:
        name = r.get("concept_name", "")
        if name and len(name) < 60 and len(name.split()) <= 8 and not name.endswith("."):
            good_names += 1
    concept_quality = (good_names / len(okf_results) * 100) if okf_results else 0

    # 4. Relation Consistency — do prereqs/unlocks point to known graph nodes?
    all_concept_names = {r.get("concept_name", "").lower() for r in okf_results}
    total_refs = 0
    resolved_refs = 0
    for r in okf_results:
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p:
                total_refs += 1
                if p.lower() in all_concept_names:
                    resolved_refs += 1
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u:
                total_refs += 1
                if u.lower() in all_concept_names:
                    resolved_refs += 1

    relation_consistency = (resolved_refs / total_refs * 100) if total_refs > 0 else 0

    # 5. DAG Validity — check for self-loops (cycle detection is expensive)
    self_loops = 0
    for r in okf_results:
        name = r.get("concept_name", "").lower()
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p.lower() == name:
                self_loops += 1
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u.lower() == name:
                self_loops += 1
    dag_validity = 100.0 if self_loops == 0 else max(0, 100 - self_loops * 10)

    # 6. Type Distribution — how well does the model differentiate types?
    type_counts = Counter(r.get("concept_type", "unknown") for r in okf_results)
    type_diversity = len(type_counts)

    # 7. Difficulty Distribution
    diff_counts = Counter(r.get("difficulty", "unknown") for r in okf_results)

    # 8. Graph connectivity
    total_concepts = graph_export["stats"]["total_concepts"]
    total_edges = graph_export["stats"]["total_edges"]
    edge_density = (total_edges / total_concepts) if total_concepts > 0 else 0

    # Orphan nodes (no edges at all)
    connected_ids = set()
    for e in graph_export["edges"]:
        connected_ids.add(e["from_id"])
        connected_ids.add(e["to_id"])
    orphan_count = total_concepts - len(connected_ids)
    connectivity = ((total_concepts - orphan_count) / total_concepts * 100) if total_concepts > 0 else 0

    # Composite Score (weighted — uses extraction_rate, not post-cleanup count)
    composite = (
        extraction_rate * 0.15 +
        schema_completeness_core * 0.15 +
        concept_quality * 0.15 +
        relation_consistency * 0.15 +
        dag_validity * 0.15 +
        connectivity * 0.25
    )

    return {
        "overall_score": round(composite, 1),
        "breakdown": {
            "extraction_rate": round(extraction_rate, 1),
            "schema_completeness_core": round(schema_completeness_core, 1),
            "schema_completeness_expanded": round(schema_completeness_full, 1),
            "concept_quality": round(concept_quality, 1),
            "relation_consistency": round(relation_consistency, 1),
            "dag_validity": round(dag_validity, 1),
            "connectivity": round(connectivity, 1),
            "edge_density": round(edge_density, 2),
        },
        "distributions": {
            "concept_types": dict(type_counts),
            "difficulty_levels": dict(diff_counts),
        },
        "stats": {
            "total_chunks": total_chunks,
            "raw_extractions": raw_extraction_count,
            "after_cleanup": len(okf_results),
            "failed_extractions": total_chunks - raw_extraction_count,
            "cleaned_out": raw_extraction_count - len(okf_results),
            "total_concepts_in_graph": total_concepts,
            "total_edges_in_graph": total_edges,
            "orphan_nodes": orphan_count,
            "self_loops": self_loops,
        }
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path: str = None, resume: bool = False):
    """Run the full Archipelago pipeline."""
    from pdf_ingestion import ingest_folder, ingest_document

    print("=" * 70)
    print("ARCHIPELAGO PIPELINE - PDF -> OKF v1.5 -> KuzuDB Graph RAG")
    print("=" * 70)

    # ── Stage 1: Chunking ──
    print("\n[1] STAGE 1: Section-Aware Document Chunking")
    print("-" * 50)

    if input_path is None:
        input_path = str(BASE_DIR / "pdfs")

    if os.path.isdir(input_path):
        chunks = ingest_folder(input_path)
    else:
        chunks = ingest_document(input_path)

    print(f"\n  Total chunks: {len(chunks)}")
    total_chunks = len(chunks)

    if not chunks:
        print("ERROR: No chunks extracted!")
        return

    # ── Stage 2: OKF v1.5 Extraction ──
    print(f"\n[2] STAGE 2: OKF v1.5 Extraction via {MODEL_NAME}")
    print("-" * 50)

    saved_file = BASE_DIR / "okf_results.json"
    if resume and saved_file.exists():
        print("  RESUMING from saved okf_results.json...")
        with open(saved_file, "r", encoding="utf-8") as f:
            okf_results = json.load(f)
        # Fix any list-type concept_names from previous runs
        fixed = []
        for r in okf_results:
            cn = r.get("concept_name", "")
            if isinstance(cn, list):
                names = [n for n in cn if isinstance(n, str)]
                r["concept_name"] = names[0] if names else ""
            if r.get("concept_name"):
                fixed.append(r)
        okf_results = fixed
        print(f"  Loaded {len(okf_results)} results (fixed list-type names)")
    else:
        okf_results = []
        for i, chunk in enumerate(chunks):
            progress = f"[{i+1}/{len(chunks)}]"
            section = chunk.get("section_title", "?")[:40]
            print(f"  {progress} {chunk['doc_id']} | {section} (p.{chunk['page_number']})", end="")
            sys.stdout.flush()

            start_time = time.time()
            result = extract_okf_v15(
                text=chunk["text"],
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page_number"],
                section_title=chunk["section_title"]
            )
            elapsed = time.time() - start_time

            if result:
                okf_results.append(result)
                print(f" -> {result['concept_name'][:35]} ({elapsed:.1f}s)")
            else:
                print(f" -> FAILED ({elapsed:.1f}s)")

        print(f"\n  Extracted: {len(okf_results)}/{total_chunks} chunks")

    # Save raw extraction results
    with open("okf_results.json", "w", encoding="utf-8") as f:
        json.dump(okf_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to okf_results.json")

    # -- Post-extraction cleanup --
    print("\n[2b] POST-EXTRACTION CLEANUP")
    print("-" * 50)
    pre_cleanup = len(okf_results)

    # 1. Remove reference/bibliography section extractions (noise)
    okf_results = [r for r in okf_results
                   if not r.get("section_title", "").lower().startswith("reference")]
    ref_removed = pre_cleanup - len(okf_results)
    print(f"  Removed {ref_removed} reference-section concepts")

    # 2. Remove self-loops (concept listing itself as prerequisite/unlock)
    loop_count = 0
    for r in okf_results:
        name_lower = r.get("concept_name", "").lower()
        old_prereqs = r.get("prerequisites", [])
        old_unlocks = r.get("unlocks", [])
        r["prerequisites"] = [p for p in old_prereqs
                              if isinstance(p, str) and p.lower() != name_lower]
        r["unlocks"] = [u for u in old_unlocks
                        if isinstance(u, str) and u.lower() != name_lower]
        loop_count += (len(old_prereqs) - len(r["prerequisites"])) + \
                      (len(old_unlocks) - len(r["unlocks"]))
    print(f"  Removed {loop_count} self-referential loops")

    # 3. Merge duplicate concept names (keep richest version)
    seen = {}
    merged_results = []
    for r in okf_results:
        key = r.get("concept_name", "").lower().replace("-", " ").replace("_", " ").strip()
        if key in seen:
            existing = seen[key]
            if len(r.get("summary", "")) > len(existing.get("summary", "")):
                existing["summary"] = r["summary"]
            existing["prerequisites"] = list(set(
                existing.get("prerequisites", []) + r.get("prerequisites", [])))
            existing["unlocks"] = list(set(
                existing.get("unlocks", []) + r.get("unlocks", [])))
            existing["tags"] = list(set(
                existing.get("tags", []) + r.get("tags", [])))
            existing_rels = {(x.get("concept", ""), x.get("relation", ""))
                            for x in existing.get("related_to", []) if isinstance(x, dict)}
            for rel in r.get("related_to", []):
                if isinstance(rel, dict):
                    key_rel = (rel.get("concept", ""), rel.get("relation", ""))
                    if key_rel not in existing_rels:
                        existing["related_to"].append(rel)
                        existing_rels.add(key_rel)
        else:
            seen[key] = r
            merged_results.append(r)
    dupe_removed = len(okf_results) - len(merged_results)
    okf_results = merged_results
    print(f"  Merged {dupe_removed} duplicate concept entries")

    # 4. Remove concepts with very short names (likely noise)
    pre_filter = len(okf_results)
    okf_results = [r for r in okf_results
                   if len(r.get("concept_name", "")) >= 3]
    noise_removed = pre_filter - len(okf_results)
    if noise_removed > 0:
        print(f"  Removed {noise_removed} too-short concept names")

    print(f"  Final: {len(okf_results)} clean concepts (was {pre_cleanup})")

    # -- Stage 3: Canonicalization --
    print(f"\n[3] STAGE 3: Entity Canonicalization")
    print("-" * 50)

    canon_map = build_canonical_map(okf_results)
    okf_results = apply_canonicalization(okf_results, canon_map)

    # Count dedup stats
    raw_concepts = len(canon_map)
    unique_concepts = len(set(canon_map.values()))
    print(f"  Raw concept mentions: {raw_concepts}")
    print(f"  Canonical concepts: {unique_concepts}")
    print(f"  Aliases resolved: {raw_concepts - unique_concepts}")

    # Post-canonicalization self-loop removal
    # (canonicalization can create new loops when concept + prereq map to same name)
    post_canon_loops = 0
    for r in okf_results:
        name_lower = r.get("concept_name", "").lower()
        old_p = r.get("prerequisites", [])
        old_u = r.get("unlocks", [])
        r["prerequisites"] = [p for p in old_p
                              if isinstance(p, str) and p.lower() != name_lower]
        r["unlocks"] = [u for u in old_u
                        if isinstance(u, str) and u.lower() != name_lower]
        post_canon_loops += (len(old_p) - len(r["prerequisites"])) + \
                            (len(old_u) - len(r["unlocks"]))
    if post_canon_loops > 0:
        print(f"  Removed {post_canon_loops} post-canonicalization self-loops")

    # ── Stage 4: KùzuDB Graph Ingestion ──
    print(f"\n[4] STAGE 4: KuzuDB Graph Ingestion (MERGE)")
    print("-" * 50)

    conn, db, graph_export = ingest_to_kuzu(okf_results)

    # Save graph export
    with open("okf_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_export, f, indent=2, ensure_ascii=False)
    print(f"  Saved to okf_graph.json")

    # ── Stage 5: Evaluation ──
    print(f"\n[5] STAGE 5: Accuracy Evaluation")
    print("-" * 50)

    accuracy = evaluate_extraction(okf_results, total_chunks, graph_export,
                                    raw_extraction_count=pre_cleanup)

    with open("accuracy.json", "w", encoding="utf-8") as f:
        json.dump(accuracy, f, indent=2)
    print(f"  Saved to accuracy.json")

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  >> Overall Accuracy Score: {accuracy['overall_score']}%")
    print(f"\n  Breakdown:")
    for metric, value in accuracy["breakdown"].items():
        bar = "#" * int(value / 5) + "." * (20 - int(value / 5))
        print(f"    {metric:30s} {bar} {value}%")

    print(f"\n  Concept Type Distribution:")
    for t, count in accuracy["distributions"]["concept_types"].items():
        print(f"    {t:20s}: {count}")

    print(f"\n  Difficulty Distribution:")
    for d, count in accuracy["distributions"]["difficulty_levels"].items():
        print(f"    {d:20s}: {count}")

    print(f"\n  Graph Stats:")
    for k, v in accuracy["stats"].items():
        print(f"    {k:30s}: {v}")

    # Print some sample concept mappings
    print(f"\n{'=' * 70}")
    print(f"SAMPLE EXTRACTED CONCEPTS")
    print(f"{'=' * 70}")
    for result in okf_results[:10]:
        print(f"\n  [{result.get('concept_type', '?'):10s}] {result['concept_name']}")
        print(f"    Difficulty: {result.get('difficulty', '?')}")
        print(f"    Summary: {result.get('summary', '')[:80]}...")
        if result.get("prerequisites"):
            print(f"    Requires: {', '.join(result['prerequisites'][:5])}")
        if result.get("unlocks"):
            print(f"    Unlocks: {', '.join(result['unlocks'][:5])}")
        if result.get("related_to"):
            for rel in result["related_to"][:3]:
                if isinstance(rel, dict):
                    print(f"    {rel.get('relation', '?'):15s} -> {rel.get('concept', '?')}")
        if result.get("tags"):
            print(f"    Tags: {', '.join(result['tags'][:5])}")

    print(f"\n{'=' * 70}")
    print(f"GRAPH EDGES (sample)")
    print(f"{'=' * 70}")
    for edge in graph_export["edges"][:20]:
        arrow = "--requires-->" if edge["edge_type"] == "REQUIRES" else \
                "--unlocks--->" if edge["edge_type"] == "UNLOCKS" else \
                f"--{edge['relation']:10s}->"
        print(f"  {edge['from_name'][:30]:30s} {arrow} {edge['to_name'][:30]}")

    print(f"\n{'=' * 70}")
    print(f"[OK] Pipeline complete! Files: okf_results.json, okf_graph.json, accuracy.json")
    print(f"{'=' * 70}")

    return okf_results, graph_export, accuracy


if __name__ == "__main__":
    args = sys.argv[1:]
    resume_mode = "--resume" in args
    args = [a for a in args if a != "--resume"]
    input_path = args[0] if args else None
    run_pipeline(input_path, resume=resume_mode)

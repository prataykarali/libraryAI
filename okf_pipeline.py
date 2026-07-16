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

try:
    from thefuzz import fuzz as _fuzz
except Exception:  # pragma: no cover - optional dependency
    _fuzz = None

# ---------------------------------------------------------------------------
# Direction / cycle helpers
# ---------------------------------------------------------------------------
# Optional domain-specific partial ordering. Used only to resolve direct A↔B
# prerequisite cycles when both concepts are in the dict. Leave empty for a
# fully generic pipeline; populate via a domain config file if desired.
_FOUNDATIONAL_PRIORITY = {}


_DIFFICULTY_RANK = {"foundational": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def break_global_cycles(okf_results: list) -> int:
    """Break all cycles in the learning progression graph (prerequisites and unlocks).
    
    Enforces a strict Directed Acyclic Graph (DAG) for learning paths.
    """
    import sys
    # Increase recursion limit just in case
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
    
    index = {r.get("concept_name", "").lower(): r for r in okf_results}
    total_removed = 0
    
    for _ in range(1000):  # limit iterations to prevent infinite loop
        # Build learning edges list (src -> dst means src must be learned before dst)
        edges = []
        for r in okf_results:
            u = r.get("concept_name", "")
            u_lower = u.lower()
            if not u_lower:
                continue
            
            # Prerequisites: p -> u (p is learned before u)
            for p in r.get("prerequisites", []):
                p_lower = p.lower()
                if p_lower and p_lower != u_lower:
                    edges.append((p_lower, u_lower, "prereq", p, u))
                    
            # Unlocks: u -> val (u unlocks val, so u is learned before val)
            for val in r.get("unlocks", []):
                val_lower = val.lower()
                if val_lower and val_lower != u_lower:
                    edges.append((u_lower, val_lower, "unlock", u, val))
        
        # Build adjacency list
        adj = {}
        edge_lookup = {}
        for src, dst, etype, orig_src, orig_dst in edges:
            adj.setdefault(src, set()).add(dst)
            edge_lookup[(src, dst)] = (etype, orig_src, orig_dst)
            
        # DFS cycle detection
        state = {}  # node -> 0: unvisited, 1: visiting, 2: visited
        parent = {}
        cycle_found = None
        
        def dfs(node):
            nonlocal cycle_found
            if cycle_found:
                return
            state[node] = 1
            for neighbor in adj.get(node, []):
                if state.get(neighbor, 0) == 1:
                    # Cycle detected! Reconstruct cycle path
                    cycle = []
                    curr = node
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent.get(curr)
                    cycle.append(neighbor)
                    cycle.reverse()
                    cycle.append(neighbor)
                    cycle_found = cycle
                    return
                elif state.get(neighbor, 0) == 0:
                    parent[neighbor] = node
                    dfs(neighbor)
                    if cycle_found:
                        return
            state[node] = 2
            
        all_nodes = set(index.keys()) | set(adj.keys())
        for node in all_nodes:
            if state.get(node, 0) == 0:
                dfs(node)
                if cycle_found:
                    break
                    
        if not cycle_found:
            break
            
        # We have a cycle path like [A, B, C, A]
        cycle_pairs = []
        for i in range(len(cycle_found) - 1):
            cycle_pairs.append((cycle_found[i], cycle_found[i+1]))
            
        # Find weakest edge in the cycle:
        # Score = diff(dst) - diff(src). Lower score = weaker/more likely incorrect.
        weakest_edge = None
        min_score = None
        
        for src, dst in cycle_pairs:
            src_res = index.get(src)
            dst_res = index.get(dst)
            
            src_diff = _DIFFICULTY_RANK.get(src_res.get("difficulty", "intermediate") if src_res else "intermediate", 2)
            dst_diff = _DIFFICULTY_RANK.get(dst_res.get("difficulty", "intermediate") if dst_res else "intermediate", 2)
            
            score = dst_diff - src_diff
            
            # Tie-breaker: score, -len(src), -len(dst)
            edge_score = (score, -len(src), -len(dst))
            
            if min_score is None or edge_score < min_score:
                min_score = edge_score
                weakest_edge = (src, dst)
                
        if weakest_edge:
            src, dst = weakest_edge
            etype, orig_src, orig_dst = edge_lookup[(src, dst)]
            
            if etype == "prereq":
                dst_res = index.get(dst)
                if dst_res:
                    dst_res["prerequisites"] = [x for x in dst_res.get("prerequisites", []) if x.lower() != src]
            else:
                src_res = index.get(src)
                if src_res:
                    src_res["unlocks"] = [x for x in src_res.get("unlocks", []) if x.lower() != dst]
                
            total_removed += 1
            
    return total_removed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Default model can be overridden with the OKF_MODEL_NAME env var so the
# pipeline stays model-agnostic.
MODEL_NAME = os.environ.get("OKF_MODEL_NAME", "qwen3.5:0.8b")
BASE_DIR = Path(__file__).resolve().parent
MAX_RETRIES = 1

# Local PyTorch model configuration (aura-qwen fine-tuned)
LOCAL_MODEL = None
LOCAL_TOKENIZER = None
_local_path = Path(__file__).resolve().parent.parent / "aura-qwen"
if not _local_path.exists():
    _local_path = Path("/home/pratay-karali/Desktop/libraryAI/aura-qwen")
LOCAL_MODE = _local_path.exists()

# Optional UNIFORM page cap applied to every PDF (not per-file). None = read the
# whole document. Override with the --max-pages CLI flag for quick test runs on
# very large books; production ingestion leaves it at None so any doc works.
MAX_PAGES_PER_DOC = None

# We NEVER feed a whole page/book to the SLM. Ingestion produces section- and
# paragraph-scale chunks; each SLM call is additionally capped to this many
# characters (~one to three paragraphs) so context stays small and focused.
MAX_CHARS_TO_SLM = 1800

# ---------------------------------------------------------------------------
# OKF v1.6 Extraction Prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT_V15 = """You are an OKF extraction engine for the Archipelago knowledge graph.
From the TEXT below, extract 1 to 5 teachable CONCEPTS as a JSON array.

Each object MUST have exactly these keys:
- concept_name: reusable noun phrase, MAX 5 words, Title Case, NO underscores (e.g. "Scientific Method", not "scientific_method")
- concept_type: one of: method, metric, technique, theory, tool, dataset, result, definition
- difficulty: one of: foundational, intermediate, advanced, expert
- summary: 1-2 sentences describing what the concept IS (not "this paper shows...")
- prerequisites: concepts a learner needs FIRST (short Title Case names) -> these become REQUIRES edges
- unlocks: concepts this ENABLES next (short Title Case names) -> these become UNLOCKS edges
- related_to: objects {{"concept": "Name", "relation": "type"}} where relation is one of: contrasts_with, uses, extends, evaluated_by, variant_of, part_of
- tags: lowercase-hyphenated keyword tags

Rules:
- Only concepts actually explained in the text. No authors, citations, section titles, or table numbers.
- ALWAYS try to fill prerequisites and unlocks - they are the whole point of the graph.
- A concept must NEVER appear in its own prerequisites or unlocks (no self-loops).
- Keep names stable across documents so the same concept merges into one node.
- If the text has no real teachable concept, return [].

EXAMPLE
Text: "The scientific method is a procedure for acquiring knowledge: it formulates questions, tests hypotheses through repeatable experiments, and revises theories based on evidence. Peer review then validates the findings before publication."
Output:
[
  {{"concept_name": "Scientific Method", "concept_type": "method", "difficulty": "intermediate", "summary": "A systematic procedure for acquiring knowledge by formulating questions, testing hypotheses through experiments, and revising theories based on evidence.", "prerequisites": ["Hypothesis", "Experimentation"], "unlocks": ["Peer Review", "Theory Building"], "related_to": [{{"concept": "Empirical Evidence", "relation": "uses"}}], "tags": ["research", "methodology"]}},
  {{"concept_name": "Peer Review", "concept_type": "technique", "difficulty": "intermediate", "summary": "A validation process in which independent experts evaluate a study's methods, results, and conclusions before publication.", "prerequisites": ["Scientific Method"], "unlocks": ["Published Research"], "related_to": [{{"concept": "Scientific Method", "relation": "evaluated_by"}}], "tags": ["research", "validation"]}}
]

TEXT:
{text}

Return ONLY the JSON array, no other text:"""


# ---------------------------------------------------------------------------
# SLM Extraction
# ---------------------------------------------------------------------------
VALID_TYPES = {"method", "metric", "technique", "theory", "tool", "dataset", "result", "definition"}
VALID_DIFFICULTIES = {"foundational", "intermediate", "advanced", "expert"}
VALID_RELATIONS = {"contrasts_with", "uses", "extends", "evaluated_by", "variant_of", "part_of"}


def infer_source_category(doc_id: str) -> str:
    """Infer a coarse source category from the organized ingestion path."""
    normalized = (doc_id or "").replace("\\", "/").lower()
    if normalized.startswith("textbooks/"):
        return "textbook"
    if normalized.startswith("papers/"):
        return "paper"
    if normalized.startswith("web_syllabi/"):
        return "web_syllabus"
    if normalized.endswith(".pdf"):
        return "pdf"
    if normalized.endswith((".md", ".markdown")):
        return "markdown"
    if normalized.endswith((".txt", ".text")):
        return "text"
    return "unknown"


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_json_payload(raw: str) -> str:
    """Extract the first JSON array/object from a chatty model response."""
    cleaned = _strip_json_fences(raw)
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if array_start != -1 and array_end > array_start:
        if object_start == -1 or array_start < object_start:
            return cleaned[array_start:array_end + 1]
    if object_start != -1 and object_end > object_start:
        return cleaned[object_start:object_end + 1]
    return cleaned


def _string_list(value) -> list:
    """Normalize scalar/nested list model output into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if isinstance(item, list):
            items.extend(str(x).strip() for x in item if isinstance(x, str) and x.strip())
        elif isinstance(item, str) and item.strip():
            items.append(item.strip())
    return list(dict.fromkeys(items))


def _normalize_related(value) -> list:
    if not isinstance(value, list):
        return []

    related = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            concept = item.strip()
            relation = "uses"
        elif isinstance(item, dict):
            concept = str(item.get("concept", "")).strip()
            relation = str(item.get("relation", "uses")).strip().lower()
        else:
            continue
        if not concept:
            continue
        if relation not in VALID_RELATIONS:
            relation = "uses"
        key = (concept.lower(), relation)
        if key not in seen:
            related.append({"concept": concept, "relation": relation})
            seen.add(key)
    return related


def normalize_okf_item(data: dict, doc_id: str, chunk_id: str,
                       page_number: int, section_title: str) -> dict | None:
    """Validate and normalize one OKF concept object with provenance."""
    if not isinstance(data, dict):
        return None

    concept_name = data.get("concept_name", "")
    if isinstance(concept_name, list):
        names = [n.strip() for n in concept_name if isinstance(n, str) and n.strip()]
        concept_name = names[0] if names else ""
    if not isinstance(concept_name, str) or not concept_name.strip():
        return None
    concept_name = concept_name.strip()
    if not is_valid_concept_name(concept_name):
        return None

    ctype = data.get("concept_type", "definition")
    if isinstance(ctype, list) and ctype:
        ctype = ctype[0]
    ctype = str(ctype).lower() if ctype else "definition"
    if ctype not in VALID_TYPES:
        ctype = "definition"

    difficulty = data.get("difficulty", "intermediate")
    if isinstance(difficulty, list) and difficulty:
        difficulty = difficulty[0]
    difficulty = str(difficulty).lower() if difficulty else "intermediate"
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "intermediate"

    summary = data.get("summary", "")
    if isinstance(summary, list):
        summary = " ".join(str(x).strip() for x in summary if str(x).strip())
    summary = str(summary).strip()

    name_lower = concept_name.lower()
    prerequisites = [p for p in _string_list(data.get("prerequisites")) if p.lower() != name_lower]
    unlocks = [u for u in _string_list(data.get("unlocks")) if u.lower() != name_lower]
    tags = [t.lower().replace(" ", "-") for t in _string_list(data.get("tags"))]

    return {
        "concept_name": concept_name,
        "concept_type": ctype,
        "difficulty": difficulty,
        "summary": summary,
        "prerequisites": prerequisites,
        "unlocks": unlocks,
        "related_to": _normalize_related(data.get("related_to")),
        "tags": list(dict.fromkeys(tags)),
        "doc_id": doc_id,
        "source_category": infer_source_category(doc_id),
        "chunk_id": chunk_id,
        "page_number": page_number,
        "section_title": section_title,
    }


def load_local_model():
    """Load the local model and tokenizer directly from disk."""
    global LOCAL_MODEL, LOCAL_TOKENIZER, LOCAL_MODE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = BASE_DIR.parent / "aura-qwen"
    if not local_path.exists():
        local_path = Path("/home/pratay-karali/Desktop/libraryAI/aura-qwen")

    print(f"  Loading local model from {local_path}...")
    try:
        LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True, fix_mistral_regex=True
        )
        LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Local model loaded successfully!")
        LOCAL_MODE = True
    except Exception as exc:
        print(f"  Error loading local model: {exc}")
        print("  ⚠️ Falling back to Ollama mode...")
        LOCAL_MODE = False


def extract_okf_v15(text: str, doc_id: str = "", chunk_id: str = "",
                    page_number: int = 0, section_title: str = "") -> list:
    """Extract OKF v1.6 concepts from a text chunk using Ollama or local model.

    We never send a whole page/book: the chunk is already section/paragraph
    scale, and we truncate to MAX_CHARS_TO_SLM at a paragraph/sentence boundary
    so the SLM only reasons over a small, coherent span.
    """
    if len(text) > MAX_CHARS_TO_SLM:
        window = text[:MAX_CHARS_TO_SLM]
        # Prefer to cut at the last paragraph break, else the last sentence end.
        cut = window.rfind("\n\n")
        if cut < MAX_CHARS_TO_SLM // 2:
            cut = max(window.rfind(". "), window.rfind("\n"))
        text = window[:cut] if cut > MAX_CHARS_TO_SLM // 2 else window

    prompt = EXTRACTION_PROMPT_V15.format(text=text)

    for attempt in range(MAX_RETRIES + 1):
        try:
            if LOCAL_MODE:
                # Local PyTorch/Transformers inference
                messages = [{"role": "user", "content": prompt}]
                try:
                    templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception:
                    templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                inputs = LOCAL_TOKENIZER(templated_prompt, return_tensors="pt").to(LOCAL_MODEL.device)

                import torch
                with torch.no_grad():
                    outputs = LOCAL_MODEL.generate(
                        **inputs,
                        max_new_tokens=768,
                        temperature=0.1,
                        do_sample=False,
                    )
                input_len = inputs.input_ids.shape[1]
                full_response = LOCAL_TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            else:
                # Ollama inference
                response_stream = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    think=False,
                    # num_predict caps output so a rambling chunk fails in ~3s
                    # instead of generating until the context window fills (~150s).
                    options={"temperature": 0.1, "top_p": 0.9, "num_predict": 768},
                )
                full_response = ""
                for chunk in response_stream:
                    token = chunk["message"]["content"]
                    full_response += token

            cleaned = _extract_json_payload(full_response)
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                if attempt < MAX_RETRIES:
                    continue
                return []

            normalized = [
                item for item in (
                    normalize_okf_item(x, doc_id, chunk_id, page_number, section_title)
                    for x in data[:5]
                )
                if item
            ]
            if normalized:
                return normalized
            if attempt < MAX_RETRIES:
                continue
            return []

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                continue
            return []
        except Exception as exc:
            print(f"    Error: {exc}")
            if attempt < MAX_RETRIES:
                continue
            return []

    return []


# ---------------------------------------------------------------------------
# Second-Pass Relation Extraction
# ---------------------------------------------------------------------------
# The fine-tuned extractor almost never fills prerequisites/unlocks/related_to
# in-line (13/1024 records). This focused second pass runs AFTER cleanup and
# canonicalization: for each surviving record we re-show the model its own
# passage plus the OTHER canonical concepts that literally appear in that
# passage, and ask ONLY for relations among them. Because relations are
# written back onto the asserting record, edge provenance is that record's
# (doc_id, chunk_id) by construction, and because targets outside the
# candidate list are rejected, no hallucinated placeholder targets can enter.
RELATION_PROMPT = """You are building a learning knowledge graph. Below is a PASSAGE from a document, a MAIN CONCEPT explained in it, and a list of CANDIDATE concepts that also appear in the passage.

PASSAGE:
{passage}

MAIN CONCEPT: {concept}

CANDIDATES: {candidates}

Based ONLY on what the passage says, classify how each candidate relates to the main concept "{concept}". Use these buckets:
- "prerequisites": candidates a learner must understand BEFORE the main concept
- "unlocks": candidates that the main concept ENABLES or leads to next
- "related_to": other real relations, as objects {{"concept": "...", "relation": "..."}} with relation one of: contrasts_with, uses, extends, evaluated_by, variant_of, part_of

Rules:
- Only use concept names copied EXACTLY from the CANDIDATES list.
- Only include a candidate if the passage actually supports the relation. Leave lists empty if nothing applies — empty lists are a good answer.
- Never include "{concept}" itself.

Return ONLY a JSON object:
{{"prerequisites": [], "unlocks": [], "related_to": []}}"""


def _generate_local(prompt: str, max_new_tokens: int = 512) -> str:
    """One local-model generation with the same chat templating as extraction."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = LOCAL_TOKENIZER(templated_prompt, return_tensors="pt").to(LOCAL_MODEL.device)
    with torch.no_grad():
        outputs = LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )
    input_len = inputs.input_ids.shape[1]
    return LOCAL_TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _passage_candidates(record: dict, all_names: list, cap: int = 8) -> list:
    """Canonical concept names (other than the record's own) that appear
    verbatim, case-insensitively, in the record's source passage."""
    passage_lower = (record.get("source_passage") or "").lower()
    own = (record.get("concept_name") or "").strip().lower()
    candidates = []
    for name in all_names:
        if name.lower() == own:
            continue
        if name.lower() in passage_lower:
            candidates.append(name)
            if len(candidates) >= cap:
                break
    return candidates


def extract_relations_for_record(record: dict, candidates: list) -> dict:
    """Prompt the local model for relations between a record's concept and the
    candidate concepts found in its passage. Returns accepted relations only
    (targets restricted to the candidate list); defensive parse like
    extract_okf_v15. Empty dict-of-empty-lists on any failure."""
    empty = {"prerequisites": [], "unlocks": [], "related_to": []}
    passage = record.get("source_passage") or ""
    if len(passage) > MAX_CHARS_TO_SLM:
        passage = passage[:MAX_CHARS_TO_SLM]
    prompt = RELATION_PROMPT.format(
        passage=passage,
        concept=record.get("concept_name", ""),
        candidates=json.dumps(candidates),
    )
    try:
        raw = _generate_local(prompt)
        cleaned = _extract_json_payload(raw)
        data = json.loads(cleaned)
    except Exception:
        return empty
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return empty

    own_lower = (record.get("concept_name") or "").strip().lower()
    cand_lookup = {c.lower(): c for c in candidates}

    def _accept_names(values):
        accepted = []
        for v in _string_list(values):
            key = v.strip().lower()
            if key != own_lower and key in cand_lookup:
                accepted.append(cand_lookup[key])
        return list(dict.fromkeys(accepted))

    accepted_related = []
    seen_rel = set()
    for rel in _normalize_related(data.get("related_to")):
        key = rel["concept"].strip().lower()
        if key == own_lower or key not in cand_lookup:
            continue
        rel["concept"] = cand_lookup[key]
        rel_key = (key, rel["relation"])
        if rel_key not in seen_rel:
            accepted_related.append(rel)
            seen_rel.add(rel_key)

    return {
        "prerequisites": _accept_names(data.get("prerequisites")),
        "unlocks": _accept_names(data.get("unlocks")),
        "related_to": accepted_related,
    }


def relation_pass(okf_results: list) -> dict:
    """Second-pass relation extraction over cleaned+canonicalized records.

    Mutates records in place: accepted relations are unioned onto the
    asserting record's prerequisites/unlocks/related_to, so edge provenance is
    that record's (doc_id, chunk_id) by construction. Targets not in the
    passage's candidate list are rejected before they can mint placeholders.
    """
    if LOCAL_MODEL is None:
        load_local_model()
    if not LOCAL_MODE or LOCAL_MODEL is None:
        print("ERROR: local model unavailable — relation pass never falls back to Ollama.")
        return {"processed": 0, "skipped": 0, "records_with_new_relations": 0,
                "relations_added": 0}

    all_names = sorted(
        {r.get("concept_name", "").strip() for r in okf_results
         if r.get("concept_name", "").strip()},
        key=len, reverse=True)  # longest first so 'Gradient Descent' beats 'Gradient'

    eligible = []
    skipped = 0
    for r in okf_results:
        if len(r.get("source_passage") or "") <= 200:
            skipped += 1
            continue
        candidates = _passage_candidates(r, all_names)
        if not candidates:
            skipped += 1
            continue
        eligible.append((r, candidates))

    print(f"\n[RELATION PASS] {len(eligible)} records with candidates "
          f"({skipped} skipped: trivial passage or no co-occurring concepts)")

    stats = {"processed": 0, "skipped": skipped,
             "records_with_new_relations": 0, "relations_added": 0}
    for i, (r, candidates) in enumerate(eligible):
        name = r.get("concept_name", "")
        print(f"  [{i+1}/{len(eligible)}] {name[:40]:40s} "
              f"({len(candidates)} candidates)", end="")
        sys.stdout.flush()
        start_time = time.time()
        rels = extract_relations_for_record(r, candidates)
        elapsed = time.time() - start_time
        stats["processed"] += 1

        added = 0
        for p in rels["prerequisites"]:
            if p not in r.setdefault("prerequisites", []):
                r["prerequisites"].append(p)
                added += 1
        for u in rels["unlocks"]:
            if u not in r.setdefault("unlocks", []):
                r["unlocks"].append(u)
                added += 1
        existing_rel = {(x.get("concept", "").lower(), x.get("relation", ""))
                        for x in r.get("related_to", []) if isinstance(x, dict)}
        for rel in rels["related_to"]:
            key = (rel["concept"].lower(), rel["relation"])
            if key not in existing_rel:
                r.setdefault("related_to", []).append(rel)
                existing_rel.add(key)
                added += 1

        if added:
            stats["records_with_new_relations"] += 1
            stats["relations_added"] += added
            summary = "; ".join(
                [f"req:{p}" for p in rels["prerequisites"]] +
                [f"unl:{u}" for u in rels["unlocks"]] +
                [f"{x['relation']}:{x['concept']}" for x in rels["related_to"]])
            print(f" -> +{added} [{summary[:60]}] ({elapsed:.1f}s)")
        else:
            print(f" -> 0 ({elapsed:.1f}s)")

    print(f"\n  Relation pass: {stats['relations_added']} relations added to "
          f"{stats['records_with_new_relations']}/{stats['processed']} records")
    return stats


def run_relations_only():
    """--relations-only mode: load saved results, clean+canonicalize, run the
    second-pass relation extraction, save, and rebuild the graph."""
    saved_file = BASE_DIR / "okf_results.json"
    if not saved_file.exists():
        print("ERROR: okf_results.json not found — run extraction first.")
        return
    with open(saved_file, "r", encoding="utf-8") as f:
        okf_results = json.load(f)
    print(f"Loaded {len(okf_results)} records from okf_results.json")

    # Cleanup/canonicalize FIRST so relation candidates are canonical names
    # from the final inventory (relation_pass runs after cleanup by design).
    okf_results = cleanup_and_canonicalize(okf_results)

    relation_pass(okf_results)

    chunk_count = len({(r.get("doc_id", ""), r.get("chunk_id", ""))
                       for r in okf_results if r.get("chunk_id")})
    return finalize_and_build(okf_results, chunk_count, chunk_count)

# Noise filters used in both extraction normalization and post-cleanup.
_JUNK_NAME_RE = re.compile(
    r"(?i)\b(authors?|contributors?|chairs?|funding|acknowledg|thank|grants?|projects?|"
    r"universit|institute|canada\s+cifar|research\s+chair|cifar\s+ai|nserc|"
    r"phd\s+program|fellowship|scholarship|discovery\s+grant|ai\s+chairs?|"
    r"computational\s+resources\s+provided|table\s+\d+|caption)\b|"
    r"best\s+model\s+without|underlined"
)
_NUMERIC_NAME_RE = re.compile(r"^\d[\d\s%\.x\-/]*$")
_FORMULA_OR_VALUE_NAME_RE = re.compile(
    r"(?i)([{}=∑∆ΔΦ]|\.{2,}|"
    r"\bvs\.?\b|\bcomparison\b|\b\d+%|\b\d+\s*of\s+tokens\b|"
    r"\breplacement\s+token\s*\(\d|"
    r"\b(system|hyperparameter)\s+dev\b|"
    r"\b(dev|test)\s+(f1|acc|accuracy|score)\b|"
    r"^test\s+scores?$|"
    r"\bstate-of-the-art\b.*\b(bleu|f1|accuracy|score)\b|"
    r"\bbatch\b|\bwithout\s+gold\s+access\b)"
)


def _concept_key(name: str) -> str:
    """Normalize a concept name for exact/near-exact self-reference checks."""
    key = re.sub(r"\([^)]*\)", "", name or "").lower()
    key = key.replace("-", " ")
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    words = [w for w in key.split() if w not in {"full", "basic", "standard", "general"}]
    normalized = []
    for word in words:
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return " ".join(normalized)


def is_same_concept_reference(a: str, b: str) -> bool:
    """True only for exact/near-exact aliases, not broader prerequisites."""
    a_key = _concept_key(a)
    b_key = _concept_key(b)
    return bool(a_key and b_key and a_key == b_key)


def is_valid_concept_name(name: str) -> bool:
    """Reject obvious metadata, numeric artifacts and non-concept names."""
    if not name or len(name.strip()) < 3:
        return False
    name = name.strip()
    if len(name) > 60 or len(name.split()) > 5:
        return False
    if _NUMERIC_NAME_RE.match(name):
        return False
    if _JUNK_NAME_RE.search(name):
        return False
    if _FORMULA_OR_VALUE_NAME_RE.search(name):
        return False
    return True


def prune_invalid_references(okf_results: list) -> dict:
    """Remove junk/self references from prerequisites, unlocks and related_to."""
    stats = {
        "invalid_prerequisites": 0,
        "invalid_unlocks": 0,
        "invalid_related": 0,
        "self_references": 0,
    }

    for r in okf_results:
        name = r.get("concept_name", "")

        clean_prereqs = []
        seen_prereqs = set()
        for p in r.get("prerequisites", []):
            if not isinstance(p, str) or not p.strip():
                stats["invalid_prerequisites"] += 1
                continue
            p = canonicalize_name(p)
            if is_same_concept_reference(p, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(p):
                stats["invalid_prerequisites"] += 1
                continue
            key = p.lower()
            if key not in seen_prereqs:
                clean_prereqs.append(p)
                seen_prereqs.add(key)

        clean_unlocks = []
        seen_unlocks = set()
        for u in r.get("unlocks", []):
            if not isinstance(u, str) or not u.strip():
                stats["invalid_unlocks"] += 1
                continue
            u = canonicalize_name(u)
            if is_same_concept_reference(u, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(u):
                stats["invalid_unlocks"] += 1
                continue
            key = u.lower()
            if key not in seen_unlocks:
                clean_unlocks.append(u)
                seen_unlocks.add(key)

        clean_related = []
        seen_related = set()
        for rel in r.get("related_to", []):
            if not isinstance(rel, dict):
                stats["invalid_related"] += 1
                continue
            concept = canonicalize_name(str(rel.get("concept", "")).strip())
            relation = str(rel.get("relation", "uses")).strip().lower()
            if is_same_concept_reference(concept, name):
                stats["self_references"] += 1
                continue
            if not is_valid_concept_name(concept):
                stats["invalid_related"] += 1
                continue
            if relation not in VALID_RELATIONS:
                relation = "uses"
            key = (concept.lower(), relation)
            if key not in seen_related:
                clean_related.append({"concept": concept, "relation": relation})
                seen_related.add(key)

        r["prerequisites"] = clean_prereqs
        r["unlocks"] = clean_unlocks
        r["related_to"] = clean_related

    return stats


# Minimal stopword list for grounding-overlap checks (content words only).
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "at", "from", "which", "can", "we",
    "you", "they", "their", "our", "not", "but", "if", "then", "than",
    "also", "such", "each", "other", "into", "over", "under", "between",
    "through", "when", "where", "how", "what", "all", "any", "some", "more",
    "most", "used", "using", "use", "based", "one", "two", "may", "will",
    "would", "should", "could", "has", "have", "had", "do", "does", "done",
    "there", "about", "both", "very", "given", "well", "only", "called",
}


def _content_words(text: str) -> set:
    """Lowercase content words (len>2, stopword-filtered) of a text."""
    return {
        w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def apply_grounding_filter(okf_results: list) -> tuple[list, dict]:
    """Grounding filter: a record must be anchored in its own source passage.

    A record is DROPPED when:
      - neither its exact concept_name nor any content word of the name
        appears in source_passage (case-insensitive), UNLESS its summary has
        >=30% content-word overlap with the passage (an abstractive concept
        genuinely described by the chunk); or
      - its summary is empty AND the exact name never appears in the passage
        (word-fragment matches alone can't justify a summary-less record).

    This catches SLM hallucinations and mode-collapse boilerplate (e.g.
    "Graph RAG" extracted "from" a math-textbook paragraph) while keeping
    valid abstractive concepts, which carry a passage-grounded summary.
    """
    stats = {
        "dropped_ungrounded": 0,
        "dropped_empty_summary": 0,
        "rescued_by_summary_overlap": 0,
        "kept": 0,
    }
    dropped_examples = {"ungrounded": [], "empty_summary": []}
    kept = []
    for r in okf_results:
        passage = (r.get("source_passage") or "").lower()
        name = (r.get("concept_name") or "").strip().lower()
        summary = (r.get("summary") or "").strip()
        if not passage or not name:
            stats["kept"] += 1
            kept.append(r)
            continue

        exact_hit = name in passage
        word_hit = exact_hit or any(w in passage for w in _content_words(name))

        if not word_hit:
            # Rescue only if the summary is demonstrably about this passage.
            summary_words = _content_words(summary)
            passage_words = _content_words(passage)
            overlap = (len(summary_words & passage_words) / len(summary_words)
                       if summary_words else 0.0)
            if overlap >= 0.30:
                stats["rescued_by_summary_overlap"] += 1
                stats["kept"] += 1
                kept.append(r)
            else:
                stats["dropped_ungrounded"] += 1
                dropped_examples["ungrounded"].append(r.get("concept_name"))
            continue

        if not summary and not exact_hit:
            stats["dropped_empty_summary"] += 1
            dropped_examples["empty_summary"].append(r.get("concept_name"))
            continue

        stats["kept"] += 1
        kept.append(r)

    total_dropped = stats["dropped_ungrounded"] + stats["dropped_empty_summary"]
    if total_dropped:
        print(f"  Grounding filter dropped {total_dropped} records:")
        print(f"    {stats['dropped_ungrounded']} ungrounded (name not in passage, "
              f"summary overlap <30%): {', '.join(dropped_examples['ungrounded'][:6])}"
              f"{'...' if len(dropped_examples['ungrounded']) > 6 else ''}")
        print(f"    {stats['dropped_empty_summary']} empty-summary without exact "
              f"name match: {', '.join(dropped_examples['empty_summary'][:6])}"
              f"{'...' if len(dropped_examples['empty_summary']) > 6 else ''}")
        if stats["rescued_by_summary_overlap"]:
            print(f"    ({stats['rescued_by_summary_overlap']} ungrounded-name records "
                  f"rescued by >=30% summary/passage overlap)")
    return kept, stats


def dedupe_identical_records(okf_results: list) -> tuple[list, int]:
    """Collapse mode-collapse duplicates: records sharing an identical
    (doc_id, concept_name, summary) triple are one observation repeated by the
    model, not independent evidence. Union their provenance and relations into
    a single record (same source-union semantics as merge_duplicate_results).
    """
    seen = {}
    deduped = []
    for r in okf_results:
        key = (
            r.get("doc_id", ""),
            (r.get("concept_name") or "").strip().lower(),
            (r.get("summary") or "").strip(),
        )
        if key in seen:
            existing = seen[key]
            existing["sources"] = _dedupe_dicts(
                _record_sources(existing) + _record_sources(r))
            existing["source_count"] = len(existing["sources"])
            for field in ("prerequisites", "unlocks", "tags"):
                existing[field] = list(dict.fromkeys(
                    existing.get(field, []) + r.get(field, [])))
            existing_rels = {(x.get("concept", ""), x.get("relation", ""))
                             for x in existing.get("related_to", []) if isinstance(x, dict)}
            for rel in r.get("related_to", []):
                if isinstance(rel, dict):
                    key_rel = (rel.get("concept", ""), rel.get("relation", ""))
                    if key_rel not in existing_rels:
                        existing.setdefault("related_to", []).append(rel)
                        existing_rels.add(key_rel)
            # Union per-relation provenance; the first record's entries win.
            merged_prov = dict(r.get("relation_provenance") or {})
            merged_prov.update(existing.get("relation_provenance") or {})
            if merged_prov:
                existing["relation_provenance"] = merged_prov
        else:
            seen[key] = r
            r["sources"] = _dedupe_dicts(_record_sources(r))
            r["source_count"] = len(r["sources"])
            deduped.append(r)
    return deduped, len(okf_results) - len(deduped)


def merge_duplicate_results(okf_results: list) -> tuple[list, int]:
    """Merge repeated concept records, keeping the richest fields."""
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
                        existing.setdefault("related_to", []).append(rel)
                        existing_rels.add(key_rel)
            # Union ALL provenance from every duplicate so cross-document
            # evidence is never undercounted. Deduplicate the source records
            # but keep one per distinct (doc/chunk/page/section) origin.
            existing["sources"] = _dedupe_dicts(
                _record_sources(existing) + _record_sources(r))
            existing["source_count"] = len(existing["sources"])
            # Per-relation provenance must survive the merge: relations copied
            # from r keep pointing at r's asserting chunk, not existing's.
            r_prov = dict(r.get("relation_provenance") or {})
            # Relations r asserted without explicit provenance default to r's
            # own (doc_id, chunk_id) — record that before it is lost.
            r_src = f"{r.get('doc_id', '')}:{r.get('chunk_id', '')}"
            if r_src != ":":
                for p in r.get("prerequisites", []):
                    r_prov.setdefault(f"prereq:{str(p).lower()}", r_src)
                for u in r.get("unlocks", []):
                    r_prov.setdefault(f"unlock:{str(u).lower()}", r_src)
                for rel in r.get("related_to", []):
                    if isinstance(rel, dict) and rel.get("concept"):
                        r_prov.setdefault(
                            f"related:{str(rel['concept']).lower()}", r_src)
            merged_prov = r_prov
            merged_prov.update(existing.get("relation_provenance") or {})
            if merged_prov:
                existing["relation_provenance"] = merged_prov
        else:
            seen[key] = r
            # Normalize the surviving record so it carries an explicit,
            # deduplicated provenance list from the outset.
            r["sources"] = _dedupe_dicts(_record_sources(r))
            r["source_count"] = len(r["sources"])
            merged_results.append(r)
    return merged_results, len(okf_results) - len(merged_results)


def prune_unresolved_references(okf_results: list) -> dict:
    """Prune truly invalid references (empty, non-string, self-refs) but keep
    cross-document references that become placeholder nodes in the graph.
    """
    stats = {
        "prerequisites": 0,
        "unlocks": 0,
        "related": 0,
        "self_references": 0,
    }

    for r in okf_results:
        concept_name = r.get("concept_name", "").strip().lower()

        clean_prereqs = []
        for p in r.get("prerequisites", []):
            if isinstance(p, str) and p.strip() and is_valid_concept_name(p):
                if p.strip().lower() == concept_name:
                    stats["self_references"] += 1
                else:
                    clean_prereqs.append(p)
            else:
                stats["prerequisites"] += 1

        clean_unlocks = []
        for u in r.get("unlocks", []):
            if isinstance(u, str) and u.strip() and is_valid_concept_name(u):
                if u.strip().lower() == concept_name:
                    stats["self_references"] += 1
                else:
                    clean_unlocks.append(u)
            else:
                stats["unlocks"] += 1

        clean_related = []
        for rel in r.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept"):
                concept = rel.get("concept", "")
                if isinstance(concept, str) and concept.strip() and is_valid_concept_name(concept):
                    if concept.strip().lower() == concept_name:
                        stats["self_references"] += 1
                    else:
                        clean_related.append(rel)
                else:
                    stats["related"] += 1
            else:
                stats["related"] += 1

        r["prerequisites"] = clean_prereqs
        r["unlocks"] = clean_unlocks
        r["related_to"] = clean_related

    return stats


# Common aliases to collapse.
# Keep this domain-agnostic. Add domain-specific aliases (e.g. biology, law)
# by editing this dict or loading a domain_aliases.json file.
ALIAS_MAP = {
    "gpt": "GPT",
    "openai gpt": "GPT",
    "lora": "Low-Rank Adaptation",
    "qlora": "Quantized Low-Rank Adaptation",
    "multi-head self-attention": "Multi-Head Attention",
    "multi head self attention": "Multi-Head Attention",
    "preference datasets": "Preference Data",
}

# Generic trailing words that don't distinguish concepts: "Transformer" and
# "Transformer Architecture" must merge into one node or edges land on a
# placeholder split from the sourced concept.
_GENERIC_SUFFIXES = (
    "architecture", "mechanism", "method", "methods",
    "technique", "techniques", "search",
)


def _merge_key(canon_lower: str) -> str:
    """Reduce a canonical name to a merge key: strip one generic suffix word
    and depluralize the last word, so near-identical names share a key."""
    words = canon_lower.split()
    if len(words) > 1 and words[-1] in _GENERIC_SUFFIXES:
        words = words[:-1]
    if words:
        w = words[-1]
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
            w = w[:-1]
        words[-1] = w
    return " ".join(words)


def canonicalize_name(name: str) -> str:
    """Normalize a concept name to a canonical form."""
    if not name:
        return ""

    # Strip whitespace
    name = name.strip()

    # Collapse snake_case / underscores the SLM sometimes emits so
    # "my_concept_name" merges with "My Concept Name".
    name = name.replace("_", " ")
    name = re.sub(r'\s+', ' ', name).strip()

    # Remove trailing periods
    name = name.rstrip(".")

    # Check alias map
    name_lower = name.lower()
    if name_lower in ALIAS_MAP:
        return ALIAS_MAP[name_lower]

    # Remove short parenthetical abbreviations: "Some Concept (SC)" → "Some Concept"
    name = re.sub(r'\s*\([^)]{1,10}\)\s*$', '', name)

    # If it's a full sentence (has a verb-like pattern), truncate
    if len(name) > 60:
        # Try to keep just the first noun phrase
        parts = name.split(",")
        name = parts[0].strip()
    if len(name) > 60:
        parts = name.split(" - ")
        name = parts[0].strip()

    # Title case — but leave short all-caps acronyms (GPT, BERT, LSTM) alone
    # so they don't become "Gpt"/"Bert".
    name = name.strip()
    if name == name.lower():
        name = name.title()
    elif name == name.upper() and len(name) > 5:
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
    merge_keys = {}  # merge key (suffix-stripped, depluralized) → canonical

    for raw in sorted(raw_names, key=lambda n: len(canonicalize_name(n))):
        if not raw:
            continue
        canon = canonicalize_name(raw)
        canon_lower = canon.lower()

        # Exact merge-key hit: "Transformer Architecture" → "Transformer",
        # "Language Models" → "Language Model". Shorter name (seen first due
        # to the sort) wins.
        mk = _merge_key(canon_lower)
        matched = False
        if len(canon_lower) > 3 and mk in merge_keys:
            existing_canon = merge_keys[mk]
            chosen = existing_canon if len(existing_canon) <= len(canon) else canon
            canon_map[raw] = chosen
            if chosen == canon:
                canonical_set[canon_lower] = canon
                merge_keys[mk] = canon
                for k, v in canon_map.items():
                    if v == existing_canon:
                        canon_map[k] = canon
            matched = True

        # Block merging distinct multi-word concepts by fuzzy name similarity.
        # We use a high cutoff (90) and fall back to strict equality if thefuzz
        # is not installed.
        if not matched:
            for existing_lower, existing_canon in canonical_set.items():
                similar = False
                if _fuzz is not None:
                    similar = _fuzz.ratio(canon_lower, existing_lower) >= 90
                else:
                    similar = canon_lower == existing_lower

                if similar and len(canon_lower) > 3:
                    # Prefer the shorter, more canonical spelling
                    chosen = existing_canon if len(existing_canon) <= len(canon) else canon
                    canon_map[raw] = chosen
                    if chosen == canon:
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
            merge_keys.setdefault(mk, canon)

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

        # Remap per-relation provenance keys ("kind:name_lower") so they still
        # match after their target names were canonicalized above.
        prov = result.get("relation_provenance")
        if isinstance(prov, dict) and prov:
            lower_map = {raw.lower(): canon for raw, canon in canon_map.items()}
            remapped = {}
            for key, src in prov.items():
                kind, _, target = key.partition(":")
                canon_target = lower_map.get(target, target)
                remapped[f"{kind}:{canon_target.lower()}"] = src
            result["relation_provenance"] = remapped

    return okf_results


# ---------------------------------------------------------------------------
# KùzuDB Graph Ingestion (MERGE semantics)
# ---------------------------------------------------------------------------
def create_concept_id(name: str) -> str:
    """Generate a stable, deterministic ID from a concept name."""
    cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
    cid = re.sub(r'_+', '_', cid).strip('_')
    return cid or 'concept'


def _kuzu_escape(value: str) -> str:
    """Escape a string literal for inline Kuzu Cypher queries.

    Kuzu uses backslash escaping (\\') — NOT SQL-style doubled quotes ('').
    Doubled quotes raise a parser exception, which ensure_concept/ensure_chunk
    silently swallowed, dropping any node/chunk whose text contained an
    apostrophe (root cause of phantom viz nodes missing from the concepts dict).
    """
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def ingest_to_kuzu(okf_results: list, db_path: str = "okf_graph.db"):
    """Ingest OKF results into KùzuDB with normalized schema and MERGE semantics."""
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
            CREATE NODE TABLE Document (
                id STRING PRIMARY KEY
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE NODE TABLE Chunk (
                id STRING PRIMARY KEY,
                chunk_id STRING,
                page_number INT64,
                section_title STRING,
                text_passage STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE NODE TABLE Concept (
                id STRING PRIMARY KEY,
                name STRING,
                concept_type STRING,
                difficulty STRING,
                summary STRING
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE HAS_CHUNK (
                FROM Document TO Chunk
            )
        """)
    except Exception:
        pass

    try:
        conn.execute("""
            CREATE REL TABLE MENTIONS (
                FROM Chunk TO Concept
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

    # Best-effort summary lookup for placeholder nodes created from edge refs.
    _summary_lookup = {}
    for _r in okf_results:
        _name = _r.get("concept_name", "")
        _s = _r.get("summary", "")
        if _s and len(_s) > len(_summary_lookup.get(_name, "")):
            _summary_lookup[_name] = _s

    def ensure_document(doc_id):
        if not doc_id:
            return
        safe_doc_id = _kuzu_escape(doc_id)
        try:
            conn.execute(f"MERGE (d:Document {{id: '{safe_doc_id}'}})")
        except Exception:
            pass

    def ensure_chunk(doc_id, chunk_id, page_number, section_title, text_passage):
        if not doc_id or not chunk_id:
            return ""
        cid = f"{doc_id}_{chunk_id}"
        safe_cid = _kuzu_escape(cid)
        safe_doc_id = _kuzu_escape(doc_id)
        safe_chunk_id = _kuzu_escape(chunk_id)
        safe_section = _kuzu_escape(section_title)
        safe_passage = _kuzu_escape(text_passage)
        
        try:
            conn.execute(f"""
                MERGE (c:Chunk {{id: '{safe_cid}'}})
                ON CREATE SET c.chunk_id = '{safe_chunk_id}',
                              c.page_number = {page_number},
                              c.section_title = '{safe_section}',
                              c.text_passage = '{safe_passage}'
            """)
            conn.execute(f"""
                MATCH (d:Document {{id: '{safe_doc_id}'}}),
                      (c:Chunk {{id: '{safe_cid}'}})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """)
        except Exception:
            pass
        return cid

    def ensure_concept(name, concept_type="definition", difficulty="intermediate",
                       summary="", is_placeholder=False):
        if not is_valid_concept_name(name):
            return ""
        cid = create_concept_id(name)
        
        safe_name = _kuzu_escape(name)
        safe_summary = _kuzu_escape((summary or _summary_lookup.get(name, ""))[:500])
        
        try:
            if is_placeholder:
                conn.execute(f"""
                    MERGE (c:Concept {{id: '{cid}'}})
                    ON CREATE SET c.name = '{safe_name}',
                                  c.concept_type = '{concept_type}',
                                  c.difficulty = '{difficulty}',
                                  c.summary = '{safe_summary}'
                """)
            else:
                conn.execute(f"""
                    MERGE (c:Concept {{id: '{cid}'}})
                    ON CREATE SET c.name = '{safe_name}',
                                  c.concept_type = '{concept_type}',
                                  c.difficulty = '{difficulty}',
                                  c.summary = '{safe_summary}'
                    ON MATCH SET c.name = '{safe_name}',
                                 c.concept_type = '{concept_type}',
                                 c.difficulty = '{difficulty}',
                                 c.summary = '{safe_summary}'
                """)
        except Exception:
            pass
        return cid

    def link_chunk_concept(chunk_db_id, concept_id):
        if not chunk_db_id or not concept_id:
            return
        safe_chunk_db_id = _kuzu_escape(chunk_db_id)
        try:
            conn.execute(f"""
                MATCH (c:Chunk {{id: '{safe_chunk_db_id}'}}),
                      (con:Concept {{id: '{concept_id}'}})
                MERGE (c)-[:MENTIONS]->(con)
            """)
        except Exception:
            pass

    def create_edge(from_id, to_id, rel_table, rel_type, source):
        """Create a relationship edge using MERGE to prevent duplicate edges."""
        if from_id == to_id:
            return False
        safe_source = _kuzu_escape(source)
        try:
            conn.execute(f"""
                MATCH (a:Concept {{id: '{from_id}'}}),
                      (b:Concept {{id: '{to_id}'}})
                MERGE (a)-[r:{rel_table}]->(b)
                ON CREATE SET r.relation_type = '{rel_type}', r.source = '{safe_source}'
            """)
            return True
        except Exception:
            return False

    # Ingest all concepts, documents, chunks, and links
    print("  Ingesting nodes and chunk associations...")
    unique_concept_ids = set()
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        
        # 3. Ensure Concept
        concept_id = ensure_concept(
            name,
            result.get("concept_type", "definition"),
            result.get("difficulty", "intermediate"),
            result.get("summary", "")
        )
        if concept_id:
            unique_concept_ids.add(concept_id)
            
        # Get all sources (provenance records) accumulated during merging
        sources = result.get("sources")
        if not (isinstance(sources, list) and sources):
            sources = [_source_record(result)]
            
        for src in sources:
            doc_id = src.get("doc_id", "")
            chunk_id = src.get("chunk_id", "")
            page_number = int(src.get("page_number", 0))
            section_title = src.get("section_title", "")
            text_passage = src.get("text_passage", "")
            
            if not doc_id or not chunk_id:
                continue
                
            # 1. Ensure Document
            ensure_document(doc_id)
            
            # 2. Ensure Chunk & Link Document -> Chunk
            chunk_db_id = ensure_chunk(doc_id, chunk_id, page_number, section_title, text_passage)
            
            # 4. Link Chunk -> Concept
            if chunk_db_id and concept_id:
                link_chunk_concept(chunk_db_id, concept_id)

        # Relation targets are NOT minted as nodes here. A target only gets an
        # edge if some record actually extracted it (checked in the edge loop
        # against the canonical inventory), so hallucinated placeholder nodes
        # (e.g. 'Grande Jura Distribution') can never enter the graph.

    print(f"    -> {len(unique_concept_ids)} unique concept nodes")

    # Create edges
    print("  Creating relationship edges...")
    # Canonical concept inventory: only names some record actually extracted
    # may be edge targets. Anything else is skipped and counted.
    extracted_names = {
        r.get("concept_name", "").strip().lower()
        for r in okf_results if r.get("concept_name")
    }
    skipped_unknown_targets = 0
    edge_count = 0
    seen_related_pairs = set()
    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        concept_id = create_concept_id(name)
        # Default provenance: this record's own chunk. Individual relations may
        # override via relation_provenance (set by merges/relation_pass), which
        # records the (doc_id, chunk_id) of the record that ASSERTED them.
        source_info = f"{result.get('doc_id', '')}:{result.get('chunk_id', '')}"
        rel_prov = result.get("relation_provenance") or {}

        for prereq in result.get("prerequisites", []):
            if isinstance(prereq, str) and prereq.strip() and is_valid_concept_name(prereq):
                if prereq.strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                prereq_id = create_concept_id(prereq)
                src = rel_prov.get(f"prereq:{prereq.lower()}", source_info)
                if create_edge(concept_id, prereq_id, "REQUIRES", "requires", src):
                    edge_count += 1

        for unlock in result.get("unlocks", []):
            if isinstance(unlock, str) and unlock.strip() and is_valid_concept_name(unlock):
                if unlock.strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                unlock_id = create_concept_id(unlock)
                src = rel_prov.get(f"unlock:{unlock.lower()}", source_info)
                if create_edge(concept_id, unlock_id, "UNLOCKS", "enables", src):
                    edge_count += 1

        for rel in result.get("related_to", []):
            if isinstance(rel, dict) and rel.get("concept") and is_valid_concept_name(rel["concept"]):
                if rel["concept"].strip().lower() not in extracted_names:
                    skipped_unknown_targets += 1
                    continue
                rel_id = create_concept_id(rel["concept"])
                rel_type = rel.get("relation", "related")
                # Symmetric-ish relations (uses/contrasts_with/variant_of) get
                # extracted from both endpoints; keep a single direction so the
                # UI doesn't draw the same association twice.
                sym_key = (min(concept_id, rel_id), max(concept_id, rel_id), rel_type)
                if sym_key in seen_related_pairs:
                    continue
                seen_related_pairs.add(sym_key)
                src = rel_prov.get(f"related:{rel['concept'].lower()}", source_info)
                if create_edge(concept_id, rel_id, "RELATED", rel_type, src):
                    edge_count += 1

    print(f"    -> {edge_count} edges created")
    if skipped_unknown_targets:
        print(f"    -> {skipped_unknown_targets} edges skipped "
              f"(target not in extracted concept inventory)")

    # Export graph to JSON
    print("  Exporting graph...")
    export = export_graph(conn)
    export["visualization"] = build_visual_graph(okf_results, export)
    export["graph_rag_index"] = build_graph_rag_index(okf_results, export)

    return conn, db, export


def export_graph(conn) -> dict:
    """Export the full graph structure to a dict, reconstructing concept sources from chunk relationships."""
    # Get concepts, documents, chunks, and their mentions
    result = conn.execute("""
        MATCH (c:Concept)
        OPTIONAL MATCH (chk:Chunk)-[:MENTIONS]->(c)
        OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(chk)
        RETURN c.id, c.name, c.concept_type, c.difficulty, c.summary, doc.id, chk.chunk_id, chk.page_number, chk.section_title, chk.text_passage
        ORDER BY c.name
    """)
    
    concepts = {}
    while result.has_next():
        row = result.get_next()
        cid = row[0]
        name = row[1]
        concept_type = row[2]
        difficulty = row[3]
        summary = row[4]
        
        doc_id = row[5]
        chunk_id = row[6]
        page_number = row[7]
        section_title = row[8]
        text_passage = row[9]
        
        if cid not in concepts:
            concepts[cid] = {
                "name": name,
                "concept_type": concept_type,
                "difficulty": difficulty,
                "summary": summary,
                "sources": []
            }
            
        if doc_id:
            source_rec = {
                "doc_id": doc_id,
                "source_category": infer_source_category(doc_id),
                "chunk_id": chunk_id,
                "page_number": int(page_number) if page_number is not None else 0,
                "section_title": section_title,
                "text_passage": text_passage
            }
            if source_rec not in concepts[cid]["sources"]:
                concepts[cid]["sources"].append(source_rec)

    # Get all edges
    edges = []
    for rel_table in ["REQUIRES", "UNLOCKS", "RELATED"]:
        try:
            result = conn.execute(f"""
                MATCH (a:Concept)-[r:{rel_table}]->(b:Concept)
                RETURN a.id, a.name, r.relation_type, b.id, b.name, r.source
            """)
            while result.has_next():
                row = result.get_next()
                edges.append({
                    "from_id": row[0],
                    "from_name": row[1],
                    "relation": row[2],
                    "to_id": row[3],
                    "to_name": row[4],
                    "source": row[5],
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


def _source_record(result: dict) -> dict:
    return {
        "doc_id": result.get("doc_id", ""),
        "source_category": result.get("source_category") or infer_source_category(result.get("doc_id", "")),
        "chunk_id": result.get("chunk_id", ""),
        "page_number": result.get("page_number", 0),
        "section_title": result.get("section_title", ""),
        "text_passage": result.get("source_passage", ""),
    }


def _dedupe_dicts(items: list) -> list:
    seen = set()
    deduped = []
    for item in items:
        key = tuple(sorted(item.items()))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _record_sources(result: dict) -> list:
    """All provenance records for a single result.

    Once records have been through merge_duplicate_results they carry an
    accumulated ``sources`` list (the union of every duplicate's provenance);
    return that when present so unioned evidence is never collapsed back to a
    single record. Otherwise fall back to the record's own source.
    """
    existing = result.get("sources")
    if isinstance(existing, list) and existing:
        return [s for s in existing if isinstance(s, dict)]
    return [_source_record(result)]


def build_visual_graph(okf_results: list, graph_export: dict) -> dict:
    """Build an Obsidian-style graph payload for local visualization."""
    nodes = {}

    for cid, concept in graph_export.get("concepts", {}).items():
        nodes[cid] = {
            "id": cid,
            "label": concept.get("name", cid),
            "concept_type": concept.get("concept_type", "definition"),
            "difficulty": concept.get("difficulty", "intermediate"),
            "summary": concept.get("summary", ""),
            "tags": [],
            "sources": [],
            "source_categories": [],
            "sections": [],
            "prerequisites": [],
            "unlocks": [],
            "related": [],
            "degree": 0,
            "source_count": 0,
        }

    for result in okf_results:
        name = result.get("concept_name", "")
        if not name:
            continue
        cid = create_concept_id(name)
        node = nodes.setdefault(cid, {
            "id": cid,
            "label": name,
            "concept_type": result.get("concept_type", "definition"),
            "difficulty": result.get("difficulty", "intermediate"),
            "summary": result.get("summary", ""),
            "tags": [],
            "sources": [],
            "source_categories": [],
            "sections": [],
            "prerequisites": [],
            "unlocks": [],
            "related": [],
            "degree": 0,
            "source_count": 0,
        })
        if len(result.get("summary", "")) > len(node.get("summary", "")):
            node["summary"] = result.get("summary", "")
        node["concept_type"] = result.get("concept_type", node["concept_type"])
        node["difficulty"] = result.get("difficulty", node["difficulty"])
        node["tags"].extend(result.get("tags", []))
        # Use the full (possibly merged) provenance list so cross-document
        # evidence unioned during dedup is preserved, not collapsed to one row.
        node["sources"].extend(_record_sources(result))
        category = result.get("source_category") or infer_source_category(result.get("doc_id", ""))
        if category:
            node["source_categories"].append(category)
        if result.get("section_title"):
            node["sections"].append(result["section_title"])
        node["prerequisites"].extend(result.get("prerequisites", []))
        node["unlocks"].extend(result.get("unlocks", []))
        node["related"].extend(result.get("related_to", []))

    links = []
    degree_counts = Counter()
    for idx, edge in enumerate(graph_export.get("edges", []), 1):
        source = edge["from_id"]
        target = edge["to_id"]
        degree_counts[source] += 1
        degree_counts[target] += 1
        links.append({
            "id": f"edge_{idx:05d}",
            "source": source,
            "target": target,
            "source_label": edge.get("from_name", source),
            "target_label": edge.get("to_name", target),
            "edge_type": edge.get("edge_type", "RELATED"),
            "relation": edge.get("relation", "related"),
            "source_ref": edge.get("source", ""),
        })

    for node in nodes.values():
        node["tags"] = sorted(set(t for t in node["tags"] if t))
        node["sources"] = _dedupe_dicts(node["sources"])
        node["source_categories"] = sorted(set(node["source_categories"]))
        node["sections"] = sorted(set(node["sections"]))
        node["prerequisites"] = sorted(set(node["prerequisites"]))
        node["unlocks"] = sorted(set(node["unlocks"]))
        node["source_count"] = len(node["sources"])
        node["degree"] = degree_counts[node["id"]]

    return {
        "nodes": sorted(nodes.values(), key=lambda x: x["label"].lower()),
        "links": links,
        "clusters": {
            "by_type": dict(Counter(n["concept_type"] for n in nodes.values())),
            "by_difficulty": dict(Counter(n["difficulty"] for n in nodes.values())),
            "by_source_category": dict(Counter(
                category
                for n in nodes.values()
                for category in (n.get("source_categories") or ["unknown"])
            )),
        },
        "stats": {
            "node_count": len(nodes),
            "link_count": len(links),
            "max_degree": max(degree_counts.values()) if degree_counts else 0,
        }
    }


def build_graph_rag_index(okf_results: list, graph_export: dict) -> dict:
    """Create a compact concept-neighborhood index for GraphRAG retrieval."""
    visual = graph_export.get("visualization") or build_visual_graph(okf_results, graph_export)
    by_id = {node["id"]: node for node in visual["nodes"]}
    index = {}

    for node_id, node in by_id.items():
        requires = []
        unlocks = []
        related = []
        for link in visual["links"]:
            if link["source"] != node_id:
                continue
            target_name = by_id.get(link["target"], {}).get("label", link["target_label"])
            if link["edge_type"] == "REQUIRES":
                requires.append(target_name)
            elif link["edge_type"] == "UNLOCKS":
                unlocks.append(target_name)
            else:
                related.append({"concept": target_name, "relation": link["relation"]})

        retrieval_terms = sorted(set(
            [node["label"], node.get("concept_type", ""), node.get("difficulty", "")] +
            node.get("tags", []) + requires + unlocks +
            [r["concept"] for r in related]
        ))
        index[node_id] = {
            "name": node["label"],
            "summary": node.get("summary", ""),
            "concept_type": node.get("concept_type", "definition"),
            "difficulty": node.get("difficulty", "intermediate"),
            "requires": sorted(set(requires)),
            "unlocks": sorted(set(unlocks)),
            "related": related,
            "sources": node.get("sources", []),
            "retrieval_terms": retrieval_terms,
            "retrieval_text": " | ".join(t for t in retrieval_terms if t),
        }

    return {
        "version": "okf-graphrag-v1",
        "concepts": index,
        "stats": {
            "total_concepts": len(index),
            "total_links": len(visual.get("links", [])),
        }
    }


def audit_graph_export(graph_export: dict) -> dict:
    """Compute deterministic graph issues that fine-tuning cannot guarantee."""
    concepts = graph_export.get("concepts", {})
    edges = graph_export.get("edges", [])
    visual = graph_export.get("visualization", {})
    visual_nodes = {n.get("id"): n for n in visual.get("nodes", [])}

    invalid_nodes = [
        c.get("name", cid)
        for cid, c in concepts.items()
        if not is_valid_concept_name(c.get("name", ""))
    ]
    empty_summary_nodes = [
        c.get("name", cid)
        for cid, c in concepts.items()
        if not (c.get("summary") or "").strip()
    ]
    placeholder_nodes = [
        n.get("label", node_id)
        for node_id, n in visual_nodes.items()
        if not n.get("sources")
    ]

    self_edges = [
        e for e in edges
        if e.get("from_id") == e.get("to_id")
        or is_same_concept_reference(e.get("from_name", ""), e.get("to_name", ""))
    ]

    requires_pairs = {
        (e.get("from_id"), e.get("to_id"))
        for e in edges
        if e.get("edge_type") == "REQUIRES"
    }
    reciprocal_requires = []
    seen = set()
    for a, b in requires_pairs:
        if (b, a) in requires_pairs and tuple(sorted((a, b))) not in seen:
            seen.add(tuple(sorted((a, b))))
            reciprocal_requires.append({
                "a": concepts.get(a, {}).get("name", a),
                "b": concepts.get(b, {}).get("name", b),
            })

    return {
        "stats": {
            "invalid_nodes": len(invalid_nodes),
            "empty_summary_nodes": len(empty_summary_nodes),
            "placeholder_nodes": len(placeholder_nodes),
            "self_edges": len(self_edges),
            "reciprocal_requires": len(reciprocal_requires),
        },
        "examples": {
            "invalid_nodes": invalid_nodes[:25],
            "empty_summary_nodes": empty_summary_nodes[:25],
            "placeholder_nodes": placeholder_nodes[:25],
            "self_edges": [
                {
                    "from": e.get("from_name"),
                    "to": e.get("to_name"),
                    "edge_type": e.get("edge_type"),
                }
                for e in self_edges[:25]
            ],
            "reciprocal_requires": reciprocal_requires[:25],
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
            "chunks_with_extractions": raw_extraction_count,
            "after_cleanup": len(okf_results),
            "failed_extractions": total_chunks - raw_extraction_count,
            "concepts_after_cleanup": len(okf_results),
            "total_concepts_in_graph": total_concepts,
            "total_edges_in_graph": total_edges,
            "orphan_nodes": orphan_count,
            "self_loops": self_loops,
        }
    }


# ---------------------------------------------------------------------------
# Shared post-extraction stages
# ---------------------------------------------------------------------------
def extract_chunks_with_model(chunks: list) -> tuple[list, int]:
    """Run OKF extraction over prose chunks with per-chunk progress output.

    Returns (okf_results, successful_chunk_count). Shared by run_pipeline and
    add_document so the extraction loop stays identical in both modes.
    """
    okf_results = []
    successful_chunk_count = 0
    for i, chunk in enumerate(chunks):
        progress = f"[{i+1}/{len(chunks)}]"
        section = chunk.get("section_title", "?")[:40]
        print(f"  {progress} {chunk['doc_id']} | {section} (p.{chunk['page_number']})", end="")
        sys.stdout.flush()

        start_time = time.time()
        results = extract_okf_v15(
            text=chunk["text"],
            doc_id=chunk["doc_id"],
            chunk_id=chunk["chunk_id"],
            page_number=chunk["page_number"],
            section_title=chunk["section_title"]
        )
        elapsed = time.time() - start_time

        if results:
            # Attach the original source passage so the UI can show the
            # highlighted chunk and deep-link back to the source page.
            passage = chunk["text"][:1600]
            for r in results:
                r["source_passage"] = passage
                r.setdefault("section_title", chunk.get("section_title", ""))
            okf_results.extend(results)
            successful_chunk_count += 1
            names = ", ".join(r["concept_name"] for r in results[:3])
            if len(results) > 3:
                names += f", +{len(results) - 3} more"
            print(f" -> {len(results)} concepts: {names[:70]} ({elapsed:.1f}s)")
        else:
            print(f" -> FAILED ({elapsed:.1f}s)")

    return okf_results, successful_chunk_count


def cleanup_and_canonicalize(okf_results: list) -> list:
    """Stages 2b+3: post-extraction cleanup and entity canonicalization.

    Idempotent — shared by finalize_and_build and the --relations-only mode
    (which must canonicalize BEFORE the relation pass so candidate names match
    the final concept inventory).
    """
    print("\n[2b] POST-EXTRACTION CLEANUP")
    print("-" * 50)
    pre_cleanup = len(okf_results)

    # 0.5. Remove non-concept metadata artifacts (authors, grants, chairs, etc.)
    okf_results = [r for r in okf_results if is_valid_concept_name(r.get("concept_name", ""))]
    junk_removed = pre_cleanup - len(okf_results)
    print(f"  Removed {junk_removed} junk/non-concept records")

    # 1. Remove reference/bibliography section extractions (noise)
    okf_results = [r for r in okf_results
                   if not r.get("section_title", "").lower().startswith("reference")]
    ref_removed = pre_cleanup - junk_removed - len(okf_results)
    print(f"  Removed {ref_removed} reference-section concepts")

    # 1.5. Grounding filter: a record must be anchored in its own source
    # passage (exact name or name content-word present), with a >=30%
    # summary/passage content-word-overlap rescue for abstractive concepts.
    # See apply_grounding_filter for the full rules and drop-count printout.
    okf_results, _ground_stats = apply_grounding_filter(okf_results)

    # 1.7. Anti-attractor dedup: mode-collapsed extractions repeat the exact
    # same (doc_id, concept_name, summary) across many chunks ('Vector' x407).
    # Collapse them to one record each, unioning provenance.
    okf_results, attractor_removed = dedupe_identical_records(okf_results)
    if attractor_removed > 0:
        print(f"  Collapsed {attractor_removed} identical "
              f"(doc, name, summary) mode-collapse duplicates")

    # 2. Remove self-loops (concept listing itself as prerequisite/unlock)
    ref_stats = prune_invalid_references(okf_results)
    print(
        "  Pruned references: "
        f"{ref_stats['invalid_prerequisites']} invalid prerequisites, "
        f"{ref_stats['invalid_unlocks']} invalid unlocks, "
        f"{ref_stats['invalid_related']} invalid related targets, "
        f"{ref_stats['self_references']} self references"
    )

    # 3. Merge duplicate concept names (keep richest version)
    okf_results, dupe_removed = merge_duplicate_results(okf_results)
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
    okf_results, post_canon_dupes = merge_duplicate_results(okf_results)

    # Count dedup stats
    raw_concepts = len(canon_map)
    unique_concepts = len(set(canon_map.values()))
    print(f"  Raw concept mentions: {raw_concepts}")
    print(f"  Canonical concepts: {unique_concepts}")
    print(f"  Aliases resolved: {raw_concepts - unique_concepts}")
    if post_canon_dupes > 0:
        print(f"  Merged {post_canon_dupes} duplicate concepts after canonicalization")

    # Post-canonicalization reference cleanup catches aliases that collapsed
    # into invalid names or near-self references.
    post_ref_stats = prune_invalid_references(okf_results)
    post_ref_removed = sum(post_ref_stats.values())
    if post_ref_removed > 0:
        print(
            "  Post-canonicalization reference prune: "
            f"{post_ref_stats['invalid_prerequisites']} invalid prerequisites, "
            f"{post_ref_stats['invalid_unlocks']} invalid unlocks, "
            f"{post_ref_stats['invalid_related']} invalid related targets, "
            f"{post_ref_stats['self_references']} self references"
        )

    unresolved_stats = prune_unresolved_references(okf_results)
    unresolved_removed = sum(unresolved_stats.values())
    if unresolved_removed > 0:
        print(
            "  Removed unresolved refs that would create placeholder nodes: "
            f"{unresolved_stats['prerequisites']} prerequisites, "
            f"{unresolved_stats['unlocks']} unlocks, "
            f"{unresolved_stats['related']} related targets"
        )

    cycles_broken = break_global_cycles(okf_results)
    if cycles_broken > 0:
        print(f"  Removed {cycles_broken} prerequisite/unlock cycle edges to enforce hierarchy DAG")

    # Fill empty summaries from the richest available source
    summary_by_name = {}
    for r in okf_results:
        name = r.get("concept_name", "")
        s = r.get("summary", "")
        if s and len(s) > len(summary_by_name.get(name, "")):
            summary_by_name[name] = s
    filled = 0
    for r in okf_results:
        name = r.get("concept_name", "")
        if not r.get("summary") and name in summary_by_name:
            r["summary"] = summary_by_name[name]
            filled += 1
    if filled > 0:
        print(f"  Filled {filled} empty summaries from sibling records")

    return okf_results


def finalize_and_build(okf_results: list, total_chunks: int,
                       successful_chunk_count: int):
    """Run every stage after extraction: save raw results, cleanup,
    canonicalization, KuzuDB graph build, exports and evaluation.

    Shared by run_pipeline (full / --resume runs) and add_document
    (incremental --add ingestion) so downstream behavior stays identical.
    """
    saved_file = BASE_DIR / "okf_results.json"

    # Save raw extraction results (anchored to BASE_DIR so resume-read and this
    # write always target the same file regardless of the current directory).
    with open(saved_file, "w", encoding="utf-8") as f:
        json.dump(okf_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to okf_results.json")

    okf_results = cleanup_and_canonicalize(okf_results)

    # ── Stage 4: KùzuDB Graph Ingestion ──
    print(f"\n[4] STAGE 4: KuzuDB Graph Ingestion (MERGE)")
    print("-" * 50)

    conn, db, graph_export = ingest_to_kuzu(okf_results, db_path=str(BASE_DIR / "okf_graph.db"))
    graph_audit = audit_graph_export(graph_export)

    # Save graph export (to root and graph_ui for static fallback hosting).
    # All artifacts are anchored to BASE_DIR so outputs land next to the
    # resume-read inputs no matter which directory the pipeline runs from.
    with open(BASE_DIR / "okf_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_export, f, indent=2, ensure_ascii=False)
    with open(BASE_DIR / "graph_audit.json", "w", encoding="utf-8") as f:
        json.dump(graph_audit, f, indent=2, ensure_ascii=False)

    static_dest = BASE_DIR / "graph_ui" / "okf_graph.json"
    if static_dest.parent.exists():
        with open(static_dest, "w", encoding="utf-8") as f:
            json.dump(graph_export, f, indent=2, ensure_ascii=False)
        print(f"  Saved to okf_graph.json, graph_audit.json and graph_ui/okf_graph.json")
    else:
        print(f"  Saved to okf_graph.json and graph_audit.json")

    # ── Stage 5: Evaluation ──
    print(f"\n[5] STAGE 5: Accuracy Evaluation")
    print("-" * 50)

    accuracy = evaluate_extraction(okf_results, total_chunks, graph_export,
                                    raw_extraction_count=successful_chunk_count)

    with open(BASE_DIR / "accuracy.json", "w", encoding="utf-8") as f:
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
    print(f"[OK] Pipeline complete! Files: okf_results.json, okf_graph.json, graph_audit.json, accuracy.json")
    print(f"{'=' * 70}")

    return okf_results, graph_export, accuracy


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path: str = None, resume: bool = False, local: bool = False):
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
        chunks = ingest_folder(input_path, max_pages=MAX_PAGES_PER_DOC)
    else:
        chunks = ingest_document(input_path, max_pages=MAX_PAGES_PER_DOC)

    raw_chunks = list(chunks)
    all_chunk_count = len(chunks)

    # Only prose chunks reach the SLM. Tables, references, front-matter and bare
    # math blocks are dropped here so they can't produce hallucinated nodes.
    kind_counts = Counter(c.get("chunk_kind", "prose") for c in chunks)
    prose_chunks = [c for c in chunks if c.get("chunk_kind", "prose") == "prose"]
    total_chunks = len(prose_chunks)

    print(f"\n  Total chunks: {all_chunk_count}  ->  {total_chunks} prose sent to SLM")
    dropped = {k: v for k, v in kind_counts.items() if k != "prose"}
    if dropped:
        print(f"  Dropped non-prose: {dropped}")

    if not prose_chunks:
        print("ERROR: No prose chunks to extract from!")
        return
    chunks = prose_chunks

    # ── Stage 2: OKF v1.5 Extraction ──
    if local:
        print(f"\n[2] STAGE 2: OKF v1.5 Extraction via local model: {_local_path}")
    else:
        print(f"\n[2] STAGE 2: OKF v1.5 Extraction via {MODEL_NAME}")
    print("-" * 50)

    saved_file = BASE_DIR / "okf_results.json"
    successful_chunk_count = 0
    if resume and saved_file.exists():
        print("  RESUMING from saved okf_results.json...")
        with open(saved_file, "r", encoding="utf-8") as f:
            okf_results = json.load(f)
        # Build map of chunks to restore page_number and section_title
        chunk_map = {(c["doc_id"], c["chunk_id"]): c for c in raw_chunks}
        # Fix any list-type concept_names from previous runs and restore chunk properties
        fixed = []
        for r in okf_results:
            cn = r.get("concept_name", "")
            if isinstance(cn, list):
                names = [n for n in cn if isinstance(n, str)]
                r["concept_name"] = names[0] if names else ""
            if r.get("doc_id") and not r.get("source_category"):
                r["source_category"] = infer_source_category(r.get("doc_id", ""))
            
            # Restore page_number, section_title, and source_passage from the current chunking pass
            doc_id = r.get("doc_id")
            chunk_id = r.get("chunk_id")
            if doc_id and chunk_id and (doc_id, chunk_id) in chunk_map:
                chunk = chunk_map[(doc_id, chunk_id)]
                r["page_number"] = chunk.get("page_number", 0)
                r["section_title"] = chunk.get("section_title", "")
                if not r.get("source_passage"):
                    r["source_passage"] = chunk.get("text", "")[:1600]
            
            if r.get("concept_name"):
                fixed.append(r)
        okf_results = fixed
        successful_chunk_count = len({
            (r.get("doc_id", ""), r.get("chunk_id", ""))
            for r in okf_results if r.get("chunk_id")
        })
        print(f"  Loaded {len(okf_results)} results (fixed list-type names)")
    else:
        if local:
            load_local_model()
        okf_results, successful_chunk_count = extract_chunks_with_model(chunks)
        print(f"\n  Extracted: {len(okf_results)} concepts from {total_chunks} chunks")

    return finalize_and_build(okf_results, total_chunks, successful_chunk_count)


# ---------------------------------------------------------------------------
# Incremental Ingestion (--add)
# ---------------------------------------------------------------------------
def compute_doc_id(path: str) -> str:
    """Derive the same doc_id the folder ingest would produce for this file.

    ingest_folder tags chunks with the path relative to the pdfs/ root (e.g.
    "papers/Edge2024_GraphRAG.pdf"); files outside pdfs/ get their basename.
    """
    resolved = Path(path).resolve()
    pdfs_root = (BASE_DIR / "pdfs").resolve()
    try:
        return str(resolved.relative_to(pdfs_root)).replace(os.sep, "/")
    except ValueError:
        return resolved.name


def add_document(path: str, limit: int = None):
    """Incrementally ingest ONE document with the LOCAL model and rebuild the
    merged graph. Re-adding a doc replaces its previous entries (no duplicates).

    limit: optional cap on prose chunks processed (fast testing on CPU).
    """
    from pdf_ingestion import ingest_document

    print("=" * 70)
    print("ARCHIPELAGO PIPELINE - INCREMENTAL ADD (single document)")
    print("=" * 70)

    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}")
        return

    doc_id = compute_doc_id(path)
    print(f"\n  Document: {path}")
    print(f"  doc_id:   {doc_id}")

    # ── Stage 1: Chunking (this document only) ──
    print("\n[1] STAGE 1: Section-Aware Document Chunking")
    print("-" * 50)

    chunks = ingest_document(path, max_pages=MAX_PAGES_PER_DOC)
    for chunk in chunks:
        chunk["doc_id"] = doc_id

    kind_counts = Counter(c.get("chunk_kind", "prose") for c in chunks)
    prose_chunks = [c for c in chunks if c.get("chunk_kind", "prose") == "prose"]
    new_chunk_total = len(prose_chunks)

    print(f"\n  Total chunks: {len(chunks)}  ->  {new_chunk_total} prose sent to SLM")
    dropped = {k: v for k, v in kind_counts.items() if k != "prose"}
    if dropped:
        print(f"  Dropped non-prose: {dropped}")

    if not prose_chunks:
        print("ERROR: No prose chunks to extract from!")
        return

    if limit is not None and limit < len(prose_chunks):
        print(f"  --limit {limit}: processing only the first {limit} prose chunks")
        prose_chunks = prose_chunks[:limit]

    # ── Stage 2: OKF v1.5 Extraction (LOCAL model only — never Ollama) ──
    print(f"\n[2] STAGE 2: OKF v1.5 Extraction via local model: {_local_path}")
    print("-" * 50)

    if LOCAL_MODEL is None:
        load_local_model()
    if not LOCAL_MODE or LOCAL_MODEL is None:
        print("ERROR: local model unavailable — --add never falls back to Ollama. Aborting.")
        return

    new_results, new_successful = extract_chunks_with_model(prose_chunks)
    print(f"\n  Extracted: {len(new_results)} concepts from {len(prose_chunks)} chunks")

    if not new_results:
        # Every chunk failed. Proceeding would delete the doc's previous
        # entries and replace them with nothing — abort instead so a bad run
        # can never destroy existing data.
        print("ERROR: extraction produced 0 concepts — leaving existing results untouched.")
        return

    # ── Merge with existing results (replace prior entries for this doc) ──
    print("\n[2a] MERGE WITH EXISTING RESULTS")
    print("-" * 50)

    saved_file = BASE_DIR / "okf_results.json"
    existing = []
    if saved_file.exists():
        with open(saved_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    before = len(existing)
    # Existing entries already carry page_number/section_title/source_passage;
    # leave them untouched — only this doc's records are replaced.
    existing = [r for r in existing if r.get("doc_id") != doc_id]
    replaced = before - len(existing)
    if replaced > 0:
        print(f"  Replaced {replaced} prior entries for {doc_id}")
    else:
        print(f"  No prior entries for {doc_id}")
    okf_results = existing + new_results
    print(f"  Merged total: {len(okf_results)} concepts "
          f"({len(existing)} existing + {len(new_results)} new)")

    # Chunk counts for the merged evaluation: distinct extracted chunks from the
    # kept existing results plus every prose chunk processed in this run.
    existing_chunk_count = len({
        (r.get("doc_id", ""), r.get("chunk_id", ""))
        for r in existing if r.get("chunk_id")
    })
    total_chunks = existing_chunk_count + len(prose_chunks)
    successful_chunk_count = existing_chunk_count + new_successful

    return finalize_and_build(okf_results, total_chunks, successful_chunk_count)


if __name__ == "__main__":
    args = sys.argv[1:]
    resume_mode = "--resume" in args
    args = [a for a in args if a != "--resume"]

    # --relations-only: second-pass relation extraction over saved results
    # (load okf_results.json -> cleanup/canonicalize -> relation_pass ->
    # save + rebuild graph). No chunk re-extraction.
    relations_only = "--relations-only" in args
    args = [a for a in args if a != "--relations-only"]

    ollama_mode = "--ollama" in args
    args = [a for a in args if a != "--ollama"]
    local_mode = LOCAL_MODE and not ollama_mode

    # --local is accepted explicitly (LOCAL_MODE already defaults it on when the
    # aura-qwen folder exists); strip it so it isn't mistaken for a path.
    args = [a for a in args if a != "--local"]

    # Optional uniform page cap: --max-pages N  (applies to every PDF)
    for i, a in enumerate(list(args)):
        if a == "--max-pages" and i + 1 < len(args):
            MAX_PAGES_PER_DOC = int(args[i + 1])
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--max-pages="):
            MAX_PAGES_PER_DOC = int(a.split("=", 1)[1])
            args = [x for x in args if x != a]
            break

    # Incremental single-document ingestion: --add <path> [--limit N]
    add_path = None
    for i, a in enumerate(list(args)):
        if a == "--add" and i + 1 < len(args):
            add_path = args[i + 1]
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--add="):
            add_path = a.split("=", 1)[1]
            args = [x for x in args if x != a]
            break

    # --limit N: only process the first N prose chunks (fast CPU testing)
    chunk_limit = None
    for i, a in enumerate(list(args)):
        if a == "--limit" and i + 1 < len(args):
            chunk_limit = int(args[i + 1])
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--limit="):
            chunk_limit = int(a.split("=", 1)[1])
            args = [x for x in args if x != a]
            break

    if relations_only:
        run_relations_only()
    elif add_path is not None:
        add_document(add_path, limit=chunk_limit)
    else:
        input_path = args[0] if args else None
        run_pipeline(input_path, resume=resume_mode, local=local_mode)

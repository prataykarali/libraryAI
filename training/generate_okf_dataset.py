#!/usr/bin/env python3
"""
High-quality OKF v2 dataset generator using subagents.
Produces 500+ validated training pairs from PDF chunks.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# OKF v2 Schema Definition
OKF_SCHEMA = {
    "type": "object",
    "properties": {
        "concept_name": {"type": "string", "minLength": 1, "maxLength": 60},
        "concept_type": {"type": "string", "enum": ["definition", "method", "technique", "principle", "algorithm", "architecture", "model", "framework", "metric", "benchmark", "dataset", "theorem", "lemma"]},
        "difficulty": {"type": "string", "enum": ["foundational", "intermediate", "advanced"]},
        "summary": {"type": "string", "minLength": 20, "maxLength": 500},
        "prerequisites": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 60}, "maxItems": 8},
        "unlocks": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 60}, "maxItems": 8},
        "related_to": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "minLength": 1, "maxLength": 60},
                    "relation": {"type": "string", "enum": ["prerequisite_of", "unlocks", "related_to", "contrasts_with", "extends", "specializes", "generalizes", "alternative_to"]}
                },
                "required": ["concept", "relation"]
            },
            "maxItems": 6
        },
        "tags": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 30}, "maxItems": 10}
    },
    "required": ["concept_name", "concept_type", "difficulty", "summary", "prerequisites", "unlocks", "related_to", "tags"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are an OKF extraction engine for the Archipelago knowledge graph.
From the TEXT below, extract 1-5 teachable CONCEPTS as a JSON array.

Each object MUST have exactly these keys:
concept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags

Rules:
- concept_name: ≤ 5 words, a reusable noun phrase, Title Case.
- concept_type: one of [definition, method, technique, principle, algorithm, architecture, model, framework, metric, benchmark, dataset, theorem, lemma]
- difficulty: one of [foundational, intermediate, advanced]
- Only concepts actually EXPLAINED in the text. No authors, citations, section titles, or passing mentions.
- prerequisites = what a learner needs FIRST; unlocks = what this ENABLES next.
- A concept must NEVER appear in its own prerequisites or unlocks.
- Keep names stable across documents so the same concept merges into one node.
- If the text has no real teachable concept, return [].
- Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures.
- Output ONLY the JSON array. No prose, no markdown fences.
- related_to relation must be one of: prerequisite_of, unlocks, related_to, contrasts_with, extends, specializes, generalizes, alternative_to
- tags: lowercase, concise, max 10 tags

TEXT:
{text}

Return ONLY the JSON array, no other text:"""

def load_chunks() -> List[Dict]:
    with open("/home/pratay-karali/Desktop/libraryAI/libraryAI/pdf_chunks.json") as f:
        return json.load(f)

def filter_chunks(chunks: List[Dict]) -> List[Dict]:
    filtered = []
    for c in chunks:
        if c.get("chunk_kind") != "prose":
            continue
        text = c.get("text", "").strip()
        if len(text) < 150:
            continue
        if text.count("et al.") > 5 or text.count("[") > 10:
            continue
        if "References" in c.get("section_title", ""):
            continue
        filtered.append(c)
    return filtered

def build_prompt(chunk: Dict) -> str:
    text = chunk.get("text", "").strip()
    # Limit text length for context window
    if len(text) > 3000:
        text = text[:3000] + "..."
    return SYSTEM_PROMPT.format(text=text)

def validate_concept(concept: Dict) -> tuple[bool, str]:
    """Validate a single concept against OKF v2 schema."""
    # Required fields
    required = ["concept_name", "concept_type", "difficulty", "summary", "prerequisites", "unlocks", "related_to", "tags"]
    for field in required:
        if field not in concept:
            return False, f"Missing required field: {field}"
    
    # concept_name: <= 5 words, Title Case
    name = concept["concept_name"]
    if len(name.split()) > 5:
        return False, f"concept_name exceeds 5 words: {name}"
    if not name.istitle() and not name.isupper():
        return False, f"concept_name not Title Case: {name}"
    
    # concept_type enum
    valid_types = ["definition", "method", "technique", "principle", "algorithm", "architecture", "model", "framework", "metric", "benchmark", "dataset", "theorem", "lemma"]
    if concept["concept_type"] not in valid_types:
        return False, f"Invalid concept_type: {concept['concept_type']}"
    
    # difficulty enum
    valid_diff = ["foundational", "intermediate", "advanced"]
    if concept["difficulty"] not in valid_diff:
        return False, f"Invalid difficulty: {concept['difficulty']}"
    
    # summary length
    if not (20 <= len(concept["summary"]) <= 500):
        return False, f"Summary length invalid: {len(concept['summary'])}"
    
    # prerequisites/unlocks: arrays of strings, max 8, no self-reference
    name_lower = name.lower()
    for field in ["prerequisites", "unlocks"]:
        items = concept[field]
        if not isinstance(items, list):
            return False, f"{field} must be array"
        if len(items) > 8:
            return False, f"{field} exceeds max 8 items"
        for item in items:
            if not isinstance(item, str) or not item.strip():
                return False, f"{field} contains empty/non-string"
            if item.lower() == name_lower:
                return False, f"Self-reference in {field}: {item}"
    
    # related_to: array of objects with concept + relation
    valid_relations = ["prerequisite_of", "unlocks", "related_to", "contrasts_with", "extends", "specializes", "generalizes", "alternative_to"]
    for rel in concept["related_to"]:
        if not isinstance(rel, dict) or "concept" not in rel or "relation" not in rel:
            return False, "related_to items must have concept and relation"
        if rel["relation"] not in valid_relations:
            return False, f"Invalid relation: {rel['relation']}"
        if rel["concept"].lower() == name_lower:
            return False, f"Self-reference in related_to: {rel['concept']}"
    
    # tags: lowercase, max 10
    tags = concept["tags"]
    if not isinstance(tags, list) or len(tags) > 10:
        return False, "tags must be array <= 10"
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip() or tag != tag.lower():
            return False, f"Invalid tag (must be lowercase): {tag}"
    
    return True, "OK"

def normalize_concept(concept: Dict) -> Dict:
    """Normalize concept fields to comply with schema."""
    # Normalize concept_name
    name = concept.get("concept_name", "").strip()
    words = name.split()
    if len(words) > 5:
        name = " ".join(words[:5])
    # Title Case
    name = " ".join(w.capitalize() for w in name.split())
    concept["concept_name"] = name
    
    # Normalize concept_type
    ctype = concept.get("concept_type", "").lower().strip()
    valid_types = ["definition", "method", "technique", "principle", "algorithm", "architecture", "model", "framework", "metric", "benchmark", "dataset", "theorem", "lemma"]
    if ctype not in valid_types:
        # Map common variants
        mapping = {
            "concept": "definition", "idea": "definition", "concept": "definition",
            "approach": "method", "strategy": "method", "technique": "technique",
            "algorithm": "algorithm", "architecture": "architecture", "model": "model",
            "framework": "framework", "metric": "metric", "benchmark": "benchmark",
            "dataset": "dataset", "theorem": "theorem", "lemma": "lemma",
            "principle": "principle"
        }
        concept["concept_type"] = mapping.get(ctype, "definition")
    else:
        concept["concept_type"] = ctype
    
    # Normalize difficulty
    diff = concept.get("difficulty", "").lower().strip()
    if diff in ["beginner", "basic", "introductory", "foundation"]:
        diff = "foundational"
    elif diff in ["intermediate", "medium", "moderate"]:
        diff = "intermediate"
    elif diff in ["advanced", "expert", "hard"]:
        diff = "advanced"
    else:
        diff = "intermediate"
    concept["difficulty"] = diff
    
    # Ensure arrays
    for field in ["prerequisites", "unlocks", "related_to", "tags"]:
        if field not in concept or not isinstance(concept[field], list):
            concept[field] = []
    
    # Normalize tags to lowercase
    concept["tags"] = [t.lower().strip() for t in concept["tags"] if isinstance(t, str) and t.strip()]
    concept["tags"] = concept["tags"][:10]
    
    # Normalize related_to
    valid_relations = ["prerequisite_of", "unlocks", "related_to", "contrasts_with", "extends", "specializes", "generalizes", "alternative_to"]
    normalized_related = []
    for rel in concept["related_to"]:
        if isinstance(rel, dict) and "concept" in rel:
            c = rel["concept"].strip()
            r = rel.get("relation", "related_to").lower().strip()
            if r not in valid_relations:
                r = "related_to"
            if c and c.lower() != name.lower():
                normalized_related.append({"concept": c, "relation": r})
    concept["related_to"] = normalized_related[:6]
    
    # Trim arrays
    concept["prerequisites"] = [p for p in concept["prerequisites"] if isinstance(p, str) and p.strip() and p.lower() != name.lower()][:8]
    concept["unlocks"] = [u for u in concept["unlocks"] if isinstance(u, str) and u.strip() and u.lower() != name.lower()][:8]
    
    return concept

def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM via available CLI tools."""
    # Try different LLM call methods
    # First try: opencode's built-in LLM if available
    try:
        result = subprocess.run(
            ["opencode", "run", "-p", prompt, "-m", model],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Try: direct python with openai/anthropic if keys available
    try:
        import os
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
    except:
        pass
    
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
    except:
        pass
    
    return ""

def extract_json_array(text: str) -> List[Dict]:
    """Extract JSON array from LLM output."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
    
    # Find first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end+1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        text = text.replace("'", '"')
        text = text.replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            return json.loads(text)
        except:
            return []

def process_chunk_with_llm(chunk: Dict) -> List[Dict]:
    """Process a single chunk with LLM and validate."""
    prompt = build_prompt(chunk)
    response = call_llm(prompt)
    if not response:
        return []
    
    concepts = extract_json_array(response)
    validated = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        c = normalize_concept(c)
        ok, msg = validate_concept(c)
        if ok:
            validated.append(c)
        else:
            print(f"  Validation failed for {c.get('concept_name', 'unknown')}: {msg}")
    return validated

def main():
    chunks = load_chunks()
    filtered = filter_chunks(chunks)
    print(f"Total chunks: {len(chunks)}")
    print(f"Filtered prose chunks: {len(filtered)}")
    
    # Group by document for balanced sampling
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c in filtered:
        by_doc[c["doc_id"]].append(c)
    
    for doc, doc_chunks in by_doc.items():
        print(f"  {doc}: {len(doc_chunks)} chunks")
    
    # Process chunks - we'll use subagents for this
    # For now, create a work queue
    work_queue = []
    for doc, doc_chunks in by_doc.items():
        # Take up to 10 chunks per document for diversity
        for chunk in doc_chunks[:15]:
            work_queue.append(chunk)
    
    print(f"\nWork queue: {len(work_queue)} chunks")
    
    # Save work queue for subagents
    with open("/home/pratay-karali/Desktop/libraryAI/libraryAI/work_queue.json", "w") as f:
        json.dump(work_queue, f)
    
    print("Work queue saved. Ready for subagent processing.")

if __name__ == "__main__":
    main()
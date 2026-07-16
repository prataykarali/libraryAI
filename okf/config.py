"""Configuration constants and prompts for the Archipelago OKF pipeline."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Default model can be overridden with the OKF_MODEL_NAME env var so the
# pipeline stays model-agnostic.
MODEL_NAME = os.environ.get("OKF_MODEL_NAME", "qwen3.5:0.8b")
BASE_DIR = Path(__file__).resolve().parent.parent
MAX_RETRIES = 1

# Local PyTorch model configuration (aura-qwen fine-tuned)
_local_path = BASE_DIR.parent / "aura-qwen"
if not _local_path.exists():
    _local_path = Path("/home/pratay-karali/Desktop/libraryAI/aura-qwen")

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
# SLM Extraction schema vocabularies
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

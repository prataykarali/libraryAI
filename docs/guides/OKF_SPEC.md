# OKF v1.6 — Open Knowledge Format & Extraction Skill

The contract the local SLM must follow to turn a text chunk into graph-ready concepts.
This is the "design" layer: schema + extraction skill + few-shot examples. The pipeline
code (`okf_pipeline.py`) enforces and repairs it; this doc is the source of truth.

---

## 1. What a concept node is

One OKF object = one **teachable concept** that could stand as a node in the Archipelago
graph. A concept is a *method, idea, tool, dataset, metric, theory, or result* — not a
sentence, not an author, not a citation, not a section heading.

Good nodes: `Scientific Method`, `Peer Review`, `Hypothesis`, `Empirical Evidence`,
`Theory Building`.
Bad nodes: `Introduction`, `Table 1`, `Smith et al.`, `we propose a model`, `the results`.

---

## 2. Schema (exact fields)

```jsonc
{
  "concept_name":  "string",   // short noun phrase, ≤ 5 words, Title Case, stable across docs
  "concept_type":  "enum",     // method | metric | technique | theory | tool | dataset | result | definition
  "difficulty":    "enum",     // foundational | intermediate | advanced | expert
  "summary":       "string",   // 1–2 sentences, what it IS (not "this paper says…")
  "prerequisites": ["string"], // concepts you must know BEFORE this  → REQUIRES edges
  "unlocks":       ["string"], // concepts this ENABLES downstream     → UNLOCKS edges
  "related_to":    [ { "concept": "string", "relation": "enum" } ],
                               // relation: uses | extends | contrasts_with | evaluated_by | variant_of | part_of
  "tags":          ["string"]  // lowercase-hyphenated keywords
}
```

Provenance (`doc_id`, `chunk_id`, `page_number`, `section_title`, `source_category`,
`source_passage`) is attached by the pipeline — **the model must NOT emit these**.
`source_passage` is the exact chunk text that produced the node so the UI can show the
highlighted source. The model only returns the 8 semantic fields above, as a JSON array
of 1–5 objects.

### Field rules that keep the graph clean
- `concept_name`: the single biggest lever. Keep it **canonical and reusable** so the same
  concept from two documents merges into one node. Prefer `Scientific Method` over
  `the scientific method described in section 3.2`.
- `prerequisites` / `unlocks`: these become the cross-document REQUIRES/UNLOCKS edges the
  whole project is testing. Populate them with *other concept names*, never with the concept
  itself (no self-loops), never with prose.
- Never invent concepts that aren't in the chunk. If the chunk is math-only or references,
  return `[]`.

---

## 3. The extraction skill (prompt the SLM runs)

The skill = system framing + rules + **one worked example** + the chunk. Small models
(0.6B–1.7B) obey a single concrete example far better than a long rule list, so the example
is load-bearing. Recommended decoding: `temperature=0.1`, `think=false`, JSON-only.

```
You are an OKF extraction engine for the Archipelago knowledge graph.
From the TEXT below, extract 1–5 teachable CONCEPTS as a JSON array.

Each object MUST have exactly these keys:
concept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags

Rules:
- concept_name: ≤ 5 words, a reusable noun phrase (e.g. "Scientific Method"), Title Case.
- Only concepts actually explained in the text. No authors, citations, or section titles.
- prerequisites = what a learner needs FIRST. unlocks = what this ENABLES next.
- A concept must never appear in its own prerequisites or unlocks.
- If the text is references, math notation, or has no real concept, return [].
- Output ONLY the JSON array. No prose, no markdown fences.

EXAMPLE
Text: "The scientific method is a procedure for acquiring knowledge that formulates
questions, tests hypotheses through repeatable experiments, and revises theories
based on evidence. Peer review then validates the findings before publication."
Output:
[
  {
    "concept_name": "Scientific Method",
    "concept_type": "method",
    "difficulty": "intermediate",
    "summary": "A systematic procedure for acquiring knowledge by formulating questions, testing hypotheses through experiments, and revising theories based on evidence.",
    "prerequisites": ["Hypothesis", "Experimentation"],
    "unlocks": ["Peer Review", "Theory Building"],
    "related_to": [{"concept": "Empirical Evidence", "relation": "uses"}],
    "tags": ["research", "methodology", "epistemology"]
  },
  {
    "concept_name": "Peer Review",
    "concept_type": "technique",
    "difficulty": "intermediate",
    "summary": "A validation process in which independent experts evaluate a study's methods, results, and conclusions before publication.",
    "prerequisites": ["Scientific Method"],
    "unlocks": ["Published Research"],
    "related_to": [{"concept": "Scientific Method", "relation": "evaluated_by"}],
    "tags": ["research", "validation"]
  }
]

TEXT:
{chunk}

Output:
```

---

## 4. Domain-agnostic guidance

The pipeline works on any document.  Apply these rules per chunk:

| Chunk content | Expected output |
|---|---|
| Prose explaining an idea/method/tool/result | 1–5 clean concept objects |
| Math notation or equations only | `[]` |
| Bibliographies / references / acknowledgements | `[]` |
| Front-matter, author lists, funding text | `[]` |
| Tables / raw numeric results | `[]` or a single metric/dataset node only if clearly defined |

A concept is **not** an author name, grant, section heading, value, or table row.

---

## 5. Known quality traps for small models (and the fix)

| Failure | Where it's handled |
|---------|-------------------|
| Malformed JSON / chatty preamble | `_extract_json_payload` + retry loop (`okf_pipeline.py`) |
| Self-loops (concept in its own prereqs) | stripped in `normalize_okf_item` + post-cleanup |
| Long sentence-y concept names | `canonicalize_name` truncation + quality metric |
| Same concept, different spellings | `ALIAS_MAP` + fuzzy dedup in `build_canonical_map` |
| Concepts pulled from reference lists | reference-section filter in post-cleanup |
| High-temperature drift | **TODO: set `options={"temperature":0.1}` in `ollama.chat`** |

Everything except the last row is already implemented. The temperature pin and the
few-shot example above are the two cheapest accuracy wins before any fine-tuning.

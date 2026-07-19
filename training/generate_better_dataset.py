#!/usr/bin/env python3
"""
Build a balanced OKF training dataset that teaches the model to extract
3-5 concepts per chunk (not just 1).

Strategy:
- Keep ALL 3+ concept examples (108)
- Keep ALL 2-concept examples (214)
- Sample only 100 1-concept examples (was 790, downsample to avoid bias)
- Keep 200 0-concept (empty) examples (teach when to output [])
- SYNTHESIZE ~200 new multi-concept examples from rich PDF chunks
  using concept names from the existing graph to ensure quality

Target: ~800 examples with distribution roughly:
  0-concept: 25%
  1-concept: 12%
  2-concept: 27%
  3-5-concept: 36%
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "training_data"

# ------------------------------------------------------------------
# 1. Load all existing examples
# ------------------------------------------------------------------
all_examples = []
for fname in ['okf_train_pairs.jsonl', 'okf_training_pairs_FIXED.jsonl', 'okf_test_pairs.jsonl']:
    fp = OUT_DIR / fname
    if fp.exists():
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    out = json.loads(d['output']) if isinstance(d['output'], str) else d['output']
                    all_examples.append({'data': d, 'concept_count': len(out)})

# Group by count
by_count = defaultdict(list)
for e in all_examples:
    by_count[e['concept_count']].append(e)

print(f"Loaded {len(all_examples)} total examples")
for k in sorted(by_count):
    print(f"  {k}-concept: {len(by_count[k])}")

# ------------------------------------------------------------------
# 2. Select examples for new dataset
# ------------------------------------------------------------------
selected = []

# All 3+ concept examples (108)
selected.extend(by_count.get(3, []) + by_count.get(4, []) + by_count.get(5, []))
print(f"\nKept ALL 3+ concept: {len(by_count.get(3,[])) + len(by_count.get(4,[])) + len(by_count.get(5,[]))}")

# All 2-concept examples (214)
selected.extend(by_count.get(2, []))
print(f"Kept ALL 2-concept: {len(by_count.get(2,[]))}")

# Sample 100 x 1-concept (was 790)
single_sample = random.sample(by_count.get(1, []), min(100, len(by_count.get(1, []))))
selected.extend(single_sample)
print(f"Sampled 1-concept: {len(single_sample)}")

# Sample 200 x 0-concept (was 900)
empty_sample = random.sample(by_count.get(0, []), min(200, len(by_count.get(0, []))))
selected.extend(empty_sample)
print(f"Sampled 0-concept: {len(empty_sample)}")

# ------------------------------------------------------------------
# 3. Synthesize multi-concept examples from PDF chunks
# ------------------------------------------------------------------
print("\n--- Synthesizing new multi-concept examples ---")

# Load PDF chunks with section titles
with open(BASE_DIR / "pdf_chunks.json") as f:
    chunks = json.load(f)

# Load graph nodes for reference
with open(BASE_DIR / "_graph_nodes.json") as f:
    graph_nodes = json.load(f)
concept_db = {n['name'].lower(): n for n in graph_nodes}

# Concept templates we know the model should extract
concept_templates = {
    "BERT": {
        "concept_name": "BERT",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Bidirectional Encoder Representations from Transformers - a bidirectional Transformer encoder pre-trained with masked language modeling and next sentence prediction.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["bert", "pre-trained-model", "transformer"]
    },
    "GPT": {
        "concept_name": "GPT",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Generative Pre-trained Transformer - a left-to-right language model using Transformer decoder architecture.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["gpt", "pre-trained-model", "transformer"]
    },
    "Fine-Tuning": {
        "concept_name": "Fine-Tuning",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["fine-tuning", "transfer-learning"]
    },
    "LoRA": {
        "concept_name": "LoRA",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into Transformer layers.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["lora", "peft", "efficient-tuning"]
    },
    "Masked Language Modeling": {
        "concept_name": "Masked Language Modeling",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A pre-training objective where random tokens are masked and the model predicts them using bidirectional context.",
        "prerequisites": ["Transformer Encoder"],
        "unlocks": [],
        "related_to": [],
        "tags": ["mlm", "masked-lm", "pre-training", "bert"]
    },
    "Next Sentence Prediction": {
        "concept_name": "Next Sentence Prediction",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A binary classification pre-training task predicting whether sentence B follows sentence A.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["nsp", "pre-training", "bert"]
    },
    "Feature-Based Approach": {
        "concept_name": "Feature-Based Approach",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A transfer learning approach where pre-trained model embeddings are used as fixed features without updating model weights.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["feature-based", "transfer-learning"]
    },
    "Retrieval-Augmented Generation": {
        "concept_name": "Retrieval-Augmented Generation",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A method that retrieves relevant documents from an external knowledge source and uses them to augment LLM generation.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["rag", "retrieval", "generation"]
    },
    "GraphRAG": {
        "concept_name": "GraphRAG",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "A graph-based approach to question answering that builds an entity knowledge graph and pre-generates community summaries.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["graphrag", "graph-rag", "knowledge-graph"]
    },
    "Multi-Head Attention": {
        "concept_name": "Multi-Head Attention",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "Attention performed in parallel across multiple heads, allowing the model to attend to different representation subspaces.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["multi-head-attention", "attention"]
    },
    "Transformer Encoder": {
        "concept_name": "Transformer Encoder",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "A stack of self-attention and feed-forward layers that encode input sequences into contextual representations.",
        "prerequisites": [],
        "unlocks": ["BERT", "Bidirectional Representation Learning"],
        "related_to": [],
        "tags": ["transformer-encoder", "transformer"]
    },
    "Transformer Decoder": {
        "concept_name": "Transformer Decoder",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "An autoregressive stack of masked self-attention and cross-attention layers that generate output sequences.",
        "prerequisites": [],
        "unlocks": ["GPT"],
        "related_to": [],
        "tags": ["transformer-decoder", "transformer"]
    },
    "Community Detection": {
        "concept_name": "Community Detection",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Algorithms that partition a graph into clusters of densely connected nodes (communities).",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["community-detection", "graph", "clustering"]
    },
    "Entity Extraction": {
        "concept_name": "Entity Extraction",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A process using language models to detect and capture meaningful entities and relationships from text.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["entity-extraction", "information-extraction"]
    },
    "Gradient Descent": {
        "concept_name": "Gradient Descent",
        "concept_type": "method",
        "difficulty": "foundational",
        "summary": "An iterative optimization algorithm that moves parameters in the direction of steepest descent of the loss function.",
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": ["gradient-descent", "optimization", "training"]
    },
}

OKF_SYSTEM = """You are an OKF extraction engine for the Archipelago knowledge graph.
From the TEXT below, extract 1-5 teachable CONCEPTS as a JSON array.

Each object MUST have exactly these keys:
concept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags

Rules:
- concept_name: ≤ 5 words, a reusable noun phrase, Title Case.
- Only concepts actually explained in the text. No authors, citations, section titles.
- prerequisites = what a learner needs FIRST; unlocks = what this ENABLES next.
- A concept must NEVER appear in its own prerequisites or unlocks.
- Keep names stable across documents so the same concept merges into one node.
- If the text has no real teachable concept, return [].
- Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures.
- Output ONLY the JSON array. No prose, no markdown fences.

TEXT:
{text}

Return ONLY the JSON array, no other text:"""

# Map of keywords → concepts that should appear together
CONCEPT_CLUSTERS = {
    ("BERT", "bidirectional", "encoder", "MLM"): ["BERT", "Transformer Encoder", "Masked Language Modeling", "Next Sentence Prediction", "Fine-Tuning"],
    ("GPT", "unidirectional", "decoder", "autoregressive"): ["GPT", "Transformer Decoder", "Fine-Tuning"],
    ("LoRA", "rank", "adaptation", "low-rank"): ["LoRA", "Fine-Tuning", "Gradient Descent"],
    ("GraphRAG", "knowledge graph", "community"): ["GraphRAG", "Retrieval-Augmented Generation", "Community Detection", "Entity Extraction"],
    ("attention", "multi-head", "transformer", "self-attention"): ["Multi-Head Attention", "Transformer Encoder", "Transformer Decoder"],
    ("fine-tuning", "transfer", "adapt", "downstream"): ["Fine-Tuning", "Feature-Based Approach", "LoRA"],
    ("masked language", "MLM", "bidirectional", "pre-train"): ["Masked Language Modeling", "BERT", "Transformer Encoder"],
}

def keyword_match(text, keywords):
    """Check how many keywords from a cluster appear in the text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)

# Find rich chunks and map them to concept clusters
synthetic = []
used_texts = set()

for cluster_kws, concept_names in CONCEPT_CLUSTERS.items():
    # Find chunks matching at least 2 keywords
    candidates = []
    for c in chunks:
        text = c.get('text', '')
        wc = len(text.split())
        if wc < 40 or wc > 350:
            continue
        if text in used_texts:
            continue
        score = keyword_match(text, cluster_kws)
        if score >= 2:
            candidates.append((score, c))
    
    # Take top matching chunks for this cluster
    candidates.sort(key=lambda x: -x[0])
    for score, c in candidates[:8]:  # up to 8 per cluster
        text = c['text']
        if text in used_texts:
            continue
        used_texts.add(text)
        
        # Select 3-5 concepts from this cluster
        n_concepts = min(len(concept_names), random.randint(3, 5))
        chosen = random.sample(concept_names, n_concepts)
        
        output = [concept_templates[cn] for cn in chosen if cn in concept_templates]
        if len(output) < 3:
            continue
        
        instruction = OKF_SYSTEM.replace("{text}", text.strip())
        
        synthetic.append({
            "instruction": instruction,
            "input": "",
            "output": json.dumps(output, indent=2),
            "doc_id": c.get('doc_id', ''),
            "chunk_id": c.get('chunk_id', ''),
            "synthetic": True,
        })

print(f"Synthesized {len(synthetic)} new multi-concept examples")

# ------------------------------------------------------------------
# 4. Combine and shuffle, split train/test
# ------------------------------------------------------------------
combined = []
for e in selected:
    d = e['data'].copy()
    d['synthetic'] = False
    combined.append(d)

combined.extend(synthetic)

random.shuffle(combined)

# Distribution check
dist = defaultdict(int)
for d in combined:
    out = json.loads(d['output']) if isinstance(d['output'], str) else d['output']
    dist[len(out)] += 1

print(f"\n=== NEW DATASET: {len(combined)} examples ===")
for k in sorted(dist):
    pct = dist[k] / len(combined) * 100
    print(f"  {k}-concept: {dist[k]} ({pct:.1f}%)")

# Split 80/20
split_idx = int(len(combined) * 0.8)
random.shuffle(combined)
train = combined[:split_idx]
test = combined[split_idx:]

# Write files
train_path = OUT_DIR / "okf_train_balanced.jsonl"
test_path = OUT_DIR / "okf_test_balanced.jsonl"

with open(train_path, 'w') as f:
    for d in train:
        f.write(json.dumps(d) + '\n')

with open(test_path, 'w') as f:
    for d in test:
        f.write(json.dumps(d) + '\n')

print(f"\nWrote {len(train)} train → {train_path}")
print(f"Wrote {len(test)} test → {test_path}")

# Also write a standalone script for verification
PYEOF
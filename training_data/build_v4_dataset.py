#!/usr/bin/env python
"""Build the v4 fine-tuning dataset for aura-qwen.

Fixes the v3 statistics the model faithfully (and disastrously) learned:
  v3: 'Vector' = 37% of labels | 51% empty outputs | 13% relation coverage
  v4 targets: <=3% per label | <=15% empty | >=60% relation coverage,
              >=25% multi-concept, >=25% textbook passages.

Sources, in priority order:
  1. v3 pairs that already carry relations (kept as-is).
  2. okf_results.json.bak2 old-model records (78 with relations, avg 4.2/rec)
     converted to pairs, grounding-validated, relation types remapped.
  3. Hand-authored gold pairs over real pdf_chunks.json passages (math
     textbook priority), stored in training_data/v4_gold_pairs.json.
  4. A small legit-empty set (symbol tables, references, figure junk).
  5. Balanced singles from v3 (label-capped).

Split 90/10 by (doc_id, page bucket) so no passage leaks across splits.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)
TD = Path("training_data")
VALID_RELATIONS = {"contrasts_with", "uses", "extends", "evaluated_by",
                   "variant_of", "part_of"}
RELATION_REMAP = {
    "related_to": None,          # drop — not a valid edge type
    "prerequisite_for": None,    # drop — handled by prerequisites list
    "applied_in": "uses",
    "used_in": "uses",
    "used_by": "uses",
    "builds_on": "extends",
    "similar_to": "variant_of",
    "component_of": "part_of",
}
VALID_TYPES = {"method", "metric", "technique", "theory", "tool",
               "dataset", "result", "definition"}
VALID_DIFF = {"foundational", "intermediate", "advanced", "expert"}

TEMPLATE = open("/tmp/claude-1000/v3_template.txt").read()

STOP = set("the a an of to in and or for with on by is are was be as at from "
           "that this it its we our their can may which such".split())


def content_words(text):
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in STOP}


def grounded(name, passage):
    """Name (or >=80% of its content words) appears in the passage."""
    pl = (passage or "").lower()
    if (name or "").lower() in pl:
        return True
    words = content_words(name)
    if not words:
        return False
    return sum(1 for w in words if w in pl) / len(words) >= 0.8


def fix_concept(c, passage, inventory):
    """Normalize one concept object; return None to discard."""
    name = (c.get("concept_name") or "").strip()
    if not name or not grounded(name, passage):
        return None
    out = {
        "concept_name": name,
        "concept_type": c.get("concept_type") if c.get("concept_type") in VALID_TYPES else "definition",
        "difficulty": c.get("difficulty") if c.get("difficulty") in VALID_DIFF else "intermediate",
        "summary": (c.get("summary") or "").strip(),
        "prerequisites": [],
        "unlocks": [],
        "related_to": [],
        "tags": [t for t in (c.get("tags") or []) if isinstance(t, str)][:6],
    }
    nl = name.lower()
    pl = (passage or "").lower()

    def keep_target(t):
        tl = t.lower()
        return tl != nl and (tl in pl or tl in inventory)

    out["prerequisites"] = [p for p in (c.get("prerequisites") or [])
                            if isinstance(p, str) and p.strip() and keep_target(p)][:6]
    out["unlocks"] = [u for u in (c.get("unlocks") or [])
                      if isinstance(u, str) and u.strip() and keep_target(u)][:6]
    seen = set()
    for r in (c.get("related_to") or []):
        if not isinstance(r, dict) or not r.get("concept"):
            continue
        rel = r.get("relation", "")
        if rel not in VALID_RELATIONS:
            rel = RELATION_REMAP.get(rel)
        if rel is None or not keep_target(r["concept"]):
            continue
        key = (r["concept"].lower(), rel)
        if key not in seen:
            out["related_to"].append({"concept": r["concept"], "relation": rel})
            seen.add(key)
    return out


def make_pair(passage, concepts, meta):
    return {
        "instruction": TEMPLATE.replace("{TEXT}", passage),
        "input": "",
        "output": json.dumps(concepts, ensure_ascii=False),
        "doc_id": meta.get("doc_id", ""),
        "chunk_id": meta.get("chunk_id", ""),
        "page_number": meta.get("page_number", 0),
        "section_title": meta.get("section_title", ""),
    }


def has_rel(c):
    return bool(c.get("prerequisites") or c.get("unlocks") or c.get("related_to"))


# ---------------------------------------------------------------- inputs
v3 = [json.loads(l) for l in open(TD / "okf_train_pairs_v3.jsonl")]
v3 += [json.loads(l) for l in open(TD / "okf_test_pairs_v3.jsonl")]
bak2 = json.load(open("okf_results.json.bak2"))
chunks = json.load(open("pdf_chunks.json"))
chunk_by_key = {(c["doc_id"], c["chunk_id"]): c for c in chunks}
gold = json.load(open(TD / "v4_gold_pairs.json"))

inventory = {r.get("concept_name", "").strip().lower() for r in bak2}
inventory |= {r.get("concept_name", "").strip().lower()
              for r in json.load(open("okf_results.json"))}
# also allow well-known curriculum names used in relations
inventory |= {n.lower() for n in [
    "Linear Algebra", "Calculus", "Probability Theory", "Matrix", "Vector",
    "Eigenvalue", "Eigenvector", "Gradient", "Derivative", "Optimization",
    "Machine Learning", "Neural Network", "Transformer", "Attention Mechanism",
    "Language Model", "Pre-training", "Fine-Tuning", "Tokenization",
    "Linear Regression", "Logistic Regression", "PCA", "SVD",
    "Principal Component Analysis", "Singular Value Decomposition",
    "Maximum Likelihood Estimation", "Bayes' Theorem", "Bayesian Inference",
    "Gaussian Distribution", "Random Variable", "Loss Function",
    "Backpropagation", "Gradient Descent", "Convex Optimization",
    "Lagrange Multipliers", "Support Vector Machine", "Kernel Trick",
    "Dimensionality Reduction", "Question Answering", "Text Classification",
    "Sequence Tagging", "Information Retrieval", "Knowledge Graph",
]}

pairs = []
used_texts = set()


def add_pair(passage, concepts, meta, source):
    key = passage[:160]
    if key in used_texts:
        return False
    used_texts.add(key)
    p = make_pair(passage, concepts, meta)
    p["_source"] = source
    pairs.append(p)
    return True


# ---- 1. v3 pairs that already have relations, re-validated ----
kept_v3_rel = 0
for p in v3:
    try:
        out = json.loads(p["output"])
    except Exception:
        continue
    if not out or not any(has_rel(c) for c in out):
        continue
    passage = re.search(r"TEXT:\n(.*)\n\nReturn ONLY the JSON array, no other text:$",
                        p["instruction"], re.S).group(1)
    fixed = [fc for c in out if (fc := fix_concept(c, passage, inventory))]
    if fixed and any(has_rel(c) for c in fixed):
        if add_pair(passage, fixed, p, "v3_relations"):
            kept_v3_rel += 1

# ---- 2. bak2 old-model records -> pairs (grouped per chunk) ----
by_chunk = defaultdict(list)
for r in bak2:
    if r.get("source_passage"):
        by_chunk[(r["doc_id"], r["chunk_id"])].append(r)
kept_bak2 = 0
for (doc, ck), recs in by_chunk.items():
    passage = max((r.get("source_passage") or "" for r in recs), key=len)
    if len(passage) < 200:
        continue
    fixed = [fc for r in recs if (fc := fix_concept(r, passage, inventory))]
    if fixed and any(has_rel(c) for c in fixed):
        meta = {"doc_id": doc, "chunk_id": ck,
                "page_number": recs[0].get("page_number", 0),
                "section_title": recs[0].get("section_title", "")}
        if add_pair(passage, fixed, meta, "bak2"):
            kept_bak2 += 1

# ---- 3. authored gold pairs (passages fetched from pdf_chunks) ----
kept_gold = 0
for g in gold:
    c = chunk_by_key.get((g["doc_id"], g["chunk_id"]))
    if c is None:
        print(f"  WARN gold pair references missing chunk {g['doc_id']}:{g['chunk_id']}")
        continue
    passage = c["text"]
    fixed = [fc for cc in g["concepts"] if (fc := fix_concept(cc, passage, inventory))]
    dropped = len(g["concepts"]) - len(fixed)
    if dropped:
        print(f"  WARN gold {g['chunk_id']}: {dropped} concept(s) failed grounding")
    if fixed:
        meta = {"doc_id": g["doc_id"], "chunk_id": g["chunk_id"],
                "page_number": c.get("page_number", 0),
                "section_title": c.get("section_title", "")}
        if add_pair(passage, fixed, meta, "gold"):
            kept_gold += 1

# ---- 4. legit empty pairs: symbol tables / references / non-prose ----
empties = []
for c in chunks:
    if c.get("chunk_kind") in ("reference", "table", "math") and len(c.get("text", "")) > 150:
        empties.append(c)
random.shuffle(empties)
target_nonempty = len(pairs)
n_empty = min(len(empties), max(10, int(0.13 * (target_nonempty + 60) / 0.87)))
kept_empty = 0
for c in empties[:n_empty]:
    if add_pair(c["text"], [], c, "empty"):
        kept_empty += 1

# ---- 5. balanced singles from v3 (label cap), to add volume ----
label_count = Counter()
for p in pairs:
    for c in json.loads(p["output"]):
        label_count[c["concept_name"]] += 1
kept_singles = 0
v3_singles = [p for p in v3 if p.get("output")]
random.shuffle(v3_singles)
total_objects = sum(label_count.values())
cap = max(3, int(0.03 * (total_objects + 200)))
for p in v3_singles:
    try:
        out = json.loads(p["output"])
    except Exception:
        continue
    if not out or any(has_rel(c) for c in out):
        continue
    passage_m = re.search(r"TEXT:\n(.*)\n\nReturn ONLY the JSON array, no other text:$",
                          p["instruction"], re.S)
    if not passage_m:
        continue
    passage = passage_m.group(1)
    fixed = [fc for c in out if (fc := fix_concept(c, passage, inventory))]
    if not fixed:
        continue
    if any(label_count[c["concept_name"]] + 1 > cap for c in fixed):
        continue
    if add_pair(passage, fixed, p, "v3_single"):
        for c in fixed:
            label_count[c["concept_name"]] += 1
        kept_singles += 1

# ---------------------------------------------------------------- split
random.shuffle(pairs)


def split_key(p):
    return (p["doc_id"], (p.get("page_number") or 0) // 10)


groups = defaultdict(list)
for p in pairs:
    groups[split_key(p)].append(p)
train, test = [], []
for key in sorted(groups, key=lambda k: (hash(str(k)) % 1000)):
    (test if len(test) < 0.1 * len(pairs) else train).extend(groups[key])

for split, fname in ((train, "okf_train_pairs_v4.jsonl"), (test, "okf_test_pairs_v4.jsonl")):
    with open(TD / fname, "w") as f:
        for p in split:
            q = {k: v for k, v in p.items() if k != "_source"}
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- report
objs = []
for p in pairs:
    objs.extend(json.loads(p["output"]))
labels = Counter(c["concept_name"] for c in objs)
nrel = sum(1 for c in objs if has_rel(c))
nonempty = [p for p in pairs if json.loads(p["output"])]
multi = sum(1 for p in nonempty if len(json.loads(p["output"])) >= 2)
three = sum(1 for p in nonempty if len(json.loads(p["output"])) >= 3)
tb = sum(1 for p in pairs if "Deisenroth" in p["doc_id"])
report = {
    "version": "v4",
    "pairs_total": len(pairs), "train": len(train), "test": len(test),
    "sources": dict(Counter(p["_source"] for p in pairs)),
    "concept_objects": len(objs),
    "relation_coverage_pct": round(100 * nrel / max(len(objs), 1), 1),
    "empty_rate_pct": round(100 * (len(pairs) - len(nonempty)) / max(len(pairs), 1), 1),
    "multi_concept_pct_of_nonempty": round(100 * multi / max(len(nonempty), 1), 1),
    "pairs_with_3plus_concepts": three,
    "textbook_pair_pct": round(100 * tb / max(len(pairs), 1), 1),
    "max_label_share_pct": round(100 * labels.most_common(1)[0][1] / max(len(objs), 1), 1) if labels else 0,
    "top20_labels": labels.most_common(20),
    "v3_baseline": {"pairs": 852, "top_label_share_pct": 36.8,
                    "empty_rate_pct": 51.1, "relation_coverage_pct": 13.1},
}
json.dump(report, open(TD / "okf_dataset_report_v4.json", "w"), indent=2)
print(json.dumps({k: v for k, v in report.items() if k != "top20_labels"}, indent=2))
print("top labels:", labels.most_common(8))

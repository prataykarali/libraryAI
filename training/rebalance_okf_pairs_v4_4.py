#!/usr/bin/env python3
"""Build v4_4 train/test from cleaned v4_3 by fixing remaining SFT gaps.

Fixes addressed (without re-extracting PDFs):
  1. Math-for-ML heavy train  → cap Deisenroth share, upsample papers
  2. Few LoRA rows            → oversample Hu2021_LoRA (+ other papers)
  3. Low empty rate           → synthetic hard-negative [] examples
  4. Name F1 / exact match    → canonicalize names + write alias eval map

Does NOT call models. Safe to re-run.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training_data"
TRAIN_IN = DATA / "okf_train_pairs_v4_3.jsonl"
TEST_IN = DATA / "okf_test_pairs_v4_3.jsonl"
TRAIN_OUT = DATA / "okf_train_pairs_v4_4.jsonl"
TEST_OUT = DATA / "okf_test_pairs_v4_4.jsonl"
REPORT_OUT = DATA / "okf_dataset_report_v4_4.json"
ALIAS_OUT = DATA / "okf_name_aliases_v4_4.json"

SEED = 42
# Target composition of TRAIN (approx)
MAX_MATH_FRAC = 0.40          # was ~0.73
MIN_PAPER_FRAC = 0.50
TARGET_EMPTY_FRAC = 0.22      # was ~0.13
MIN_LORA_TRAIN = 40           # was 6 (via controlled oversample)
MAX_TRAIN = 900               # avoid explosion

MATH_DOC_SUBSTR = "Deisenroth_Math_For_ML"
LORA_DOC_SUBSTR = "Hu2021_LoRA"

# Shared instruction prefix style (matches existing pairs)
INSTR_PREFIX = (
    "You are an OKF extraction engine for the Archipelago knowledge graph.\n"
    "From the TEXT below, extract 1-5 teachable CONCEPTS as a JSON array.\n\n"
    "Each object MUST have exactly these keys: concept_name, concept_type, "
    "difficulty, summary, prerequisites, unlocks, related_to, tags.\n"
    "If the passage has no teachable AIML concept, return [].\n"
    "Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures.\n"
    "Do not extract celebrities, authors as concepts, or evaluation boilerplate.\n\n"
    "TEXT:\n"
)

NAME_ALIASES = {
    "lora": "Low-Rank Adaptation",
    "low rank adaptation": "Low-Rank Adaptation",
    "low-rank adaptation": "Low-Rank Adaptation",
    "fine tuning": "Fine-Tuning",
    "fine-tuning": "Fine-Tuning",
    "finetuning": "Fine-Tuning",
    "self attention": "Self-Attention",
    "self-attention": "Self-Attention",
    "graph rag": "Graph RAG",
    "graphrag": "Graph RAG",
    "vector rag": "Vector RAG",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "rag": "Retrieval-Augmented Generation",
    "bert": "BERT",
    "multi head attention": "Multi-Head Attention",
    "multi-head attention": "Multi-Head Attention",
    "scaled dot product attention": "Scaled Dot-Product Attention",
    "scaled dot-product attention": "Scaled Dot-Product Attention",
    "pre training": "Pre-Training",
    "pre-training": "Pre-Training",
    "pretraining": "Pre-Training",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_text(instruction: str) -> str:
    for m in ("TEXT:\n", "TEXT:"):
        if m in instruction:
            return instruction.split(m, 1)[1].strip()
    return instruction or ""


def is_math(r: dict) -> bool:
    return MATH_DOC_SUBSTR in (r.get("doc_id") or "")


def is_lora(r: dict) -> bool:
    return LORA_DOC_SUBSTR in (r.get("doc_id") or "")


def is_paper(r: dict) -> bool:
    d = r.get("doc_id") or ""
    return d.startswith("papers/") or d.endswith(".pdf") and "textbooks/" not in d


def is_empty(r: dict) -> bool:
    return (r.get("output") or "").strip() == "[]"


def norm_key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (name or "").lower())).strip()


def canon_name(name: str) -> str:
    k = norm_key(name)
    if k in NAME_ALIASES:
        return NAME_ALIASES[k]
    # light patches
    n = (name or "").strip()
    n = re.sub(r"(?i)\bfine-tuning\b", "Fine-Tuning", n)
    n = re.sub(r"(?i)\bself-attention\b", "Self-Attention", n)
    n = re.sub(r"(?i)\bgraph rag\b", "Graph RAG", n)
    n = re.sub(r"(?i)\bpre-training\b", "Pre-Training", n)
    return n


def canonicalize_row(row: dict) -> dict:
    r = deepcopy(row)
    try:
        arr = json.loads(r.get("output") or "[]")
    except Exception:
        arr = []
    if not isinstance(arr, list):
        arr = []
    new = []
    for c in arr:
        if not isinstance(c, dict):
            continue
        c = dict(c)
        c["concept_name"] = canon_name(c.get("concept_name") or "")
        # canonicalize string lists
        for key in ("prerequisites", "unlocks"):
            vals = []
            for x in c.get(key) or []:
                if isinstance(x, str) and x.strip():
                    vals.append(canon_name(x))
            c[key] = vals
        rel = []
        for x in c.get("related_to") or []:
            if isinstance(x, dict) and x.get("concept"):
                y = dict(x)
                y["concept"] = canon_name(y["concept"])
                rel.append(y)
            elif isinstance(x, str) and x.strip():
                rel.append({"concept": canon_name(x), "relation": "related_to"})
        c["related_to"] = rel
        if c["concept_name"]:
            new.append(c)
    r["output"] = json.dumps(new, ensure_ascii=False)
    return r


def make_empty_example(doc_id: str, chunk_id: str, text: str, page: int = 0) -> dict:
    return {
        "instruction": INSTR_PREFIX + text.strip(),
        "input": "",
        "output": "[]",
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "page_number": page,
        "section_title": "",
    }


# Hard negatives: model must NOT invent AIML concepts / celebrities
SYNTHETIC_EMPTY_TEXTS = [
    "Public figures mentioned in entertainment media include Taylor Swift, Britney Spears, "
    "and Justin Timberlake. Their personal lives dominate tabloid coverage this week.",
    "Winner=1 (Graph RAG). Answer 1 is better because it lists more celebrities across "
    "film, television, music, and sports without technical methodology.",
    "References\n[50] Alec Radford, Jeff Wu, Rewon Child. Language models are unsupervised "
    "multitask learners, 2019. URL https://example.com/paper.pdf",
    "[25] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented "
    "recurrent nets. Proceedings of the 28th International Conference, 2015.",
    "Table of Contents\n1. Introduction ........................ 1\n2. Related Work ..................... 5\n"
    "3. Experiments ........................ 12\nAcknowledgments .................... 40",
    "Copyright © 2021. All rights reserved. This PDF is for personal use only. "
    "No redistribution without permission of the publisher.",
    "The weather in Bangalore is expected to remain cloudy with a high of 28°C. "
    "Fans are excited about the weekend football match.",
    "idV is the identity endomorphism on V. Assume π is a projection. Calculate "
    "Im(idV − π) and ker(idV − π) as a function of Im(π) and ker(π). (Exercise only.)",
    "Funding: NSERC, Canada CIFAR AI Chair, and compute credits from the university cluster. "
    "We thank anonymous reviewers for helpful comments.",
    "Figure 3: Screenshot of the demo UI. (No additional technical definition in caption.)",
    "Behind the Tech with Kevin Scott features conversations with industry leaders. "
    "Episode transcript metadata only.",
    "Recipe: mix flour, sugar, and eggs. Bake at 180°C for 25 minutes until golden brown.",
]


def oversample(rows: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    if not rows:
        return []
    if len(rows) >= target_n:
        return rows[:target_n]
    out = list(rows)
    while len(out) < target_n:
        out.append(deepcopy(rng.choice(rows)))
    return out


def rebalance_train(train: list[dict], rng: random.Random) -> list[dict]:
    train = [canonicalize_row(r) for r in train]
    math_rows = [r for r in train if is_math(r)]
    paper_rows = [r for r in train if not is_math(r)]
    lora_rows = [r for r in train if is_lora(r)]
    empty_rows = [r for r in train if is_empty(r)]

    # Target train size first, then allocate quotas
    target_total = min(MAX_TRAIN, 700)
    max_math = max(50, int(target_total * MAX_MATH_FRAC))  # ~40%
    want_empty = int(target_total * TARGET_EMPTY_FRAC)       # ~22%
    n_non_empty = target_total - want_empty

    # 1) Cap math (among non-empty preference later)
    rng.shuffle(math_rows)
    math_empty = [r for r in math_rows if is_empty(r)]
    math_non = [r for r in math_rows if not is_empty(r)]
    # Keep a small math empty slice; rest of math budget is non-empty
    math_empty_keep = math_empty[: max(8, max_math // 10)]
    math_non_keep = math_non[: max(0, max_math - len(math_empty_keep))]
    math_rows = math_empty_keep + math_non_keep

    # 2) Boost LoRA + keep other papers
    lora_boosted = oversample(lora_rows, MIN_LORA_TRAIN, rng)
    non_lora_papers = [r for r in paper_rows if not is_lora(r)]
    # Oversample non-LoRA papers so papers fill most of non-empty budget
    papers_kept = non_lora_papers + lora_boosted
    paper_non_empty = [r for r in papers_kept if not is_empty(r)]
    # Ensure paper non-empty pool is large enough via oversample
    paper_target = max(len(paper_non_empty), int(n_non_empty * 0.65))
    paper_non_empty = oversample(paper_non_empty, paper_target, rng)

    # 3) Non-empty mix: papers first, then math fill (capped)
    math_ne = [r for r in math_rows if not is_empty(r)]
    non_empty_core = list(paper_non_empty)
    # add math until hit n_non_empty, respecting max_math on final out
    for r in math_ne:
        if len(non_empty_core) >= n_non_empty:
            break
        non_empty_core.append(r)
    # if still short, oversample papers again
    while len(non_empty_core) < n_non_empty and paper_non_empty:
        non_empty_core.append(deepcopy(rng.choice(paper_non_empty)))
    non_empty_core = non_empty_core[:n_non_empty]

    # Enforce math cap inside non-empty
    ne_math = [r for r in non_empty_core if is_math(r)]
    ne_other = [r for r in non_empty_core if not is_math(r)]
    if len(ne_math) > max_math:
        ne_math = ne_math[:max_math]
        # refill with papers
        while len(ne_other) + len(ne_math) < n_non_empty and paper_non_empty:
            ne_other.append(deepcopy(rng.choice(paper_non_empty)))
        non_empty_core = (ne_other + ne_math)[:n_non_empty]

    # 4) Empties: real + synthetic hard negatives
    existing_empty = [r for r in empty_rows]
    existing_empty += [r for r in math_rows if is_empty(r)]
    # dedupe by (doc, chunk)
    seen = set()
    uniq_empty = []
    for r in existing_empty:
        k = (r.get("doc_id"), r.get("chunk_id"))
        if k in seen:
            continue
        seen.add(k)
        uniq_empty.append(r)

    synth = []
    for i, text in enumerate(SYNTHETIC_EMPTY_TEXTS):
        synth.append(
            make_empty_example(
                doc_id="synthetic/hard_negatives.md",
                chunk_id=f"synth_empty_{i:03d}",
                text=text,
                page=i + 1,
            )
        )
    empties = uniq_empty + synth
    while len(empties) < want_empty:
        base = rng.choice(synth)
        e = deepcopy(base)
        e["chunk_id"] = f"synth_empty_x{len(empties):03d}"
        empties.append(e)
    empties = empties[:want_empty]

    out = non_empty_core + empties
    rng.shuffle(out)
    return out[:target_total]


def rebalance_test(test: list[dict]) -> list[dict]:
    """Keep test unique (no oversample); only canonicalize + ensure contam stays empty."""
    return [canonicalize_row(r) for r in test]


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    docs = Counter(r.get("doc_id") for r in rows)
    return {
        "n": n,
        "empty": sum(1 for r in rows if is_empty(r)),
        "empty_pct": round(100 * sum(1 for r in rows if is_empty(r)) / max(1, n), 2),
        "math": sum(1 for r in rows if is_math(r)),
        "math_pct": round(100 * sum(1 for r in rows if is_math(r)) / max(1, n), 2),
        "lora": sum(1 for r in rows if is_lora(r)),
        "lora_pct": round(100 * sum(1 for r in rows if is_lora(r)) / max(1, n), 2),
        "papers": sum(1 for r in rows if (r.get("doc_id") or "").startswith("papers/")),
        "docs": dict(docs.most_common(12)),
    }


def main():
    rng = random.Random(SEED)
    if not TRAIN_IN.exists() or not TEST_IN.exists():
        raise SystemExit(f"Need {TRAIN_IN} and {TEST_IN}")

    train_in = load_jsonl(TRAIN_IN)
    test_in = load_jsonl(TEST_IN)
    train_out = rebalance_train(train_in, rng)
    test_out = rebalance_test(test_in)

    write_jsonl(TRAIN_OUT, train_out)
    write_jsonl(TEST_OUT, test_out)

    # Alias map for alias-aware F1 at eval time
    surfaces = defaultdict(list)
    for canon in set(NAME_ALIASES.values()):
        surfaces[canon].append(canon)
    for k, v in NAME_ALIASES.items():
        surfaces[v].append(k)
        surfaces[v].append(v)
    alias_payload = {k: sorted(set(vs)) for k, vs in surfaces.items()}
    ALIAS_OUT.write_text(json.dumps(alias_payload, indent=2), encoding="utf-8")

    report = {
        "seed": SEED,
        "policy": {
            "max_math_frac": MAX_MATH_FRAC,
            "target_empty_frac": TARGET_EMPTY_FRAC,
            "min_lora_train": MIN_LORA_TRAIN,
            "max_train": MAX_TRAIN,
            "note": "LoRA boost uses oversampling of existing LoRA chunks; "
                    "for true diversity, re-extract more Hu2021 chunks into pairs.",
        },
        "before": {"train": stats(train_in), "test": stats(test_in)},
        "after": {"train": stats(train_out), "test": stats(test_out)},
        "outputs": {
            "train": str(TRAIN_OUT),
            "test": str(TEST_OUT),
            "aliases": str(ALIAS_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== v4_4 rebalance ===")
    print("BEFORE train:", report["before"]["train"])
    print("AFTER  train:", report["after"]["train"])
    print("AFTER  test: ", report["after"]["test"])
    print(f"wrote {TRAIN_OUT}")
    print(f"wrote {TEST_OUT}")
    print(f"wrote {ALIAS_OUT}")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()

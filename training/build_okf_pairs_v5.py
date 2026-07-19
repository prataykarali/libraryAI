#!/usr/bin/env python3
"""Build balanced OKF SFT pairs v5 — production-prompt aligned.

Root causes fixed vs v4.4 / lib-qwen collapse:
  1. instruction = EXACT EXTRACTION_PROMPT_V15 (same as okf.extraction serve path)
  2. no synthetic empty spam (≤12% empties, unique only)
  3. no 16× LoRA table oversampling (max multiplicity 2)
  4. math capped ~38%; papers kept unique-first
  5. strict checker loop rejects ungrounded / junk / schema-broken gold
  6. reports non-empty vs empty separately

Sources: cleaned unique rows from v4_2.bak (or v4_3), plus curated hard-negatives.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okf.config import EXTRACTION_PROMPT_V15, VALID_TYPES, VALID_DIFFICULTIES, VALID_RELATIONS

DATA = ROOT / "training_data"
# Prefer pristine cleaned unique set
SRC_CANDIDATES = [
    DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak",
    DATA / "okf_train_pairs_v4_3.jsonl",
    DATA / "okf_test_pairs_v4_3.jsonl",
]
# Also merge test v4_3 unique chunks then re-split (more LoRA unique for train)
TEST_SRC = DATA / "okf_test_pairs_v4_3.jsonl"
TRAIN_BAK = DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak"

OUT_TRAIN = DATA / "okf_train_pairs_v5.jsonl"
OUT_TEST = DATA / "okf_test_pairs_v5.jsonl"
OUT_REPORT = DATA / "okf_dataset_report_v5.json"
OUT_ALIAS = DATA / "okf_name_aliases_v5.json"

SEED = 42
TEST_RATIO = 0.18
MAX_MATH_FRAC = 0.38
TARGET_EMPTY_FRAC = 0.20
MAX_MULTIPLICITY = 2          # never more than 2 copies of same (doc,chunk)
MIN_NON_EMPTY_PAPER = 0.45    # share of non-empty that should be papers/syllabus
MAX_TRAIN = 900
MAX_NAME_WORDS = 6

# Curated UNIQUE hard negatives (not cloned 20×)
HARD_NEGATIVES = [
    ("synthetic/hard_neg_v5.md", "hn_celebs",
     "Public figures repeatedly mentioned in entertainment articles include Taylor Swift, "
     "Britney Spears, Travis Kelce, and Justin Timberlake."),
    ("synthetic/hard_neg_v5.md", "hn_winner_eval",
     "Empowerment: Winner=1 (Graph RAG). Answer 1 is better because it lists celebrities "
     "across film, television, music, and sports without technical methodology."),
    ("synthetic/hard_neg_v5.md", "hn_biblio",
     "[50] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. "
     "Language models are unsupervised multitask learners, 2019. URL https://example.com/x.pdf"),
    ("synthetic/hard_neg_v5.md", "hn_biblio2",
     "[25] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented "
     "recurrent nets. In Proceedings of the 28th International Conference, 2015."),
    ("synthetic/hard_neg_v5.md", "hn_toc",
     "Table of Contents\n1. Introduction ........................ 1\n2. Related Work ..................... 5\n"
     "3. Experiments ........................ 12\nAcknowledgments .................... 40"),
    ("synthetic/hard_neg_v5.md", "hn_copyright",
     "Copyright © 2021. All rights reserved. This PDF is for personal use only. "
     "No redistribution without permission of the publisher."),
    ("synthetic/hard_neg_v5.md", "hn_weather",
     "The weather in Bangalore is expected to remain cloudy with a high of 28°C. "
     "Fans are excited about the weekend football match."),
    ("synthetic/hard_neg_v5.md", "hn_funding",
     "Funding: NSERC, Canada CIFAR AI Chair, and compute credits from the university cluster. "
     "We thank anonymous reviewers for helpful comments."),
    ("synthetic/hard_neg_v5.md", "hn_recipe",
     "Recipe: mix flour, sugar, and eggs. Bake at 180°C for 25 minutes until golden brown."),
    ("synthetic/hard_neg_v5.md", "hn_podcast",
     "Behind the Tech with Kevin Scott features conversations with industry leaders. "
     "Episode transcript metadata only; no ML method is defined."),
    ("synthetic/hard_neg_v5.md", "hn_ui_caption",
     "Figure 3: Screenshot of the demo UI. (No additional technical definition in caption.)"),
    ("synthetic/hard_neg_v5.md", "hn_exercise",
     "idV is the identity endomorphism on V. Assume π is a projection. Calculate "
     "Im(idV − π) and ker(idV − π) as a function of Im(π) and ker(π). (Exercise only.)"),
]

NAME_ALIASES = {
    "lora": "Low-Rank Adaptation",
    "low rank adaptation": "Low-Rank Adaptation",
    "low-rank adaptation": "Low-Rank Adaptation",
    "fine tuning": "Fine-Tuning",
    "fine-tuning": "Fine-Tuning",
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
    "pre training": "Pre-Training",
    "pre-training": "Pre-Training",
    "pca": "Principal Component Analysis",
    "principal component analysis": "Principal Component Analysis",
}

PERSON_RE = re.compile(
    r"(?i)^(britney|taylor|justin|kevin\s+scott|alec\s+radford|jeff\s+wu|"
    r"ilya\s+sutskever|armand\s+joulin|tomas\s+mikolov)\b"
)
CONTAM_TEXT = re.compile(
    r"(?is)(britney|taylor\s+swift|winner\s*=\s*\d+\s*\(\s*graph\s*rag\s*\)|"
    r"public\s+figures\s+who\s+are\s+repeatedly|behind\s+the\s+tech\s+with\s+kevin)"
)
BIBLIO_RE = re.compile(r"(?m)^\s*\[\d+\]\s+[A-Z]")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_text(instruction: str) -> str:
    t = instruction or ""
    for m in ("TEXT:\n", "TEXT:"):
        if m in t:
            t = t.split(m, 1)[1]
            break
    # strip trailing serve footer if present
    for foot in (
        "\n\nReturn ONLY the JSON array, no other text:",
        "\nReturn ONLY the JSON array, no other text:",
        "Return ONLY the JSON array",
    ):
        if foot in t:
            t = t.split(foot)[0]
    return t.strip()


def make_instruction(text: str) -> str:
    """Exact production prompt — must match okf.extraction serve path."""
    # EXTRACTION_PROMPT_V15 ends with TEXT:\n{text}\n\nReturn ONLY...
    # but format uses {text} once; ensure we don't double braces issues
    return EXTRACTION_PROMPT_V15.format(text=text)


def norm_key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (name or "").lower())).strip()


def canon_name(name: str) -> str:
    n = (name or "").strip()
    k = norm_key(n)
    if k in NAME_ALIASES:
        return NAME_ALIASES[k]
    n = re.sub(r"(?i)\bfine-tuning\b", "Fine-Tuning", n)
    n = re.sub(r"(?i)\bself-attention\b", "Self-Attention", n)
    n = re.sub(r"(?i)\bgraph rag\b", "Graph RAG", n)
    n = re.sub(r"(?i)\bpre-training\b", "Pre-Training", n)
    return n


def grounded(name: str, text: str) -> bool:
    if not name or not text:
        return False
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if len(w) > 2]
    t = text.lower()
    if not words:
        return name.lower() in t
    hits = sum(1 for w in words if w in t)
    return hits >= max(1, (len(words) + 1) // 2)


# ───────────────────────── STRICT CHECKER ─────────────────────────

class CheckResult:
    def __init__(self):
        self.ok = True
        self.reasons: list[str] = []

    def fail(self, reason: str):
        self.ok = False
        self.reasons.append(reason)


def check_row(row: dict) -> CheckResult:
    """Strict gold quality gate."""
    cr = CheckResult()
    instr = row.get("instruction") or ""
    out = row.get("output") or ""
    text = extract_text(instr)

    # 1) Instruction must be production-style
    if "Scientific Method" not in instr and "EXAMPLE" not in instr:
        # production prompt contains EXAMPLE with Scientific Method
        if "1 to 5 teachable CONCEPTS" not in instr and "1-5 teachable" not in instr:
            cr.fail("instruction_missing_okf_role")
    if "TEXT:" not in instr:
        cr.fail("instruction_missing_TEXT")
    # Prefer full V15
    if len(instr) < 1500 and row.get("doc_id", "").startswith("synthetic/"):
        pass  # synthetic still uses full V15 via make_instruction
    if "concept_type: one of:" not in instr and "EXAMPLE" not in instr:
        # allow only if full EXTRACTION_PROMPT was used (has EXAMPLE)
        if "EXAMPLE" not in instr:
            cr.fail("instruction_not_production_v15")

    # 2) Output JSON
    try:
        arr = json.loads(out)
    except Exception:
        cr.fail("output_not_json")
        return cr
    if not isinstance(arr, list):
        cr.fail("output_not_list")
        return cr
    if len(arr) > 5:
        cr.fail("too_many_concepts")

    # Contaminated text must be empty
    if CONTAM_TEXT.search(text) and arr:
        cr.fail("contam_text_must_be_empty")
    if BIBLIO_RE.search(text[:500]) and len(text) < 900 and arr:
        # allow if strongly ML
        if not re.search(r"(?i)\b(learning|transformer|attention|gradient|embedding)\b", text):
            cr.fail("biblio_should_be_empty")

    for c in arr:
        if not isinstance(c, dict):
            cr.fail("concept_not_dict")
            continue
        for k in ("concept_name", "concept_type", "difficulty", "summary",
                  "prerequisites", "unlocks", "related_to", "tags"):
            if k not in c:
                cr.fail(f"missing_key:{k}")
        name = (c.get("concept_name") or "").strip()
        if not name:
            cr.fail("empty_name")
            continue
        if len(name.split()) > MAX_NAME_WORDS:
            cr.fail("name_too_long")
        if PERSON_RE.search(name):
            cr.fail("person_concept")
        if not grounded(name, text):
            cr.fail(f"ungrounded:{name}")
        ctype = c.get("concept_type")
        if ctype not in VALID_TYPES:
            cr.fail(f"bad_type:{ctype}")
        diff = c.get("difficulty")
        if diff not in VALID_DIFFICULTIES:
            cr.fail(f"bad_difficulty:{diff}")
        summary = (c.get("summary") or "").strip()
        if len(summary) < 15:
            cr.fail("summary_too_short")
        if re.search(r"(?i)winner\s*=\s*\d+", summary):
            cr.fail("eval_junk_summary")
        # self-ref
        nl = name.lower()
        for lst_name in ("prerequisites", "unlocks"):
            for x in c.get(lst_name) or []:
                if isinstance(x, str) and x.lower().strip() == nl:
                    cr.fail("self_ref")
        for x in c.get("related_to") or []:
            if isinstance(x, dict):
                rel = (x.get("relation") or "").lower()
                if rel and rel not in VALID_RELATIONS:
                    cr.fail(f"bad_relation:{rel}")

    # False empty: long technical passage with strong AIML terms should not be []
    # Skip pure exercises / formula-only debris
    if not arr and len(text) > 500 and not re.search(r"(?i)\b(exercise|calculate|assume now)\b", text):
        strong = re.findall(
            r"(?i)\b(transformer|attention|lora|fine-?tun|bert|embedding|"
            r"neural network|backpropagation|principal component)\b",
            text,
        )
        if len(set(w.lower() for w in strong)) >= 3 and not BIBLIO_RE.search(text[:300]):
            cr.fail("false_empty_on_teachable")

    return cr


def clean_concepts(arr: list, text: str) -> list:
    cleaned = []
    seen = set()
    for c in arr or []:
        if not isinstance(c, dict):
            continue
        name = canon_name((c.get("concept_name") or "").strip())
        if not name or name.lower() in seen:
            continue
        if PERSON_RE.search(name):
            continue
        if not grounded(name, text):
            continue
        if len(name.split()) > MAX_NAME_WORDS:
            continue
        summary = (c.get("summary") or "").strip()
        if len(summary) < 15 or re.search(r"(?i)winner\s*=", summary):
            summary = f"{name} is a teachable concept explained in this passage."
        if re.search(r"(?i)^(these|this|the following|we |answer )", summary):
            summary = f"{name}: {summary}"

        def scrub_str_list(items):
            out = []
            for x in items or []:
                if not isinstance(x, str):
                    continue
                cx = canon_name(x)
                if not cx or cx.lower() == name.lower():
                    continue
                out.append(cx)
            return out[:4]

        rel = []
        for x in c.get("related_to") or []:
            if isinstance(x, dict) and x.get("concept"):
                cn = canon_name(x["concept"])
                if not cn or cn.lower() == name.lower():
                    continue
                relation = (x.get("relation") or "uses").lower()
                if relation not in VALID_RELATIONS:
                    relation = "uses"
                rel.append({"concept": cn, "relation": relation})
            elif isinstance(x, str) and x.strip():
                cn = canon_name(x)
                if cn and cn.lower() != name.lower():
                    rel.append({"concept": cn, "relation": "uses"})
        ctype = c.get("concept_type") if c.get("concept_type") in VALID_TYPES else "definition"
        diff = c.get("difficulty") if c.get("difficulty") in VALID_DIFFICULTIES else "intermediate"
        tags = c.get("tags") if isinstance(c.get("tags"), list) else []
        tags = [str(t) for t in tags if t][:6]
        if not tags:
            tags = [re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]]

        cleaned.append({
            "concept_name": name,
            "concept_type": ctype,
            "difficulty": diff,
            "summary": summary[:220],
            "prerequisites": scrub_str_list(c.get("prerequisites")),
            "unlocks": scrub_str_list(c.get("unlocks")),
            "related_to": rel[:6],
            "tags": tags,
        })
        seen.add(name.lower())
    return cleaned[:5]


def row_from_source(src: dict) -> dict | None:
    text = extract_text(src.get("instruction") or "")
    if not text or len(text.strip()) < 40:
        return None
    # force empty on contam
    if CONTAM_TEXT.search(text):
        arr = []
    else:
        try:
            arr = json.loads(src.get("output") or "[]")
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
        arr = clean_concepts(arr, text)
        # biblio force empty
        if BIBLIO_RE.search(text[:400]) and len(text) < 900 and not re.search(
            r"(?i)\b(learning|transformer|attention|embedding)\b", text
        ):
            arr = []

    return {
        "instruction": make_instruction(text[:1800]),
        "input": "",
        "output": json.dumps(arr, ensure_ascii=False),
        "doc_id": src.get("doc_id") or "unknown",
        "chunk_id": src.get("chunk_id") or "chunk_0",
        "page_number": src.get("page_number") or 0,
        "section_title": src.get("section_title") or "",
    }


def is_math(r): return "Deisenroth" in (r.get("doc_id") or "")
def is_paper(r): return (r.get("doc_id") or "").startswith("papers/")
def is_lora(r): return "Hu2021_LoRA" in (r.get("doc_id") or "")
def is_empty(r): return (r.get("output") or "").strip() == "[]"
def is_synth(r): return (r.get("doc_id") or "").startswith("synthetic/")


def pool_unique_sources() -> list[dict]:
    """Merge all unique chunks across all versions, rebuild with V15."""
    by_key = {}
    paths = [
        DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak",
        DATA / "okf_test_pairs_v4_2.jsonl.v4_2.bak",
        DATA / "okf_train_pairs_v4_3.jsonl",
        DATA / "okf_test_pairs_v4_3.jsonl",
        DATA / "okf_train_pairs_v4_4.jsonl",
        DATA / "okf_test_pairs_v4_4.jsonl",
    ]
    for path in paths:
        if not path.exists():
            continue
        for r in load_jsonl(path):
            k = (r.get("doc_id"), r.get("chunk_id"))
            if k[0] and k[1]:
                by_key[k] = r  # later files overwrite ok
    rows = []
    for src in by_key.values():
        nr = row_from_source(src)
        if nr:
            rows.append(nr)
    # add unique hard negatives
    for doc, cid, text in HARD_NEGATIVES:
        rows.append({
            "instruction": make_instruction(text),
            "input": "",
            "output": "[]",
            "doc_id": doc,
            "chunk_id": cid,
            "page_number": 0,
            "section_title": "",
        })
    return rows



def strict_filter(rows: list[dict]) -> tuple[list[dict], Counter]:
    kept, stats = [], Counter()
    for r in rows:
        cr = check_row(r)
        if cr.ok:
            kept.append(r)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1
            for reason in cr.reasons:
                stats[f"drop:{reason}"] += 1
    return kept, stats


def balance(rows: list[dict], rng: random.Random) -> list[dict]:
    """Balance without destructive oversampling. Hard caps on math + empty rate."""
    by_key = {}
    for r in rows:
        by_key[(r["doc_id"], r["chunk_id"])] = r
    rows = list(by_key.values())

    empties = [r for r in rows if is_empty(r)]
    non_empty = [r for r in rows if not is_empty(r)]
    math_ne = [r for r in non_empty if is_math(r)]
    paper_ne = [
        r for r in non_empty
        if is_paper(r) or "syllabi" in (r.get("doc_id") or "") or "probable" in (r.get("doc_id") or "")
    ]
    other_ne = [r for r in non_empty if not is_math(r) and r not in paper_ne]

    # Fixed non-empty budget, hard math cap
    target_ne = min(len(non_empty), int(MAX_TRAIN * (1 - TARGET_EMPTY_FRAC)))
    max_math = max(20, int(target_ne * MAX_MATH_FRAC))
    rng.shuffle(math_ne)
    rng.shuffle(paper_ne)
    rng.shuffle(other_ne)
    math_ne = math_ne[:max_math]

    selected_ne: list[dict] = []
    # papers first (all unique), then other, then math (capped)
    for pool in (paper_ne, other_ne, math_ne):
        for r in pool:
            if len(selected_ne) >= target_ne:
                break
            selected_ne.append(r)
        if len(selected_ne) >= target_ne:
            break

    # Drop any math beyond cap (safety)
    kept, math_count = [], 0
    for r in selected_ne:
        if is_math(r):
            if math_count >= max_math:
                continue
            math_count += 1
        kept.append(r)
    selected_ne = kept

    # Mild boost (max 2×) for definitional LoRA only
    lora_ne = [r for r in selected_ne if is_lora(r)]
    for r in list(lora_ne):
        if sum(1 for x in selected_ne if is_lora(x)) >= min(18, max(1, len(lora_ne)) * MAX_MULTIPLICITY):
            break
        names = " ".join(c.get("concept_name", "") for c in json.loads(r["output"])).lower()
        if re.search(r"lora|low-rank|adaptation|fine-tun|adapter|peft", names):
            selected_ne.append(deepcopy(r))

    # Empties: unique only, force ~TARGET_EMPTY_FRAC of final set
    rng.shuffle(empties)
    synth_e = [r for r in empties if is_synth(r)]
    # Limit synthetic empties to prevent them from dominating the pool
    synth_e = synth_e[:15]
    real_e = [r for r in empties if not is_synth(r)]
    
    want_empty = max(
        len(synth_e),
        int(round(len(selected_ne) * TARGET_EMPTY_FRAC / max(1e-6, (1 - TARGET_EMPTY_FRAC)))),
    )
    want_empty = min(want_empty, max(12, int(MAX_TRAIN * 0.16)))
    
    selected_e = list(synth_e)
    for r in real_e:
        if len(selected_e) >= want_empty:
            break
        selected_e.append(r)
    selected_e = selected_e[:want_empty]

    out = selected_ne + selected_e
    rng.shuffle(out)
    return out[:MAX_TRAIN]


def split_train_test(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Chunk-level split with forced empty/math/paper balance on BOTH sides."""
    # collapse to one row per chunk (prefer first)
    by_key: dict[tuple, dict] = {}
    for r in rows:
        k = (r["doc_id"], r["chunk_id"])
        if k not in by_key:
            by_key[k] = r
        # keep extra LoRA copy only as multiplicity on train later
    items = list(by_key.values())

    empties = [r for r in items if is_empty(r)]
    non_empty = [r for r in items if not is_empty(r)]
    rng.shuffle(empties)
    rng.shuffle(non_empty)

    n_test = max(40, int(len(items) * TEST_RATIO))
    n_test_empty = max(4, int(n_test * TARGET_EMPTY_FRAC))
    n_test_empty = min(n_test_empty, max(1, len(empties) // 3))  # leave majority empties for train
    n_test_ne = n_test - n_test_empty

    test_e = empties[:n_test_empty]
    train_e = empties[n_test_empty:]

    # non-empty: stratify papers / math / other into both
    def split_pool(pool, n_te):
        rng.shuffle(pool)
        n_te = min(n_te, max(0, len(pool) // 4))  # keep most for train
        return pool[n_te:], pool[:n_te]

    papers = [r for r in non_empty if is_paper(r) or "syllabi" in (r.get("doc_id") or "")]
    maths = [r for r in non_empty if is_math(r)]
    others = [r for r in non_empty if r not in papers and r not in maths]

    # allocate test non-empty slots
    n_te_paper = max(6, n_test_ne // 3)
    n_te_math = max(4, n_test_ne // 4)
    n_te_other = max(0, n_test_ne - n_te_paper - n_te_math)

    tr_p, te_p = split_pool(papers, n_te_paper)
    tr_m, te_m = split_pool(maths, n_te_math)
    tr_o, te_o = split_pool(others, n_te_other)

    train = tr_p + tr_m + tr_o + train_e
    test = te_p + te_m + te_o + test_e

    # Duplicate all non-math non-empty chunks in train up to MAX_MULTIPLICITY = 2
    non_math_ne = [r for r in train if not is_math(r) and not is_empty(r)]
    for r in non_math_ne:
        train.append(deepcopy(r))

    # HARD math cap on final train (after dupes) — absolute ceiling
    non_math = [r for r in train if not is_math(r)]
    math_rows = [r for r in train if is_math(r)]
    rng.shuffle(math_rows)
    # math ≤ 38% of final; also absolute max so papers dominate
    max_math_train = min(
        len(math_rows),
        max(30, int(len(non_math) * MAX_MATH_FRAC / max(1e-6, (1 - MAX_MATH_FRAC)))),
        int((len(non_math) + len(math_rows)) * MAX_MATH_FRAC),
    )
    # Prefer: math count so that math/(math+non_math) ≤ MAX_MATH_FRAC
    max_math_train = min(len(math_rows), int(len(non_math) * MAX_MATH_FRAC / (1 - MAX_MATH_FRAC)))
    train = non_math + math_rows[:max_math_train]

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    docs = Counter(r["doc_id"] for r in rows)
    empty = sum(1 for r in rows if is_empty(r))
    # production prompt check
    v15 = sum(1 for r in rows if "EXAMPLE" in r["instruction"] and "Scientific Method" in r["instruction"])
    uniq = len({(r["doc_id"], r["chunk_id"]) for r in rows})
    lora_u = len({(r["doc_id"], r["chunk_id"]) for r in rows if is_lora(r)})
    return {
        "n": n,
        "unique_chunks": uniq,
        "empty": empty,
        "empty_pct": round(100 * empty / n, 2),
        "math": sum(1 for r in rows if is_math(r)),
        "math_pct": round(100 * sum(1 for r in rows if is_math(r)) / n, 2),
        "lora_rows": sum(1 for r in rows if is_lora(r)),
        "lora_unique": lora_u,
        "papers": sum(1 for r in rows if is_paper(r)),
        "prod_v15_prompt_pct": round(100 * v15 / n, 2),
        "mean_instr_len": round(sum(len(r["instruction"]) for r in rows) / n, 1),
        "docs_top": docs.most_common(10),
    }


def checker_gate(train: list[dict], test: list[dict]) -> tuple[bool, list[str]]:
    """Balancer acceptance criteria."""
    issues = []
    ts, es = stats(train), stats(test)
    if ts["prod_v15_prompt_pct"] < 99:
        issues.append(f"train prod prompt {ts['prod_v15_prompt_pct']}% < 99%")
    if not (8 <= ts["empty_pct"] <= 24):
        issues.append(f"train empty_pct {ts['empty_pct']} not in [8,24]")
    if ts["math_pct"] > 40:
        issues.append(f"train math_pct {ts['math_pct']} > 40")
    if ts["lora_unique"] < 3:
        issues.append(f"lora unique chunks {ts['lora_unique']} < 3")
    if ts["n"] < 250:
        issues.append(f"train n {ts['n']} < 250")
    if es["n"] < 35:
        issues.append(f"test n {es['n']} < 35")
    if es["empty_pct"] > 35:
        issues.append(f"test empty_pct {es['empty_pct']} > 35 (empties leaked to test)")
    # no train/test chunk leak
    tr = {(r["doc_id"], r["chunk_id"]) for r in train}
    te = {(r["doc_id"], r["chunk_id"]) for r in test}
    leak = tr & te
    if leak:
        issues.append(f"chunk leak {len(leak)}")
    # strict re-check sample
    fails = 0
    for r in train + test:
        if not check_row(r).ok:
            fails += 1
    if fails:
        issues.append(f"strict_check_failures {fails}")
    # multiplicity
    counts = Counter((r["doc_id"], r["chunk_id"]) for r in train)
    if counts and max(counts.values()) > MAX_MULTIPLICITY:
        issues.append(f"multiplicity {max(counts.values())} > {MAX_MULTIPLICITY}")
    return (len(issues) == 0), issues


def balancer_checker_loop(max_iters: int = 5) -> dict:
    rng = random.Random(SEED)
    history = []
    train = test = []
    filter_stats = Counter()

    for it in range(1, max_iters + 1):
        pool = pool_unique_sources()
        filtered, filter_stats = strict_filter(pool)
        balanced = balance(filtered, rng)
        # re-filter after balance
        balanced, fs2 = strict_filter(balanced)
        filter_stats += fs2
        train, test = split_train_test(balanced, rng)
        # re-apply production prompt (idempotent)
        for r in train + test:
            text = extract_text(r["instruction"])
            # if instruction already V15, extract_text still works
            if "EXAMPLE" not in r["instruction"]:
                r["instruction"] = make_instruction(text[:1800])
            else:
                # re-normalize from text inside
                r["instruction"] = make_instruction(text[:1800])

        ok, issues = checker_gate(train, test)
        history.append({
            "iter": it,
            "ok": ok,
            "issues": issues,
            "train": stats(train),
            "test": stats(test),
            "filter": dict(filter_stats),
        })
        print(f"[loop {it}] ok={ok} issues={issues}")
        print(f"  train={stats(train)}")
        print(f"  test={stats(test)}")
        if ok:
            break
        # adjust knobs slightly on failure
        if any("empty_pct" in i for i in issues):
            # nudge by adding/removing empties next iter via TARGET — mutate module is hard;
            # rebuild with different seed offset
            rng = random.Random(SEED + it * 17)
        if any("math_pct" in i for i in issues):
            rng = random.Random(SEED + it * 31)

    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_TEST, test)

    surfaces = defaultdict(list)
    for k, v in NAME_ALIASES.items():
        surfaces[v].append(k)
        surfaces[v].append(v)
    OUT_ALIAS.write_text(json.dumps({k: sorted(set(vs)) for k, vs in surfaces.items()}, indent=2))

    final_ok, final_issues = checker_gate(train, test)
    report = {
        "version": "v5",
        "seed": SEED,
        "final_ok": final_ok,
        "final_issues": final_issues,
        "policy": {
            "prompt": "EXTRACTION_PROMPT_V15 exact",
            "max_math_frac": MAX_MATH_FRAC,
            "target_empty_frac": TARGET_EMPTY_FRAC,
            "max_multiplicity": MAX_MULTIPLICITY,
            "sources": [str(TRAIN_BAK), str(TEST_SRC)],
        },
        "train": stats(train),
        "test": stats(test),
        "filter_stats": dict(filter_stats),
        "loop_history": history,
        "outputs": {
            "train": str(OUT_TRAIN),
            "test": str(OUT_TEST),
            "aliases": str(OUT_ALIAS),
        },
        "diagnosis_summary": {
            "lib_qwen_worsened": True,
            "root_causes": [
                "empty-collapse from v4.4 synthetic empty spam (22%)",
                "train/serve prompt mismatch (short vs EXTRACTION_PROMPT_V15)",
                "16x LoRA weak-table oversampling",
                "catastrophic forgetting of non-empty extraction on aura continue-tune",
            ],
            "keep_ingestion_model": "aura-qwen",
            "retrain_on": "okf_train_pairs_v5.jsonl with production prompt",
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    print("=== OKF v5 balancer ↔ strict checker loop ===")
    report = balancer_checker_loop()
    print("\n=== FINAL ===")
    print(json.dumps({
        "final_ok": report["final_ok"],
        "issues": report["final_issues"],
        "train": report["train"],
        "test": report["test"],
    }, indent=2))
    print(f"wrote {OUT_TRAIN}")
    print(f"wrote {OUT_TEST}")
    print(f"wrote {OUT_REPORT}")
    if not report["final_ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

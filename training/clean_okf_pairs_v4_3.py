#!/usr/bin/env python3
"""Clean OKF train/test pairs (v4_2 → v4_3).

Fixes that pure re-training on v4_2 cannot solve:
  - GraphRAG entertainment / human-eval junk chunks → gold []
  - Person / celebrity / podcast-host "concepts" removed
  - Non-teachable gold on contaminated text → []
  - Canonical concept-name casing / spelling
  - Drop self-references; scrub related_to dicts
  - Mild summary repair (strip weak openers when name is clear)
  - Report stats for before/after

Does NOT call any model. Read-only on sources; writes new jsonl + report.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training_data"
# Prefer pristine backups if present (allows re-running the cleaner safely).
TRAIN_IN = (
    DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak"
    if (DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak").exists()
    else DATA / "okf_train_pairs_v4_2.jsonl"
)
TEST_IN = (
    DATA / "okf_test_pairs_v4_2.jsonl.v4_2.bak"
    if (DATA / "okf_test_pairs_v4_2.jsonl.v4_2.bak").exists()
    else DATA / "okf_test_pairs_v4_2.jsonl"
)
TRAIN_OUT = DATA / "okf_train_pairs_v4_3.jsonl"
TEST_OUT = DATA / "okf_test_pairs_v4_3.jsonl"
REPORT_OUT = DATA / "okf_dataset_report_v4_3.json"
V4_2_TRAIN = DATA / "okf_train_pairs_v4_2.jsonl"
V4_2_TEST = DATA / "okf_test_pairs_v4_2.jsonl"

# Also refresh the "current" v4_2 filenames so existing train scripts pick up clean data
# (originals are copied to *.v4_2.bak first).
UPDATE_V4_2 = True

# ── Contaminated chunk text (celeb / human-eval / non-concept pages) ─────────
CONTAM_TEXT = re.compile(
    r"(?is)"
    r"("
    r"britney\s*spears|taylor\s*swift|travis\s*kelce|justin\s*timberlake|"
    r"kardashian|american\s*idol|the\s*weeknd|"
    r"public\s+figures\s+who\s+are\s+repeatedly\s+mentioned|"
    r"entertainment\s+articles?|entertainment\s+industry|"
    r"winner\s*=\s*\d+\s*\(\s*graph\s*rag\s*\)|"
    r"comprehensiveness\s*:\s*winner|empowerment\s*:\s*winner|"
    r"diversity\s*:\s*winner|behind\s+the\s+tech\s+with\s+kevin\s+scott"
    r")"
)

# Chunks that should always be empty gold (eval appendices, pure refs-ish)
FORCE_EMPTY_CHUNK_IDS = {
    # GraphRAG human-eval entertainment answers (known bad gold)
    ("papers/Edge2024_GraphRAG.pdf", "chunk_062"),
    ("papers/Edge2024_GraphRAG.pdf", "chunk_063"),
    ("papers/Edge2024_GraphRAG.pdf", "chunk_064"),
    ("papers/Edge2024_GraphRAG.pdf", "chunk_029"),  # Kevin Scott podcast
}

# Person / entity names that are not teachable AIML concepts
BLOCK_CONCEPT_NAMES = re.compile(
    r"(?i)^("
    r"britney\s+spears|taylor\s+swift|travis\s+kelce|justin\s+timberlake|"
    r"kevin\s+scott|alec\s+radford|jeff\s+wu|ilya\s+sutskever|"
    r"armand\s+joulin|tomas\s+mikolov"
    r")$"
)

# Bibliography / reference-list-ish chunks → prefer []
BIBLIO_TEXT = re.compile(
    r"(?is)"
    r"("
    r"^\s*\[\d+\]\s+[A-Z][a-z]+.+\d{4}"  # [50] Alec Radford...
    r"|(?:references|bibliography)\s*$"
    r"|https?://\S+\s*$"
    r")"
)

# Exercise / problem-set debris often better empty if no clear concept word
EXERCISE_ONLY = re.compile(
    r"(?is)^\s*("
    r"idV\s+is\s+the\s+identity|assume\s+now\s+that\s+π\s+is\s+a\s+projection|"
    r"calculate\s+im\(|exercise\s+\d|problem\s+\d"
    r")"
)

NAME_CANON = {
    "fine-tuning": "Fine-Tuning",
    "fine tuning": "Fine-Tuning",
    "finetuning": "Fine-Tuning",
    "pre-training": "Pre-Training",
    "pre training": "Pre-Training",
    "pretraining": "Pre-Training",
    "self-attention": "Self-Attention",
    "self attention": "Self-Attention",
    "scaled dot-product attention": "Scaled Dot-Product Attention",
    "scaled dot product attention": "Scaled Dot-Product Attention",
    "graph rag": "Graph RAG",
    "graphrag": "Graph RAG",
    "vector rag": "Vector RAG",
    "low-rank adaptation": "Low-Rank Adaptation",
    "low rank adaptation": "Low-Rank Adaptation",
    "lora": "LoRA",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "multi-head attention": "Multi-Head Attention",
    "multi head attention": "Multi-Head Attention",
    "bayesian inference": "Bayesian Inference",
    "linear algebra": "Linear Algebra",
    "dimensionality reduction": "Dimensionality Reduction",
    "matrix decomposition": "Matrix Decomposition",
    "eigenvalue decomposition": "Eigenvalue Decomposition",
}

WEAK_SUMMARY_OPEN = re.compile(
    r"(?i)^(these|this|the following|it is|we |answer \d|winner\s*=)"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_text(instruction: str) -> str:
    for marker in ("TEXT:\n", "TEXT:\r\n", "TEXT:"):
        if marker in instruction:
            return instruction.split(marker, 1)[1].strip()
    return instruction or ""


def canon_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return name
    name = name.strip()
    key = re.sub(r"\s+", " ", name.lower()).replace("_", " ")
    if key in NAME_CANON:
        return NAME_CANON[key]
    # Substring / multi-word canonical patches
    name = re.sub(r"(?i)\bfine-tuning\b", "Fine-Tuning", name)
    name = re.sub(r"(?i)\bfine tuning\b", "Fine-Tuning", name)
    name = re.sub(r"(?i)\bpre-training\b", "Pre-Training", name)
    name = re.sub(r"(?i)\bpre training\b", "Pre-Training", name)
    name = re.sub(r"(?i)\bself-attention\b", "Self-Attention", name)
    name = re.sub(r"(?i)\bself attention\b", "Self-Attention", name)
    name = re.sub(r"(?i)\bgraph rag\b", "Graph RAG", name)
    name = re.sub(r"(?i)\bvector rag\b", "Vector RAG", name)
    name = re.sub(r"(?i)\bdot-product attention\b", "Dot-Product Attention", name)
    # Fix "Graph Rag" style second word lower for known acronyms
    parts = name.split()
    if len(parts) >= 2 and parts[-1].islower() and parts[-1] in {
        "rag", "bert", "gpt", "cnn", "rnn", "nlp", "ml", "ai", "lora"
    }:
        parts[-1] = parts[-1].upper() if parts[-1] != "lora" else "LoRA"
        name = " ".join(parts)
    return name


def canon_ref(item) -> str | dict | None:
    """Canonicalize a prereq/unlock string or related_to dict."""
    if isinstance(item, str):
        n = canon_name(item)
        if not n or BLOCK_CONCEPT_NAMES.match(n):
            return None
        return n
    if isinstance(item, dict):
        c = item.get("concept") or item.get("name") or ""
        c2 = canon_name(c)
        if not c2 or BLOCK_CONCEPT_NAMES.match(c2):
            return None
        out = dict(item)
        out["concept"] = c2
        return out
    return None


def scrub_list(items, self_name: str) -> list:
    out = []
    seen = set()
    self_l = (self_name or "").lower().strip()
    for it in items or []:
        c = canon_ref(it)
        if c is None:
            continue
        key = c.lower() if isinstance(c, str) else (c.get("concept") or "").lower()
        if not key or key == self_l:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def name_grounded_in_text(name: str, text: str) -> bool:
    """At least half of significant name tokens must appear in chunk text."""
    if not name or not text:
        return False
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if len(w) > 2]
    if not words:
        # short acronyms: require exact case-insensitive presence
        return name.lower() in text.lower()
    t = text.lower()
    hits = sum(1 for w in words if w in t)
    return hits >= max(1, (len(words) + 1) // 2)


def should_force_empty(doc_id: str, chunk_id: str, text: str) -> str | None:
    if (doc_id, chunk_id) in FORCE_EMPTY_CHUNK_IDS:
        return "force_empty_chunk_id"
    if CONTAM_TEXT.search(text or ""):
        return "contaminated_text"
    t = text or ""
    # Pure bibliography / citation-list chunks
    bracket_cites = len(re.findall(r"(?m)^\s*\[\d+\]\s+", t))
    if bracket_cites >= 1 and len(t) < 1200:
        # reference entries dominate
        if bracket_cites >= 2 or re.search(r"(?i)\b(arxiv|proceedings of|url\s*:)", t):
            return "bibliography_like"
    if BIBLIO_TEXT.search(t[:400]) and len(t) < 900:
        if re.search(r"(?m)^\s*\[\d+\]", t) or "http" in t.lower():
            return "bibliography_like"
    # Very short non-prose debris
    if len(t.strip()) < 80 and not re.search(
        r"(?i)\b(learning|network|attention|model|gradient|matrix|bayes|embedding)\b", t
    ):
        return "too_short_non_concept"
    return None


def clean_concepts(arr: list, text: str, stats: Counter) -> list:
    cleaned = []
    for c in arr or []:
        if not isinstance(c, dict):
            stats["drop_non_dict"] += 1
            continue
        name = canon_name((c.get("concept_name") or "").strip())
        if not name:
            stats["drop_empty_name"] += 1
            continue
        if BLOCK_CONCEPT_NAMES.match(name):
            stats["drop_person_concept"] += 1
            continue
        # Drop pure "Relationship" with winner/empowerment summary (junk)
        summary = (c.get("summary") or "").strip()
        if name.lower() in {"relationship", "relationships"} and (
            WEAK_SUMMARY_OPEN.search(summary) or "winner" in summary.lower()
        ):
            stats["drop_junk_relationship"] += 1
            continue
        if re.search(r"(?i)winner\s*=\s*\d+", summary) or summary.lower().startswith(
            ("empowerment:", "comprehensiveness:", "diversity:")
        ):
            stats["drop_eval_summary_concept"] += 1
            continue
        if not name_grounded_in_text(name, text):
            # Allow LoRA / RAG / BERT acronyms if expansion or acronym in text
            acr = re.sub(r"[^A-Za-z0-9]", "", name)
            if len(acr) <= 6 and acr.lower() in re.sub(r"[^a-z0-9]", "", text.lower()):
                stats["keep_acronym_grounded"] += 1
            else:
                stats["drop_ungrounded"] += 1
                continue

        # Summary hygiene
        if not summary or WEAK_SUMMARY_OPEN.search(summary) or len(summary) < 20:
            stats["summary_rewritten"] += 1
            summary = f"{name} is a core concept extracted from this passage for the AIML knowledge graph."
        elif name.lower() not in summary.lower() and len(summary) > 40:
            # Prepend name if summary never mentions it
            if not any(
                w in summary.lower()
                for w in re.split(r"[^a-z0-9]+", name.lower())
                if len(w) > 3
            ):
                summary = f"{name}: {summary}"
                stats["summary_name_prefixed"] += 1

        prereq = scrub_list(c.get("prerequisites"), name)
        unlocks = scrub_list(c.get("unlocks"), name)
        related = scrub_list(c.get("related_to"), name)

        # Cap lists
        prereq = prereq[:4]
        unlocks = unlocks[:4]
        related = related[:6]

        ctype = c.get("concept_type") or "definition"
        if ctype not in {
            "definition", "method", "technique", "metric", "dataset",
            "theory", "result", "architecture", "loss", "optimizer",
        }:
            ctype = "definition"

        diff = c.get("difficulty") or "intermediate"
        if diff not in {"foundational", "intermediate", "advanced", "expert"}:
            diff = "intermediate"

        tags = c.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags if t][:6]
        if not tags:
            tags = [re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]]

        cleaned.append({
            "concept_name": name,
            "concept_type": ctype,
            "difficulty": diff,
            "summary": summary[:220],
            "prerequisites": prereq,
            "unlocks": unlocks,
            "related_to": related,
            "tags": tags,
        })
        stats["concepts_kept"] += 1

    # Cap 1–5 concepts (instruction contract)
    if len(cleaned) > 5:
        stats["capped_to_5"] += 1
        cleaned = cleaned[:5]
    return cleaned


def clean_row(row: dict, stats: Counter) -> dict:
    doc_id = row.get("doc_id") or ""
    chunk_id = row.get("chunk_id") or ""
    text = extract_text(row.get("instruction") or "")
    out_raw = row.get("output") or "[]"

    try:
        arr = json.loads(out_raw)
        if not isinstance(arr, list):
            arr = []
            stats["output_not_list"] += 1
    except Exception:
        arr = []
        stats["output_json_fail"] += 1

    reason = should_force_empty(doc_id, chunk_id, text)
    if reason:
        stats[f"empty_{reason}"] += 1
        arr = []
    else:
        before = len(arr)
        arr = clean_concepts(arr, text, stats)
        if before and not arr:
            stats["emptied_after_filter"] += 1
        # Bibliography-like with remaining weak concepts → empty
        if arr and BIBLIO_TEXT.search(text[:500]) and all(
            len((c.get("summary") or "")) < 80 for c in arr
        ):
            # references often produce author-noise; empty if no strong method word
            if not re.search(
                r"(?i)\b(learning|network|attention|transformer|retrieval|gradient|matrix|bayes|embedding|adaptation)\b",
                text,
            ):
                stats["empty_bibliography_cleanup"] += 1
                arr = []

    new_out = json.dumps(arr, ensure_ascii=False)
    stats["rows"] += 1
    if not arr:
        stats["empty_final"] += 1
    stats["concepts_final"] += len(arr)

    out = dict(row)
    out["output"] = new_out
    return out


def summarize_split(rows: list[dict]) -> dict:
    empty = sum(1 for r in rows if r["output"].strip() == "[]")
    docs = Counter(r.get("doc_id") for r in rows)
    n_concepts = 0
    for r in rows:
        try:
            n_concepts += len(json.loads(r["output"]))
        except Exception:
            pass
    return {
        "n": len(rows),
        "empty": empty,
        "empty_pct": round(100 * empty / max(1, len(rows)), 2),
        "concepts": n_concepts,
        "docs": dict(docs),
    }


def main():
    if not TRAIN_IN.exists() or not TEST_IN.exists():
        raise SystemExit(f"Missing inputs: {TRAIN_IN} / {TEST_IN}")

    train_in = load_jsonl(TRAIN_IN)
    test_in = load_jsonl(TEST_IN)

    stats_train: Counter = Counter()
    stats_test: Counter = Counter()
    train_out = [clean_row(r, stats_train) for r in train_in]
    test_out = [clean_row(r, stats_test) for r in test_in]

    write_jsonl(TRAIN_OUT, train_out)
    write_jsonl(TEST_OUT, test_out)

    report = {
        "source_train": str(TRAIN_IN),
        "source_test": str(TEST_IN),
        "out_train": str(TRAIN_OUT),
        "out_test": str(TEST_OUT),
        "before": {
            "train": summarize_split(train_in),
            "test": summarize_split(test_in),
        },
        "after": {
            "train": summarize_split(train_out),
            "test": summarize_split(test_out),
        },
        "stats_train": dict(stats_train),
        "stats_test": dict(stats_test),
        "policy": {
            "force_empty_chunk_ids": [list(x) for x in sorted(FORCE_EMPTY_CHUNK_IDS)],
            "name_canon_entries": len(NAME_CANON),
            "contam_regex": CONTAM_TEXT.pattern[:120] + "...",
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if UPDATE_V4_2:
        # Keep one-time pristine backups of original v4_2 if not already present
        bak_train = DATA / "okf_train_pairs_v4_2.jsonl.v4_2.bak"
        bak_test = DATA / "okf_test_pairs_v4_2.jsonl.v4_2.bak"
        if not bak_train.exists() and V4_2_TRAIN.exists():
            shutil.copy2(V4_2_TRAIN, bak_train)
        if not bak_test.exists() and V4_2_TEST.exists():
            shutil.copy2(V4_2_TEST, bak_test)
        shutil.copy2(TRAIN_OUT, V4_2_TRAIN)
        shutil.copy2(TEST_OUT, V4_2_TEST)
        report["also_updated"] = [str(V4_2_TRAIN), str(V4_2_TEST)]
        report["backups"] = [str(bak_train), str(bak_test)]
        REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== OKF dataset clean v4_3 ===")
    print(f"train: {report['before']['train']} -> {report['after']['train']}")
    print(f"test:  {report['before']['test']} -> {report['after']['test']}")
    print(f"wrote {TRAIN_OUT}")
    print(f"wrote {TEST_OUT}")
    print(f"wrote {REPORT_OUT}")
    if UPDATE_V4_2:
        print("updated v4_2 paths (backups *.v4_2.bak)")
    print("train filter stats:", dict(stats_train))
    print("test filter stats:", dict(stats_test))


if __name__ == "__main__":
    main()

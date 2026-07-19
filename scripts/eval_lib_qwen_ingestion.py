#!/usr/bin/env python3
"""Evaluate lib-qwen for OKF ingestion readiness.

Loads /home/.../lib-qwen (not aura-qwen), runs:
  1) Smoke probes (LoRA, empty/junk, math, Britney-class)
  2) Stratified sample of okf_test_pairs_v4_4.jsonl
  3) Exact + alias-normalized concept-name metrics

Writes report to training_data/lib_qwen_ingestion_eval.json
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LIB_QWEN = ROOT.parent / "lib-qwen"
TEST_PATH = ROOT / "training_data" / "okf_test_pairs_v4_4.jsonl"
ALIAS_PATH = ROOT / "training_data" / "okf_name_aliases_v4_4.json"
OUT_PATH = ROOT / "training_data" / "lib_qwen_ingestion_eval.json"

# Production prompt (system path)
import sys
sys.path.insert(0, str(ROOT))
from okf.config import EXTRACTION_PROMPT_V15, MAX_CHARS_TO_SLM  # noqa: E402


def extract_text_from_instruction(instr: str) -> str:
    for m in ("TEXT:\n", "TEXT:"):
        if m in instr:
            return instr.split(m, 1)[1].strip()
    return instr


def load_aliases() -> dict:
    if ALIAS_PATH.exists():
        return json.loads(ALIAS_PATH.read_text())
    return {}


def norm_name(name: str, aliases: dict) -> str:
    k = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (name or "").lower())).strip()
    if not k:
        return ""
    for canon, surfaces in aliases.items():
        ckey = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", canon.lower())).strip()
        surf = {
            re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()
            for s in surfaces
        }
        surf.add(ckey)
        if k in surf or k == ckey:
            return ckey
    return k


def parse_json_array(raw: str):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    a0, a1 = cleaned.find("["), cleaned.rfind("]")
    if a0 != -1 and a1 > a0:
        cleaned = cleaned[a0 : a1 + 1]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return data, None
        return None, "not_list"
    except Exception as e:
        return None, str(e)


def concept_names(arr) -> list[str]:
    if not arr:
        return []
    out = []
    for c in arr:
        if isinstance(c, dict):
            n = (c.get("concept_name") or "").strip()
            if n:
                out.append(n)
    return out


def name_sets(pred_names, gold_names, aliases):
    p = {norm_name(n, aliases) for n in pred_names if norm_name(n, aliases)}
    g = {norm_name(n, aliases) for n in gold_names if norm_name(n, aliases)}
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    prec = tp / (tp + fp) if tp + fp else (1.0 if not g and not p else 0.0)
    rec = tp / (tp + fn) if tp + fn else (1.0 if not g and not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact_count = len(pred_names) == len(gold_names)
    empty_exact = (len(pred_names) == 0 and len(gold_names) == 0)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_count": exact_count,
        "empty_exact": empty_exact,
        "pred": sorted(p),
        "gold": sorted(g),
    }


class LibQwenExtractor:
    def __init__(self, model_path: Path):
        print(f"Loading model from {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s  cuda={torch.cuda.is_available()}")

    def generate(self, prompt: str, max_new_tokens: int = 768) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            templated = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            templated = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(templated, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def extract_system_path(self, text: str) -> tuple[list | None, str, str | None]:
        """Same prompt path as okf.extraction.extract_okf_v15."""
        if len(text) > MAX_CHARS_TO_SLM:
            window = text[:MAX_CHARS_TO_SLM]
            cut = window.rfind("\n\n")
            if cut < MAX_CHARS_TO_SLM // 2:
                cut = max(window.rfind(". "), window.rfind("\n"))
            text = window[:cut] if cut > MAX_CHARS_TO_SLM // 2 else window
        prompt = EXTRACTION_PROMPT_V15.format(text=text)
        raw = self.generate(prompt)
        arr, err = parse_json_array(raw)
        return arr, raw, err


def load_test_rows():
    rows = [json.loads(l) for l in TEST_PATH.open() if l.strip()]
    return rows


def build_probe_set(rows: list[dict], rng_seed: int = 42) -> list[dict]:
    """Stratified probes: empties, LoRA, GraphRAG junk, math, papers."""
    import random

    rng = random.Random(rng_seed)
    probes = []
    used = set()

    def add(r, tag):
        k = (r.get("doc_id"), r.get("chunk_id"))
        if k in used:
            return
        used.add(k)
        item = dict(r)
        item["_tag"] = tag
        probes.append(item)

    empties = [r for r in rows if (r.get("output") or "").strip() == "[]"]
    lora = [r for r in rows if "Hu2021_LoRA" in (r.get("doc_id") or "")]
    graphrag = [r for r in rows if "GraphRAG" in (r.get("doc_id") or "")]
    math = [r for r in rows if "Deisenroth" in (r.get("doc_id") or "")]
    bert = [r for r in rows if "BERT" in (r.get("doc_id") or "") or "Vaswani" in (r.get("doc_id") or "") or "Lewis" in (r.get("doc_id") or "")]

    for r in empties[:6]:
        add(r, "empty_gold")
    for r in lora[:8]:
        add(r, "lora")
    for r in graphrag[:4]:
        add(r, "graphrag")
    for r in math[:6]:
        add(r, "math")
    for r in bert[:4]:
        add(r, "other_paper")

    # celeb-like text in instruction if any remain
    for r in rows:
        t = extract_text_from_instruction(r.get("instruction") or "")
        if re.search(r"(?i)britney|taylor swift|winner\s*=\s*1", t):
            add(r, "contam_text")

    # fill to ~30 with random
    rest = [r for r in rows if (r.get("doc_id"), r.get("chunk_id")) not in used]
    rng.shuffle(rest)
    for r in rest:
        if len(probes) >= 32:
            break
        add(r, "random")

    return probes


def schema_ok(arr) -> bool:
    if arr is None:
        return False
    if arr == []:
        return True
    req = {"concept_name", "concept_type", "difficulty", "summary"}
    for c in arr:
        if not isinstance(c, dict):
            return False
        if not req.issubset(c.keys()):
            return False
    return True


def has_celeb_concept(arr) -> bool:
    if not arr:
        return False
    blob = " ".join(
        (c.get("concept_name") or "") + " " + (c.get("summary") or "")
        for c in arr
        if isinstance(c, dict)
    )
    return bool(
        re.search(
            r"(?i)britney|taylor swift|justin timberlake|kevin scott|kardashian",
            blob,
        )
    )


def main():
    if not LIB_QWEN.exists():
        raise SystemExit(f"Missing model: {LIB_QWEN}")
    if not TEST_PATH.exists():
        raise SystemExit(f"Missing test: {TEST_PATH}")

    aliases = load_aliases()
    rows = load_test_rows()
    probes = build_probe_set(rows)
    print(f"Probes: {len(probes)}  (from test n={len(rows)})")
    print("Tags:", Counter(p["_tag"] for p in probes))

    ext = LibQwenExtractor(LIB_QWEN)

    results = []
    f1s_exact = []
    f1s_alias = []
    json_ok = 0
    schema = 0
    empty_gold_n = 0
    empty_exact_n = 0
    celeb_fail = 0
    self_ref = 0
    latencies = []

    for i, row in enumerate(probes):
        text = extract_text_from_instruction(row.get("instruction") or "")
        gold_arr, gerr = parse_json_array(row.get("output") or "[]")
        gold_names = concept_names(gold_arr or [])

        t0 = time.time()
        pred_arr, raw, err = ext.extract_system_path(text)
        dt = time.time() - t0
        latencies.append(dt)

        pred_names = concept_names(pred_arr) if pred_arr is not None else []
        ok_json = err is None and pred_arr is not None
        if ok_json:
            json_ok += 1
        if ok_json and schema_ok(pred_arr):
            schema += 1

        # exact string F1
        pset = {n.lower().strip() for n in pred_names}
        gset = {n.lower().strip() for n in gold_names}
        tp = len(pset & gset)
        fp = len(pset - gset)
        fn = len(gset - pset)
        pe = tp / (tp + fp) if tp + fp else (1.0 if not gset and not pset else 0.0)
        re_ = tp / (tp + fn) if tp + fn else (1.0 if not gset and not pset else 0.0)
        f1e = 2 * pe * re_ / (pe + re_) if pe + re_ else 0.0
        f1s_exact.append(f1e)

        alias_m = name_sets(pred_names, gold_names, aliases)
        f1s_alias.append(alias_m["f1"])

        if not gold_names:
            empty_gold_n += 1
            if not pred_names:
                empty_exact_n += 1

        if has_celeb_concept(pred_arr or []):
            celeb_fail += 1

        # self-ref
        for c in pred_arr or []:
            if not isinstance(c, dict):
                continue
            nm = (c.get("concept_name") or "").lower().strip()
            for lst in (c.get("prerequisites") or [], c.get("unlocks") or []):
                for x in lst:
                    if isinstance(x, str) and x.lower().strip() == nm:
                        self_ref += 1

        rec = {
            "i": i,
            "tag": row["_tag"],
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "latency_s": round(dt, 2),
            "json_ok": ok_json,
            "schema_ok": ok_json and schema_ok(pred_arr),
            "parse_err": err,
            "gold_names": gold_names,
            "pred_names": pred_names,
            "f1_exact": round(f1e, 4),
            "f1_alias": round(alias_m["f1"], 4),
            "raw_preview": (raw or "")[:220],
        }
        results.append(rec)
        print(
            f"[{i+1}/{len(probes)}] {row['_tag']:12} "
            f"json={ok_json} f1e={f1e:.2f} f1a={alias_m['f1']:.2f} "
            f"{dt:.1f}s  gold={gold_names[:3]} pred={pred_names[:3]}"
        )

    n = len(probes)
    by_tag = {}
    for tag in sorted({r["tag"] for r in results}):
        sub = [r for r in results if r["tag"] == tag]
        by_tag[tag] = {
            "n": len(sub),
            "json_ok": sum(1 for r in sub if r["json_ok"]) / len(sub),
            "mean_f1_exact": sum(r["f1_exact"] for r in sub) / len(sub),
            "mean_f1_alias": sum(r["f1_alias"] for r in sub) / len(sub),
        }

    report = {
        "model": str(LIB_QWEN),
        "test_set": str(TEST_PATH),
        "n_probes": n,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics": {
            "json_valid_pct": round(100 * json_ok / n, 2),
            "schema_ok_pct": round(100 * schema / n, 2),
            "mean_f1_exact": round(sum(f1s_exact) / n, 4),
            "mean_f1_alias": round(sum(f1s_alias) / n, 4),
            "empty_gold_n": empty_gold_n,
            "empty_exact_match_pct": round(
                100 * empty_exact_n / empty_gold_n, 2
            )
            if empty_gold_n
            else None,
            "celeb_concept_failures": celeb_fail,
            "self_ref_count": self_ref,
            "latency_s_mean": round(sum(latencies) / n, 2),
            "latency_s_p50": round(sorted(latencies)[n // 2], 2),
            "latency_s_max": round(max(latencies), 2),
        },
        "by_tag": by_tag,
        "ingestion_gate": {},
        "examples": results,
    }

    m = report["metrics"]
    # Ingestion readiness heuristic
    gates = {
        "json_valid_ge_95": m["json_valid_pct"] >= 95,
        "schema_ok_ge_90": m["schema_ok_pct"] >= 90,
        "alias_f1_ge_0_35": m["mean_f1_alias"] >= 0.35,
        "no_celeb_concepts": m["celeb_concept_failures"] == 0,
        "empty_match_ge_40_if_present": (
            m["empty_exact_match_pct"] is None
            or m["empty_exact_match_pct"] >= 40
        ),
        "self_ref_low": m["self_ref_count"] <= max(2, n // 10),
    }
    report["ingestion_gate"] = {
        "checks": gates,
        "pass_count": sum(1 for v in gates.values() if v),
        "total": len(gates),
        "ready_for_pilot_ingest": all(
            gates[k]
            for k in (
                "json_valid_ge_95",
                "schema_ok_ge_90",
                "no_celeb_concepts",
                "self_ref_low",
            )
        )
        and m["mean_f1_alias"] >= 0.30,
        "ready_for_full_replace": all(gates.values()) and m["mean_f1_alias"] >= 0.45,
        "notes": [
            "Probe set is stratified (~32), not full 197 — treat as go/no-go smoke.",
            "ready_for_pilot_ingest: can point extraction at lib-qwen for limited re-ingest with cleanup still on.",
            "ready_for_full_replace: replace aura-qwen as default only if full 197 eval still strong.",
        ],
    }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n========== SUMMARY ==========")
    print(json.dumps(report["metrics"], indent=2))
    print("gates:", report["ingestion_gate"])
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Live latency gates for pilot readiness (Session 8).

Opt-in only:
    RUN_LIVE_E2E=1 .venv/bin/python -m pytest tests/e2e/test_latency.py -v

Warm-up: 3 queries before measurement (embeddings / model load).
Report: p50, p95, p99 across 10 timed queries after warm-up.
Gates:
  - deterministic RAG (no synthesis)  < 2.0s warm
  - full pipeline + Qwen wording       < 8.0s warm
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import requests

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_E2E") != "1",
        reason="Requires RUN_LIVE_E2E=1",
    ),
]

# Stable curriculum-style queries (repeat as needed for N samples).
QUERY_POOL = [
    "What is LoRA?",
    "What is attention?",
    "What is BERT?",
    "What is RAG?",
    "What are prerequisites for fine-tuning?",
    "Explain multi-head attention",
    "What is parameter-efficient fine-tuning?",
    "How does retrieval-augmented generation work?",
    "What is self-attention?",
    "What is low-rank adaptation?",
]

WARMUP_N = 3
MEASURE_N = 10
DETERMINISTIC_BUDGET_S = 2.0
SYNTHESIS_BUDGET_S = 8.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list (p in 0..100)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Nearest-rank: index = ceil(p/100 * n) - 1
    k = max(0, min(len(sorted_vals) - 1, int((p / 100.0) * len(sorted_vals) + 0.999999) - 1))
    # Use linear interpolation between closest ranks for smoother p95/p99 on n=10
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _latency_report(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "mean": statistics.fmean(samples) if samples else float("nan"),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "min": ordered[0] if ordered else float("nan"),
        "max": ordered[-1] if ordered else float("nan"),
        "samples": samples,
    }


def _post_chat_timed(inference_url: str, payload: dict, timeout: float = 120) -> float:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{inference_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, f"chat returned {resp.status_code}: {resp.text[:200]}"
    assert resp.text.strip(), "empty chat body"
    return elapsed


def _warmup(inference_url: str, synthesis: bool) -> None:
    for i in range(WARMUP_N):
        q = QUERY_POOL[i % len(QUERY_POOL)]
        payload = {"query": q, "mode": "rag_synthesis"}
        if synthesis:
            payload["synthesis"] = True
        _post_chat_timed(inference_url, payload, timeout=180)


def _measure(inference_url: str, synthesis: bool, n: int = MEASURE_N) -> list[float]:
    samples = []
    for i in range(n):
        q = QUERY_POOL[i % len(QUERY_POOL)]
        payload = {"query": q, "mode": "rag_synthesis"}
        if synthesis:
            payload["synthesis"] = True
        samples.append(_post_chat_timed(inference_url, payload, timeout=180))
    return samples


def _print_report(label: str, report: dict) -> None:
    print(
        f"\n[{label}] n={report['n']}  "
        f"p50={report['p50']:.3f}s  p95={report['p95']:.3f}s  p99={report['p99']:.3f}s  "
        f"mean={report['mean']:.3f}s  min={report['min']:.3f}s  max={report['max']:.3f}s"
    )


def _ollama_reachable(ollama_url: str, timeout: float = 3) -> bool:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_deterministic_rag_under_2s(INFERENCE_URL):
    """Warm deterministic RAG (no Qwen wording) must stay under 2s (p95)."""
    try:
        ready = requests.get(f"{INFERENCE_URL}/api/readiness", timeout=5)
    except requests.exceptions.RequestException as exc:
        pytest.fail(f"Inference server unreachable: {exc}")
    assert ready.status_code in (200, 503), ready.text[:200]

    _warmup(INFERENCE_URL, synthesis=False)
    samples = _measure(INFERENCE_URL, synthesis=False, n=MEASURE_N)
    report = _latency_report(samples)
    _print_report("deterministic_rag", report)

    # Gate: warm p95 under budget (mean and p50 must also be sane)
    assert report["p50"] < DETERMINISTIC_BUDGET_S, (
        f"deterministic p50={report['p50']:.3f}s >= {DETERMINISTIC_BUDGET_S}s budget"
    )
    assert report["p95"] < DETERMINISTIC_BUDGET_S, (
        f"deterministic p95={report['p95']:.3f}s >= {DETERMINISTIC_BUDGET_S}s budget; "
        f"full report={report}"
    )


def test_qwen_synthesis_under_8s(INFERENCE_URL, OLLAMA_URL):
    """Warm full pipeline + Qwen wording pass must stay under 8s (p95)."""
    if not _ollama_reachable(OLLAMA_URL):
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}; required for synthesis latency gate")

    try:
        ready = requests.get(f"{INFERENCE_URL}/api/readiness", timeout=5)
    except requests.exceptions.RequestException as exc:
        pytest.fail(f"Inference server unreachable: {exc}")
    assert ready.status_code in (200, 503), ready.text[:200]

    _warmup(INFERENCE_URL, synthesis=True)
    samples = _measure(INFERENCE_URL, synthesis=True, n=MEASURE_N)
    report = _latency_report(samples)
    _print_report("qwen_synthesis", report)

    assert report["p50"] < SYNTHESIS_BUDGET_S, (
        f"synthesis p50={report['p50']:.3f}s >= {SYNTHESIS_BUDGET_S}s budget"
    )
    assert report["p95"] < SYNTHESIS_BUDGET_S, (
        f"synthesis p95={report['p95']:.3f}s >= {SYNTHESIS_BUDGET_S}s budget; "
        f"full report={report}"
    )

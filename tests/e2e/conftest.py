"""Fixtures for live end-to-end tests (opt-in via RUN_LIVE_E2E=1).

These fixtures are scoped to tests under tests/e2e/ only.
"""

from __future__ import annotations

import os
import time

import fitz
import pytest
import requests


# ── Live service URL fixtures (env-overridable) ───────────────────────────────

@pytest.fixture(scope="session")
def GRAPH_URL() -> str:
    return os.getenv("GRAPH_URL", "http://localhost:5050").rstrip("/")


@pytest.fixture(scope="session")
def INFERENCE_URL() -> str:
    return os.getenv("INFERENCE_URL", "http://localhost:5051").rstrip("/")


@pytest.fixture(scope="session")
def CHAT_URL() -> str:
    return os.getenv("CHAT_URL", "http://localhost:5052").rstrip("/")


@pytest.fixture(scope="session")
def OLLAMA_URL() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


# ── Health / readiness waiter ─────────────────────────────────────────────────

def wait_until_ready(url: str, timeout: float = 60, interval: float = 1) -> requests.Response:
    """Poll ``url`` until it returns a successful HTTP response or timeout.

    Accepts any 2xx as ready. Raises ``TimeoutError`` if the deadline expires.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=min(interval, 5))
            if 200 <= resp.status_code < 300:
                return resp
            # Some readiness endpoints return 503 when not fully ready — keep polling.
            last_error = RuntimeError(f"HTTP {resp.status_code} from {url}")
        except requests.exceptions.RequestException as exc:
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"Service at {url} not ready within {timeout}s: {last_error}")


# ── Synthetic multi-page PDF for ingestion tests ──────────────────────────────

@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a small multi-page PDF with distinctive ML / LoRA content.

    Content is designed so a later chat query for "Synthetic Concept X" or
    "LoRA" / fine-tuning can assert presence after ingestion.
    """
    pdf_path = tmp_path / "synthetic_concept_x_lora.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 50), "Synthetic Concept X", fontsize=16)
    page1.insert_text(
        (50, 100),
        (
            "Synthetic Concept X is a controlled test concept used to verify "
            "end-to-end library ingestion. It describes a parameter-efficient "
            "fine-tuning approach related to Low-Rank Adaptation (LoRA)."
        ),
        fontsize=11,
    )
    page1.insert_text(
        (50, 180),
        (
            "In practice, Synthetic Concept X freezes the pretrained backbone "
            "and injects small trainable adapters so that fine-tuning is cheap "
            "and citation-preserving for RAG systems."
        ),
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_text((50, 50), "LoRA and Fine-Tuning", fontsize=16)
    page2.insert_text(
        (50, 100),
        (
            "LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank "
            "matrices. Fine-tuning with LoRA reduces the number of trainable "
            "parameters while retaining most of the full fine-tuning quality."
        ),
        fontsize=11,
    )
    page2.insert_text(
        (50, 180),
        (
            "This document uniquely mentions the phrase "
            "ARCHIPELAGO_E2E_MARKER_SYNTHETIC_CONCEPT_X so automated tests can "
            "query for it after PDF ingestion completes."
        ),
        fontsize=11,
    )

    page3 = doc.new_page()
    page3.insert_text((50, 50), "Summary", fontsize=16)
    page3.insert_text(
        (50, 100),
        (
            "Summary: Synthetic Concept X, LoRA, and fine-tuning form a small "
            "curriculum for multi-page PDF extraction and graph linking tests."
        ),
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

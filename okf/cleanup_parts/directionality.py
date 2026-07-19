"""Prerequisite directionality constraints and bibliography detection.

Used during post-extraction cleanup so inverted REQUIRES never enter the graph
(e.g. Neural Network requiring Graph Neural Network).
"""
from __future__ import annotations

import re

from okf.cleanup_parts._base import _DIFFICULTY_RANK
from okf.util import create_concept_id

# section titles that should never yield extracted concepts
_BIBLIO_SECTION_RE = re.compile(
    r"(?i)^\s*("
    r"references?|bibliography|works\s+cited|literature\s+cited|"
    r"acknowledg(?:e)?ments?|appendix(\s+[a-z0-9]+)?|"
    r"notes\s+and\s+references|further\s+reading"
    r")\b"
)

# (target_id, prereq_id) pairs that must never appear as target REQUIRES prereq
_BLOCKED_PREREQ_PAIRS = frozenset({
    ("neural_network", "graph_neural_network"),
    ("neural_network", "dimensionality_reduction"),
    ("linear_regression", "neural_network"),
    ("eigenvalue", "neural_network"),
    ("linear_algebra", "linear_regression"),
    ("linear_regression", "support_vector_machine"),
    ("matrix_inverse", "linear_regression"),
    ("linear_algebra", "dimensionality_reduction"),
})

# Name fragments: if prereq name contains specialized form of target, drop
# e.g. target "Neural Network", prereq "Graph Neural Network"
_SPECIALIZATION_MARKERS = (
    "graph neural", "convolutional neural", "recurrent neural",
    "deep neural", "spiking neural", "capsule network",
)


def is_bibliography_section(section_title: str) -> bool:
    """True when the section is bibliography / references / similar noise."""
    title = (section_title or "").strip()
    if not title:
        return False
    if _BIBLIO_SECTION_RE.search(title):
        return True
    low = title.lower()
    # mid-title hits common in papers
    for token in ("references", "bibliography", "works cited"):
        if token in low and len(low) < 80:
            return True
    return False


def _name_id(name: str) -> str:
    return create_concept_id(name or "")


def is_inverted_prerequisite(target_name: str, prereq_name: str,
                             target_diff: str = "", prereq_diff: str = "") -> bool:
    """Return True when ``prereq`` should NOT be a prerequisite of ``target``."""
    t_id = _name_id(target_name)
    p_id = _name_id(prereq_name)
    if not t_id or not p_id or t_id == p_id:
        return True
    if (t_id, p_id) in _BLOCKED_PREREQ_PAIRS:
        return True

    t_rank = _DIFFICULTY_RANK.get((target_diff or "").lower().strip())
    p_rank = _DIFFICULTY_RANK.get((prereq_diff or "").lower().strip())
    # Difficulty labels are noisy in pilot extracts — only drop egregious gaps
    if t_rank is not None and p_rank is not None and (p_rank - t_rank) >= 2:
        return True

    t_low = (target_name or "").lower()
    p_low = (prereq_name or "").lower()
    # Specialization of the same family should not be a prereq of the parent
    if "neural network" in t_low and any(m in p_low for m in _SPECIALIZATION_MARKERS):
        if "graph" in p_low or "convolutional" in p_low or "recurrent" in p_low:
            if "graph" not in t_low and "convolutional" not in t_low:
                return True
    return False


def prune_inverted_prerequisites(okf_results: list) -> int:
    """Drop inverted / blocked prerequisite links from extraction records.

    Returns number of prereq entries removed.
    """
    # Build difficulty index by lower name
    diff_by_name = {}
    for r in okf_results:
        name = (r.get("concept_name") or "").strip()
        if name:
            diff_by_name[name.lower()] = r.get("difficulty") or "intermediate"

    removed = 0
    for r in okf_results:
        target = r.get("concept_name") or ""
        t_diff = r.get("difficulty") or "intermediate"
        clean = []
        for p in r.get("prerequisites") or []:
            if not isinstance(p, str) or not p.strip():
                continue
            p_diff = diff_by_name.get(p.lower(), "")
            if is_inverted_prerequisite(target, p, t_diff, p_diff):
                removed += 1
                continue
            clean.append(p)
        r["prerequisites"] = clean
    return removed

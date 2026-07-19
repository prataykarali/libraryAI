"""Proxy accuracy metrics for the extraction pipeline."""

from collections import Counter

from okf.canonicalize import canonicalize_name, _concept_key

# Fuzzy fallback for gold comparison (P1.8). rapidfuzz backs thefuzz (see
# requirements.txt); import it directly for token_set_ratio scoring.
try:
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover - optional dependency
    _rf_fuzz = None

# Minimum token_set_ratio for two canonically-unmatched names to count as TP.
FUZZY_MATCH_THRESHOLD = 90

__all__ = [
    'Counter',
    'FUZZY_MATCH_THRESHOLD',
    '_concept_key',
    'canonicalize_name',
    '_rf_fuzz',
]

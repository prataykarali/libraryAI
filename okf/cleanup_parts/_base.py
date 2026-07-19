"""Post-extraction cleanup: junk filters, grounding, dedup, cycle breaking."""

import re

from okf.canonicalize import (
    apply_canonicalization,
    build_canonical_map,
    canonicalize_name,
    is_same_concept_reference,
)
from okf.config import VALID_RELATIONS
from okf.util import _dedupe_dicts, _record_sources

# ---------------------------------------------------------------------------
# Direction / cycle helpers
# ---------------------------------------------------------------------------
# Optional domain-specific partial ordering. Used only to resolve direct A↔B
# prerequisite cycles when both concepts are in the dict. Leave empty for a
# fully generic pipeline; populate via a domain config file if desired.
_FOUNDATIONAL_PRIORITY = {}


_DIFFICULTY_RANK = {"foundational": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# Noise filters used in both extraction normalization and post-cleanup.
_JUNK_NAME_RE = re.compile(
    r"(?i)\b(authors?|contributors?|chairs?|funding|acknowledg|thank|grants?|projects?|"
    r"universit|institute|canada\s+cifar|research\s+chair|cifar\s+ai|nserc|"
    r"phd\s+program|fellowship|scholarship|discovery\s+grant|ai\s+chairs?|"
    r"computational\s+resources\s+provided|table\s+\d+|caption)\b|"
    r"best\s+model\s+without|underlined"
)
_NUMERIC_NAME_RE = re.compile(r"^\d[\d\s%\.x\-/]*$")
_FORMULA_OR_VALUE_NAME_RE = re.compile(
    r"(?i)([{}=∑∆ΔΦ]|\.{2,}|"
    r"\bvs\.?\b|\bcomparison\b|\b\d+%|\b\d+\s*of\s+tokens\b|"
    r"\breplacement\s+token\s*\(\d|"
    r"\b(system|hyperparameter)\s+dev\b|"
    r"\b(dev|test)\s+(f1|acc|accuracy|score)\b|"
    r"^test\s+scores?$|"
    r"\bstate-of-the-art\b.*\b(bleu|f1|accuracy|score)\b|"
    r"\bbatch\b|\bwithout\s+gold\s+access\b)"
)

# Minimal stopword list for grounding-overlap checks (content words only).
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "at", "from", "which", "can", "we",
    "you", "they", "their", "our", "not", "but", "if", "then", "than",
    "also", "such", "each", "other", "into", "over", "under", "between",
    "through", "when", "where", "how", "what", "all", "any", "some", "more",
    "most", "used", "using", "use", "based", "one", "two", "may", "will",
    "would", "should", "could", "has", "have", "had", "do", "does", "done",
    "there", "about", "both", "very", "given", "well", "only", "called",
}

__all__ = [
    "VALID_RELATIONS",
    "_DIFFICULTY_RANK",
    "_FOUNDATIONAL_PRIORITY",
    "_JUNK_NAME_RE",
    "_NUMERIC_NAME_RE",
    "_FORMULA_OR_VALUE_NAME_RE",
    "_STOPWORDS",
    "_dedupe_dicts",
    "_record_sources",
    "apply_canonicalization",
    "build_canonical_map",
    "canonicalize_name",
    "is_same_concept_reference",
    "re",
]

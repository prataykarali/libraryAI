"""Concept name canonicalization: alias resolution and fuzzy merging."""

import re

try:
    from thefuzz import fuzz as _fuzz
except Exception:  # pragma: no cover - optional dependency
    _fuzz = None


def _concept_key(name: str) -> str:
    """Normalize a concept name for exact/near-exact self-reference checks."""
    key = re.sub(r"\([^)]*\)", "", name or "").lower()
    key = key.replace("-", " ")
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    words = [w for w in key.split() if w not in {"full", "basic", "standard", "general"}]
    normalized = []
    for word in words:
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return " ".join(normalized)


def is_same_concept_reference(a: str, b: str) -> bool:
    """True only for exact/near-exact aliases, not broader prerequisites."""
    a_key = _concept_key(a)
    b_key = _concept_key(b)
    return bool(a_key and b_key and a_key == b_key)


# Common aliases to collapse.
# Keep this domain-agnostic. Add domain-specific aliases (e.g. biology, law)
# by editing this dict or loading a domain_aliases.json file.
ALIAS_MAP = {
    "gpt": "GPT",
    "openai gpt": "GPT",
    "lora": "Low-Rank Adaptation",
    "qlora": "Quantized Low-Rank Adaptation",
    "multi-head self-attention": "Multi-Head Attention",
    "multi head self attention": "Multi-Head Attention",
    "preference datasets": "Preference Data",
}

# Generic trailing words that don't distinguish concepts: "Transformer" and
# "Transformer Architecture" must merge into one node or edges land on a
# placeholder split from the sourced concept.
_GENERIC_SUFFIXES = (
    "architecture", "mechanism", "method", "methods",
    "technique", "techniques", "search",
)


def _merge_key(canon_lower: str) -> str:
    """Reduce a canonical name to a merge key: strip one generic suffix word
    and depluralize the last word, so near-identical names share a key."""
    words = canon_lower.split()
    if len(words) > 1 and words[-1] in _GENERIC_SUFFIXES:
        words = words[:-1]
    if words:
        w = words[-1]
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
            w = w[:-1]
        words[-1] = w
    return " ".join(words)


def canonicalize_name(name: str) -> str:
    """Normalize a concept name to a canonical form."""
    if not name:
        return ""

    # Strip whitespace
    name = name.strip()

    # Collapse snake_case / underscores the SLM sometimes emits so
    # "my_concept_name" merges with "My Concept Name".
    name = name.replace("_", " ")
    name = re.sub(r'\s+', ' ', name).strip()

    # Remove trailing periods
    name = name.rstrip(".")

    # Check alias map
    name_lower = name.lower()
    if name_lower in ALIAS_MAP:
        return ALIAS_MAP[name_lower]

    # Remove short parenthetical abbreviations: "Some Concept (SC)" → "Some Concept"
    name = re.sub(r'\s*\([^)]{1,10}\)\s*$', '', name)

    # If it's a full sentence (has a verb-like pattern), truncate
    if len(name) > 60:
        # Try to keep just the first noun phrase
        parts = name.split(",")
        name = parts[0].strip()
    if len(name) > 60:
        parts = name.split(" - ")
        name = parts[0].strip()

    # Title case — but leave short all-caps acronyms (GPT, BERT, LSTM) alone
    # so they don't become "Gpt"/"Bert".
    name = name.strip()
    if name == name.lower():
        name = name.title()
    elif name == name.upper() and len(name) > 5:
        name = name.title()

    return name


def build_canonical_map(okf_results: list) -> dict:
    """
    Build a mapping from raw concept names → canonical names.
    Deduplicates similar concepts via fuzzy matching.
    """
    raw_names = set()
    for result in okf_results:
        cn = result.get("concept_name", "")
        if isinstance(cn, str) and cn:
            raw_names.add(cn)
        for p in result.get("prerequisites", []):
            if isinstance(p, str) and p:
                raw_names.add(p)
        for u in result.get("unlocks", []):
            if isinstance(u, str) and u:
                raw_names.add(u)
        for r in result.get("related_to", []):
            if isinstance(r, dict) and isinstance(r.get("concept"), str):
                raw_names.add(r.get("concept", ""))

    # Canonicalize all names
    canon_map = {}
    canonical_set = {}  # canonical_lower → canonical
    merge_keys = {}  # merge key (suffix-stripped, depluralized) → canonical

    for raw in sorted(raw_names, key=lambda n: len(canonicalize_name(n))):
        if not raw:
            continue
        canon = canonicalize_name(raw)
        canon_lower = canon.lower()

        # Exact merge-key hit: "Transformer Architecture" → "Transformer",
        # "Language Models" → "Language Model". Shorter name (seen first due
        # to the sort) wins.
        mk = _merge_key(canon_lower)
        matched = False
        if len(canon_lower) > 3 and mk in merge_keys:
            existing_canon = merge_keys[mk]
            chosen = existing_canon if len(existing_canon) <= len(canon) else canon
            canon_map[raw] = chosen
            if chosen == canon:
                canonical_set[canon_lower] = canon
                merge_keys[mk] = canon
                for k, v in canon_map.items():
                    if v == existing_canon:
                        canon_map[k] = canon
            matched = True

        # Block merging distinct multi-word concepts by fuzzy name similarity.
        # We use a high cutoff (90) and fall back to strict equality if thefuzz
        # is not installed.
        if not matched:
            for existing_lower, existing_canon in canonical_set.items():
                similar = False
                if _fuzz is not None:
                    similar = _fuzz.ratio(canon_lower, existing_lower) >= 90
                else:
                    similar = canon_lower == existing_lower

                if similar and len(canon_lower) > 3:
                    # Prefer the shorter, more canonical spelling
                    chosen = existing_canon if len(existing_canon) <= len(canon) else canon
                    canon_map[raw] = chosen
                    if chosen == canon:
                        canonical_set[canon_lower] = canon
                        # Re-map anything that pointed to the old longer name
                        for k, v in canon_map.items():
                            if v == existing_canon:
                                canon_map[k] = canon
                    matched = True
                    break

        if not matched:
            canon_map[raw] = canon
            canonical_set[canon_lower] = canon
            merge_keys.setdefault(mk, canon)

    return canon_map


def apply_canonicalization(okf_results: list, canon_map: dict) -> list:
    """Apply canonical name mapping to all concept references in OKF results."""
    for result in okf_results:
        raw_name = result.get("concept_name", "")
        result["concept_name"] = canon_map.get(raw_name, canonicalize_name(raw_name))

        result["prerequisites"] = [
            canon_map.get(p, canonicalize_name(p))
            for p in result.get("prerequisites", [])
            if isinstance(p, str) and p.strip()
        ]
        result["unlocks"] = [
            canon_map.get(u, canonicalize_name(u))
            for u in result.get("unlocks", [])
            if isinstance(u, str) and u.strip()
        ]

        new_related = []
        for r in result.get("related_to", []):
            if isinstance(r, dict) and r.get("concept"):
                r["concept"] = canon_map.get(r["concept"], canonicalize_name(r["concept"]))
                new_related.append(r)
        result["related_to"] = new_related

        # Remap per-relation provenance keys ("kind:name_lower") so they still
        # match after their target names were canonicalized above.
        prov = result.get("relation_provenance")
        if isinstance(prov, dict) and prov:
            lower_map = {raw.lower(): canon for raw, canon in canon_map.items()}
            remapped = {}
            for key, src in prov.items():
                kind, _, target = key.partition(":")
                canon_target = lower_map.get(target, target)
                remapped[f"{kind}:{canon_target.lower()}"] = src
            result["relation_provenance"] = remapped

    return okf_results

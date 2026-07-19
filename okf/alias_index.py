"""alias_index.py — Alias and acronym index for okf ingestion package."""
import re
from okf.canonicalize import ALIAS_MAP, canonicalize_name

def extract_acronym(text: str) -> str:
    """Build an acronym from significant words (e.g. Low-Rank Adaptation -> LRA)."""
    if not text or not isinstance(text, str):
        return ""
    paren = re.findall(r"\(([A-Za-z0-9]{2,12})\)", text)
    if paren:
        return paren[0].upper()
    words = re.split(r"[\s\-_/.,]+", text.upper())
    stop = {"OF", "FOR", "AND", "OR", "THE", "A", "AN", "IN", "ON", "AT", "TO", "FROM", "BY", "WITH"}
    significant = [w for w in words if w and w not in stop and len(w) > 1 and w.isalpha()]
    computed = "".join(w[0] for w in significant)
    return computed

def generate_aliases_for_name(name: str) -> list[str]:
    """Generate all variations/aliases for a single concept name."""
    aliases = {name.lower().strip()}
    # Base form without parentheses: "Low-Rank Adaptation (LoRA)" -> "Low-Rank Adaptation"
    base = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    if base:
        aliases.add(base.lower())
    # Extract acronyms from parentheses
    for p in re.findall(r"\(([A-Za-z0-9]{2,12})\)", name):
        aliases.add(p.lower())
    # Extract acronym of whole name
    acr = extract_acronym(name)
    if acr and len(acr) >= 2:
        aliases.add(acr.lower())
    
    # Common variations: replace hyphens/underscores with spaces
    expanded = set(aliases)
    for a in aliases:
        expanded.add(a.replace("-", " ").replace("_", " "))
        expanded.add(a.replace(" ", ""))
        expanded.add(a.replace(" ", "-"))
    return sorted(list(expanded))

def build_alias_index(concept_names: list[str]) -> dict[str, str]:
    """Maps all aliases, acronyms, and canonicalized versions to their canonical name."""
    alias_index = {}
    for name in concept_names:
        if not name:
            continue
        canon = canonicalize_name(name)
        # Register the canonical name itself
        alias_index[canon.lower().strip()] = canon
        # Register original name
        alias_index[name.lower().strip()] = canon
        
        # Register generated aliases
        for alias in generate_aliases_for_name(canon):
            alias_index[alias] = canon
        for alias in generate_aliases_for_name(name):
            alias_index[alias] = canon
            
    # Also register ALIAS_MAP entries
    for key, val in ALIAS_MAP.items():
        alias_index.setdefault(key.lower().strip(), canonicalize_name(val))
        
    return alias_index

def resolve_concept_name(name: str, alias_index: dict[str, str]) -> str:
    """Resolve a name to its canonical name using the alias index, falling back to canonicalized name."""
    if not name:
        return ""
    norm = name.lower().strip()
    if norm in alias_index:
        return alias_index[norm]
    # Check variations
    variations = [
        norm.replace("-", " ").replace("_", " "),
        norm.replace(" ", ""),
        norm.replace(" ", "-")
    ]
    for var in variations:
        if var in alias_index:
            return alias_index[var]
            
    return canonicalize_name(name)

def names_equivalent(a: str, b: str, alias_index: dict[str, str]) -> bool:
    """Check if two names refer to the same concept in an alias-aware manner."""
    if not a or not b:
        return False
    res_a = resolve_concept_name(a, alias_index)
    res_b = resolve_concept_name(b, alias_index)
    return res_a.lower().strip() == res_b.lower().strip()

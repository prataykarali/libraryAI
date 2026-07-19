"""Alias / acronym helpers and PDF deep-link formatters."""
from __future__ import annotations

import re

from archipelago.inference import state as st

def _node_name(node):
    """Return the display name used by both graph exports and Kuzu records."""
    return (node or {}).get("label") or (node or {}).get("name") or "Unnamed concept"


def extract_acronym(text):
    """Build an acronym from significant words (e.g. Low-Rank Adaptation → LRA).

    Also returns parenthetical forms when present: ``Low-Rank Adaptation (LoRA)``
    yields both the computed acronym and ``LORA`` from the parentheses.
    """
    if not text or not isinstance(text, str):
        return ""
    paren = re.findall(r"\(([A-Za-z0-9]{2,12})\)", text)
    words = re.split(r"[\s\-_/.,]+", text.upper())
    stop = {"OF", "FOR", "AND", "OR", "THE", "A", "AN", "IN", "ON", "AT", "TO", "FROM", "BY", "WITH", "IS", "IT"}
    significant = [w for w in words if w and w not in stop and len(w) > 1 and w.isalpha()]
    computed = "".join(w[0] for w in significant)
    # Prefer explicit parenthetical acronym when present (LoRA, RAG, BERT).
    if paren:
        return paren[0].upper()
    if computed in stop:
        return ""
    return computed


def generate_aliases(concept):
    """Alias list for a concept: name, label, tags, acronyms, simple variants."""
    aliases = set()
    name = (concept or {}).get("name") or (concept or {}).get("label") or ""
    label = (concept or {}).get("label") or ""
    cid = (concept or {}).get("id") or (concept or {}).get("name") or ""
    for raw in (name, label):
        if not raw:
            continue
        aliases.add(raw.lower().strip())
        # Strip parenthetical: "Low-Rank Adaptation (LoRA)" → both parts
        base = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
        if base:
            aliases.add(base.lower())
        for p in re.findall(r"\(([A-Za-z0-9]{2,12})\)", raw):
            aliases.add(p.lower())
        acr = extract_acronym(raw)
        if acr and len(acr) >= 2:
            aliases.add(acr.lower())
    for tag in (concept or {}).get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            aliases.add(tag.lower().strip())
    # Add explicit mapping keys from _CORE_CONCEPT_MAP
    if cid:
        for k, targets in _CORE_CONCEPT_MAP.items():
            if any(t in cid.lower() for t in targets):
                aliases.add(k.lower())
    # Common compact forms
    expanded = set(aliases)
    for a in aliases:
        expanded.add(a.replace("-", " ").replace("_", " "))
        expanded.add(a.replace(" ", ""))
        expanded.add(a.replace(" ", "-"))
    # Prevent common stop words and extremely generic single-token words from being aliases
    stop_words = {"of", "for", "and", "or", "the", "a", "an", "in", "on", "at", "to", "from", "by", "with", "is", "it", "that", "this", "these", "those", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did"}
    generic_words = {"model", "models", "network", "networks", "data", "system", "systems", "learning", "training", "method", "methods", "analysis", "function", "functions", "value", "values", "variable", "variables", "error", "errors", "parameter", "parameters", "process", "processes", "layer", "layers", "step", "steps", "task", "tasks", "set", "sets", "weight", "weights", "vector", "vectors", "base", "bases", "language", "logic"}
    
    exact_names = {name.lower().strip(), label.lower().strip()}
    filtered = set()
    for a in expanded:
        if not a or not a.strip():
            continue
        al = a.lower().strip()
        if al in exact_names:
            filtered.add(al)
        elif al not in stop_words and al not in generic_words:
            filtered.add(al)
    return list(filtered)



def pdf_page_url(doc_id, page_number=None):
    """Deep-link into a served PDF: /pdfs/{doc_id}#page=N."""
    if not doc_id:
        return ""
    url = f"{st.PDF_BASE_URL}/pdfs/{doc_id}"
    if isinstance(page_number, int) and page_number > 0:
        url += f"#page={page_number}"
    return url


def markdown_pdf_link(label, doc_id, page_number=None):
    """Markdown link for in-bubble pedagogy (clickable #page=N)."""
    url = pdf_page_url(doc_id, page_number)
    if not url:
        return label or ""
    safe_label = (label or doc_id or "source").replace("]", "\\]")
    return f"[{safe_label}]({url})"


# Explicit core concept mappings: query alias → expected core concept_id substring
_CORE_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "lora": ("low_rank_adaptation",),
    "low-rank adaptation": ("low_rank_adaptation",),
    "low rank adaptation": ("low_rank_adaptation",),
    "rag": ("retrieval_augmented", "retrieval-augmented", "vector_rag", "rag"),
    "rags": ("retrieval_augmented", "retrieval-augmented", "vector_rag", "rag"),
    "retrieval-augmented generation": ("retrieval_augmented",),
    "retrieval augmented generation": ("retrieval_augmented",),
    "vector rag": ("vector_rag", "retrieval_augmented"),
    "attention": ("attention_mechanism", "self_attention", "multi_head_attention"),
    "self-attention": ("self_attention", "attention_mechanism"),
    "attention mechanism": ("attention_mechanism",),
    "bert": ("bert", "bidirectional_transformer", "bidirectional_encoder"),
    "pca": ("principal_component_analysis", "pca"),
    "transformer": ("transformer", "attention_mechanism"),
    "gpt": ("gpt", "language_model"),
    "graph rag": ("graph_rag", "graphrag"),
    "graphrag": ("graph_rag", "graphrag"),
    "fine-tuning": ("fine_tuning", "fine-tuning"),
    "fine tuning": ("fine_tuning", "fine-tuning"),
    "finetuning": ("fine_tuning",),
    # Agents / frameworks — map onto best available pilot-graph nodes
    "agent": ("ai_agent", "react_synergizing", "tool_use"),
    "agents": ("ai_agent", "react_synergizing", "tool_use"),
    "ai agent": ("ai_agent", "react_synergizing"),
    "ai agents": ("ai_agent", "react_synergizing"),
    "agentic": ("ai_agent", "react_synergizing"),
    "tool use": ("tool_use", "ai_agent", "react_synergizing"),
    "tool calling": ("tool_use", "ai_agent", "react_synergizing"),
    "react": ("react_synergizing",),
    "langchain": ("langchain", "ai_agent", "react_synergizing"),
    "llamaindex": ("llamaindex", "llama_index", "retrieval_augmented"),
    "llama index": ("llamaindex", "llama_index"),
    "framework": ("langchain", "llamaindex"),
    "frameworks": ("langchain", "llamaindex"),
    "neural network": ("neural_network",),
    "neural networks": ("neural_network",),
    "deep learning": ("deep_learning", "neural_network"),
    "adam": ("adam_optimizer",),
    "chain-of-thought": ("chain_of_thought_prompting",),
    "chain of thought": ("chain_of_thought_prompting",),
}

# Variant/sub-concept title fragments that indicate this is NOT a core node
_VARIANT_FRAGS = (
    "applied to", "via ", "with ", "batching", "speedup", "scaling factor",
    "variant", "efficient ", "extended", "for ", "based on", "using ",
)


def _core_concept_bonus(query, concept_id, label, aliases=None):
    """Boost short/core concept nodes when the query is an acronym or short alias.

    Penalizes long family/variant titles when the user asked simply for a core
    concept (e.g. 'LoRA', 'BERT', 'PCA', 'attention').
    Works generically for all concepts via _CORE_CONCEPT_MAP and variant frags;
    no longer requires hard-coding each concept explicitly.
    """
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    # Strip question framing for matching
    q_core = re.sub(
        r"^(what is|what's|whats|explain|tell me about|how does|define|"
        r"how to|how do i|build an?|create an?|suggest|recommend)\s+",
        "",
        q,
        flags=re.I,
    ).strip(" ?!.")
    # Fold short plurals for map lookup (rags→rag, agents→agent)
    q_core_lookup = q_core
    if q_core.endswith("s") and len(q_core) <= 8 and q_core[:-1] in _CORE_CONCEPT_MAP:
        q_core_lookup = q_core[:-1]
    label_l = (label or "").lower()
    cid = (concept_id or "").lower()
    alias_set = {a.lower() for a in (aliases or [])}

    bonus = 0.0

    # 1) Exact alias/id/label match → strong bonus
    if q_core in alias_set or q_core_lookup in alias_set or q_core == label_l or q_core.replace(" ", "_") == cid:
        bonus += 0.35

    # 2) Core concept map: if q_core maps to expected concept id fragments, boost them;
    #    penalize nodes whose ids DON'T match but whose label mentions the query token.
    core_targets = _CORE_CONCEPT_MAP.get(q_core) or _CORE_CONCEPT_MAP.get(q_core_lookup)
    # Also try last 1–3 tokens (e.g. "build an ai agent" → "ai agent" / "agent")
    if not core_targets:
        parts = q_core.split()
        for n in (3, 2, 1):
            if len(parts) >= n:
                frag = " ".join(parts[-n:])
                core_targets = _CORE_CONCEPT_MAP.get(frag)
                if core_targets:
                    break
    if core_targets:
        if any(t in cid for t in core_targets):
            bonus += 0.45
        else:
            # Is this a variant/sub-concept of the same concept? Mild penalty.
            if q_core in label_l and len(label_l.split()) > 3:
                bonus -= 0.20

    # 3) Attention sub-node special case: penalize "LoRA applied to attention"
    if q_core in ("attention", "self-attention", "attention mechanism"):
        if "lora" in label_l or "applied" in label_l:
            bonus -= 0.28

    # 4) Acronym short-query heuristics (applies to any ≤6-char alphabetic query)
    q_acr = extract_acronym(q_core) if " " in q_core else q_core.upper()
    if len(q_core) <= 6 and q_core.isalpha():
        if q_core in alias_set:
            bonus += 0.30
        # Prefer concise concept titles over long descriptive/variant names
        word_count = len(label_l.split())
        if word_count <= 4:
            bonus += 0.12
        elif word_count >= 6:
            bonus -= 0.18
        # Generic variant/sub-concept title penalty for ALL short acronym queries
        if any(x in label_l for x in _VARIANT_FRAGS) and word_count > 3:
            bonus -= 0.22

    return bonus


def _clean_query_words(text):
    """Return a set of cleaned, normalised words from `text`.

    Only trailing punctuation and a plain plural 's' are removed — NOT
    leading characters.  The previous implementation used str.strip() with 's'
    in the character set, which also ate leading 's', mangling words like
    'suggest' → 'uggest' and 'various' → 'variou'.
    """
    import re as _re
    result = set()
    for word in (text or "").lower().split():
        # Strip leading/trailing punctuation but NOT the letter 's' from the front
        cleaned = word.strip("?,.!-:;()[]{}\"'")
        # Optionally strip a trailing 's' that looks like a simple plural
        # (only if the stem has >2 chars to avoid stripping too aggressively)
        if cleaned.endswith("s") and len(cleaned) > 3:
            stem = cleaned[:-1]
        else:
            stem = cleaned
        # Use the fuller form if both are long enough
        for tok in (cleaned, stem):
            if len(tok) > 1:
                result.add(tok)
    return result



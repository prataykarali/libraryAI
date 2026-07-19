"""AIML scope classification for chat routing (query-time only).

Uses ARCHIPELAGO_OLLAMA_MODEL for a tiny yes/no classify. Never loads the
ingestion/extraction model (aura-qwen). Keywords are assistive only.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from archipelago.inference import state as st

# Three generic refusals only — intent_gate chooses which bucket, not keywords.
OUT_OF_SCOPE_MESSAGE = (
    "That one's a bit outside what this library covers — my shelves hold AI/ML "
    "theory and the mathematics behind it (deep learning, neural networks, "
    "transformers, optimization, and friends). "
    "Ask me about any of those and I'll open a grounded path with real sources!"
)
NOT_IN_CORPUS_MESSAGE = (
    "That's a real AI/ML topic, but it's not detailed in the books and papers "
    "indexed here yet. If you meant a related concept from our shelves, just "
    "name it — I'm happy to point you to the closest thing we do have."
)
IMPLEMENTATION_REFUSAL_MESSAGE = (
    "I'm a theory library at heart, so I skip code generation, install guides, "
    "and deployment walkthroughs — but the mathematics and architecture behind "
    "what you're building are exactly my shelf. Want the theoretical side of it?"
)


_SCOPE_SYSTEM = (
    "Is the query about artificial intelligence, machine learning, deep learning, or machine learning mathematics? "
    "Answer YES or NO."
)

# In-process LRU cache: normalized query → in_scope bool
_CACHE: OrderedDict[str, bool] = OrderedDict()
_CACHE_MAX = 256


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def parse_scope_answer(text: str) -> bool | None:
    """Parse model output. Returns True/False, or None if unparseable."""
    raw = (text or "").strip().lower()
    if not raw:
        return None
    # First non-empty line / first token
    token = re.split(r"[\s,.\n!?;:]+", raw, maxsplit=1)[0]
    if token in ("yes", "y", "true", "in", "in-scope", "inscope"):
        return True
    if token in ("no", "n", "false", "out", "out-of-scope", "outofscope"):
        return False
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    if "yes" in raw and "no" not in raw:
        return True
    if "no" in raw and "yes" not in raw:
        return False
    return None


def check_aiml_scope_via_llm(query: str) -> bool | None:
    """Ask Ollama yes/no. Returns None on transport/parse failure."""
    key = _normalize_query(query)
    if not key:
        return True
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    try:
        import ollama

        client = ollama.Client(host="http://localhost:11434")
        response = client.chat(
            model=st.DEFAULT_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SCOPE_SYSTEM},
                {"role": "user", "content": query},
            ],
            think=False,
            options={"temperature": 0.0, "num_predict": 8},
        )
        ans = (response.get("message") or {}).get("content", "")
        parsed = parse_scope_answer(ans)
        if parsed is None:
            return None
        _CACHE[key] = parsed
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
        return parsed
    except Exception as e:
        print(f"Ollama scope check failed: {e}")
        return None


def is_aiml_in_scope(
    query: str,
    *,
    chitchat: bool = False,
    offtopic_keyword: bool = False,
    has_domain_terms: bool = False,
    strong_anchor: bool = False,
    best_cos: float = 0.0,
    best_lex: float = 0.0,
    force_llm: bool = False,
    learning_intent: bool = False,
    graph_evidence: bool = False,
) -> tuple[bool, str]:
    """Hybrid AIML scope decision.

    Returns (in_scope, reason).

    Policy (graph-grounded first — keyword lists are weak tiebreakers only):
    - graph_evidence (alias/acronym/core/high-sim hit in the live graph) → in.
      The graph, not a hardcoded word list, decides what the library covers.
    - chitchat → in-scope (handled as general_chat by router)
    - offtopic keyword markers without domain terms → out (keyword assist,
      only reachable when the graph shows no related concept)
    - strong graph anchor → in (fast path, no LLM)
    - domain terms present → in (skip LLM)
    - learning intent without offtopic → fail-open on LLM miss
    - force_llm or ambiguous learning/library without domain → LLM
    - LLM fail + strong domain/embed → fail-open (in)
    - LLM fail + no domain signal → fail-closed (out) when force_llm
    """
    if chitchat:
        return True, "chitchat_skip"

    # Graph-grounded scope: the graph has something genuinely related — in.
    if graph_evidence:
        return True, "graph_evidence"

    # High-confidence direct concept matches are always in scope
    if best_lex >= 0.70 or best_cos >= 0.70:
        return True, "strong_graph_match"


    if offtopic_keyword and not has_domain_terms:
        return False, "offtopic_keyword"

    is_study_intent = bool(re.search(
        r"\b(learn|study|syllabus|curriculum|course|prereq|understand|wanna|want\s+to|explain)\b",
        query, re.I
    ))

    # Fast reject completely off-topic queries with no domain signals to avoid brittle LLM calls
    if (
        not chitchat
        and not strong_anchor
        and not has_domain_terms
        and (not learning_intent or not is_study_intent)
        and best_cos < 0.48
        and best_lex < 0.48
    ):
        return False, "no_domain_signal_low_similarity"

    if strong_anchor and not force_llm:
        return True, "strong_anchor_fast_path"

    # Domain keywords alone are enough — don't waste a brittle LLM call
    if has_domain_terms and not offtopic_keyword:
        return True, "domain_terms_present"

    # High confidence concept hit without calling LLM
    if (
        not force_llm
        and has_domain_terms
        and (best_cos >= 0.40 or best_lex >= 0.45)
    ):
        return True, "domain_signal_fast_path"

    if not force_llm and has_domain_terms and best_cos >= 0.28:
        return True, "domain_soft_fast_path"

    # Learning intents without clear offtopic markers: prefer in-scope
    # (router can still soft-chat if graph is empty)
    if learning_intent and not offtopic_keyword and not force_llm:
        if is_study_intent or has_domain_terms:
            return True, "learning_intent_open"

    needs_llm = force_llm or (
        not has_domain_terms and best_cos < 0.48 and best_lex < 0.45
    )
    if not needs_llm and has_domain_terms:
        return True, "domain_terms_present"

    llm = check_aiml_scope_via_llm(query)
    if llm is True:
        return True, "llm_yes"
    if llm is False:
        # Explicit "no" wins on forced LLM paths (e.g. library "books about stars").
        # Only soft-open when we did not force the classifier (chat learning queries).
        if learning_intent and not offtopic_keyword and not force_llm:
            return True, "llm_no_learning_open"
        return False, "llm_no"

    # Failure policy — fail-open for learning intents (transport/parse miss),
    # but forced-LLM paths (e.g. library "books about stars") stay fail-closed:
    # embedding cosine alone is not domain evidence there (stars ≈ Vector).
    if has_domain_terms or strong_anchor or (not force_llm and best_cos >= 0.48):
        return True, "llm_fail_open"
    if learning_intent and not offtopic_keyword and not force_llm:
        return True, "llm_fail_open_learning"
    if force_llm or not has_domain_terms:
        return False, "llm_fail_closed"
    return True, "llm_fail_default_open"


def clear_scope_cache() -> None:
    _CACHE.clear()

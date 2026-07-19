"""Concept ranking, anchors, and intent heuristics."""
from __future__ import annotations

import re
import torch

from thefuzz import fuzz

from archipelago.inference import state as st
from archipelago.inference.aliases import (
    extract_acronym, generate_aliases, _core_concept_bonus, _clean_query_words,
)
from archipelago.inference.embeddings import get_snowflake_embedding, _disable_embeddings

# Short tech tokens whose trailing plural "s" should fold for retrieval.
_PLURAL_TECH = frozenset({
    "rag", "llm", "cnn", "rnn", "gpt", "agent", "gnn", "mlp", "svm",
    "bert", "vae", "gan", "rl", "ml", "dl",
})

_GREETING_PREFIX = re.compile(
    r"^(?:hi|hello|hey|yo|sup|please|thanks|thank you)"
    r"(?:\s+(?:there|folks|team|all))?"
    r"[\s,!.]*",
    re.I,
)
_POLITE_FILLER = re.compile(
    r"\b(?:"
    r"well|so|anyway|anyways|\bby\s+the\s+way\b|\bbtw\b|"
    r"how\s+are\s+you|how'?s\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things|"
    r"nice\s+to\s+meet\s+you|good\s+to\s+see\s+you|"
    r"let'?s|let\s+us|"
    r"can\s+we|can\s+you|i\s+wanna|i\s+want\s+to|i'?d\s+like\s+to|"
    r"discuss\s+about|talk\s+about|tell\s+me\s+about|explain\s+to\s+me|"
    r"ok\s+so|alright\s+so|okay\s+so|"
    r"can you|could you|would you|please|suggest me|tell me|"
    r"how (?:can|do|to) (?:i|we|one) (?:learn|start)|"
    r"i (?:wanna|want to|would like to)|"
    r"various (?:sorts?|types?|kinds?|flavou?rs?) of|a bit about|something about)\b",
    re.I,
)
_IDENTITY_PATTERNS = (
    "who are you", "what are you", "tell me about yourself",
    "introduce yourself", "what is your name", "what's your name",
    "whats your name", "what can you do", "what do you do",
    "how do you work", "what are your capabilities",
    "are you an ai", "are you a bot", "are you human",
    "your name", "about you", "about yourself",
)
_ONBOARDING_PATTERNS = (
    r"\bstart\s+learning\b",
    r"\bgetting\s+started\b",
    r"\bhow\s+(?:do\s+i|to)\s+start\b",
    r"\blearn(?:ing)?\s+(?:ai|a\.?i\.?|ml|aiml|ai/?ml|machine\s+learning|deep\s+learning)\b",
    r"\b(?:ai|a\.?i\.?|ml|aiml|ai/?ml|machine\s+learning)\s+(?:syllabus|curriculum|roadmap|path)\b",
    r"\b(?:syllabus|curriculum|roadmap)\s+(?:for\s+)?(?:ai|ml|aiml|machine\s+learning)\b",
    # "i wanna learn ML" is onboarding; "i wanna learn about RAG types" is a
    # concept ask — require the learning target itself to be broad AI/ML.
    r"\bi\s+(?:wanna|want to)\s+(?:start\s+)?learn(?:ing)?\s+"
    r"(?:about\s+)?(?:ai|a\.?i\.?|ml|aiml|ai/?ml|machine\s+learning|deep\s+learning)\b",
    r"\bintro(?:duction)?\s+to\s+(?:ai|ml|aiml|machine\s+learning)\b",
    r"\bwhere\s+(?:do|should)\s+i\s+start\b",
    r"\bbeginner\s+(?:path|guide|roadmap)\b",
)


def normalize_user_query(query: str) -> str:
    """Strip greetings/politeness and fold short tech plurals (RAGS→RAG).

    Used for ranking/routing so noisy chatty questions still pin graph concepts.
    The original query is preserved for display and free-chat replies.
    """
    q = (query or "").strip()
    if not q:
        return q
    # Peel leading greetings a few times (e.g. "hi hey can you…")
    for _ in range(3):
        nq = _GREETING_PREFIX.sub("", q).strip(" ,")
        if nq == q:
            break
        q = nq
    q = _POLITE_FILLER.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,.?!")
    q = re.sub(r"\.{2,}", " ", q)
    q = re.sub(r"\?{2,}", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    tokens = []
    for w in q.split():
        bare = w.strip("?,.!;:()[]\"'")
        low = bare.lower()
        if low.endswith("s") and len(low) <= 6 and low[:-1] in _PLURAL_TECH:
            # Preserve original casing of the stem when possible
            stem = bare[:-1]
            tokens.append(stem)
        else:
            tokens.append(w.strip("?,.!;:()[]\"'"))
    return " ".join(t for t in tokens if t).strip() or (query or "").strip()


def _is_identity(query: str) -> bool:
    bare = (query or "").strip().lower().rstrip("!?.")
    return any(p in bare for p in _IDENTITY_PATTERNS)


def _is_onboarding(query: str) -> bool:
    """Broad AIML syllabus / 'start learning' intents (not a single concept pin).

    Requires an AIML domain cue so "I want to learn about stars" is not hijacked.
    """
    q = (query or "").strip().lower()
    if not q or _is_identity(q):
        return False
    if not any(re.search(p, q) for p in _ONBOARDING_PATTERNS):
        return False
    # Domain cue: AIML keywords or broader domain term list
    if re.search(
        r"\b(ai|a\.?i\.?|ml|aiml|ai/?ml|machine\s+learning|deep\s+learning|"
        r"data\s+science|neural|llm|transformer)\b",
        q,
    ):
        return True
    return _has_domain_terms(q)


def _is_chitchat(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or len(q) < 2:
        return True
    # Identity is its own route — not pure social chitchat
    if _is_identity(q):
        return False
    if _is_onboarding(q):
        return False
    chitchat = (
        "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening",
        "how are you", "what's up", "whats up", "bye", "ok", "okay", "cool", "lol",
        "yo", "sup", "good night", "gm", "gn",
    )
    bare = q.rstrip("!?.")
    if bare in chitchat:
        return True
    if len(q.split()) <= 4 and any(bare == c or bare.startswith(c + " ") for c in chitchat):
        return True
    return False


def _has_domain_terms(query: str) -> bool:
    q = f" {(query or '').lower()} "
    for term in st._DOMAIN_TERMS:
        # word-ish match: pad short tokens to avoid matching "ai" inside "said"
        if len(term) <= 2:
            if f" {term} " in q or q.strip().startswith(term + " ") or q.strip().endswith(" " + term):
                return True
            if re.search(rf"\b{re.escape(term)}\b", q):
                return True
        elif term in q:
            return True
    return False


def _is_learning_intent(query: str) -> bool:
    q = (query or "").lower()
    if _is_chitchat(q):
        return False
    return any(k in q for k in st._LEARNING_INTENT)


def _is_offtopic(query: str) -> bool:
    q = (query or "").lower()
    return any(m in q for m in st._OFFTOPIC_MARKERS)


def _is_learning_or_domain_query(query: str) -> bool:
    """True when the user wants AIML learning / tech discussion (not weather etc.)."""
    if _is_chitchat(query) or _is_offtopic(query):
        return False
    if _has_domain_terms(query):
        return True
    # Learning intent alone is not enough ("what is the weather?") — need domain terms
    # OR a non-trivial question that still looks technical via domain terms only.
    return False


def _expand_query_for_retrieval(query: str) -> str:
    """Light expansion for embedder only — helps 'AI agents' find nearby ML nodes."""
    q = (query or "").strip()
    if not q:
        return q
    ql = q.lower()
    extras = []
    if _has_domain_terms(ql) or _is_learning_intent(ql):
        extras.append("machine learning artificial intelligence deep learning")
    if re.search(r"\bagents?\b", ql):
        extras.append(
            "autonomous agents multi-agent systems tool use planning decision making "
            "language models reinforcement learning"
        )
    if "rag" in ql or "retrieval" in ql:
        extras.append("retrieval augmented generation dense retrieval documents")
    # Informal topic shortcuts (embedder expansion only — not the scope gate)
    if re.search(r"\bcomp(?:uter)?\s*vision\b", ql) or re.search(r"\bcv\b", ql):
        extras.append("computer vision image recognition convolutional neural networks")
    if re.search(r"\bnlp\b|natural language", ql):
        extras.append("natural language processing text transformers language models")
    if not extras:
        return q
    return q + " | " + " ".join(extras)


def _score_lexical_fit(query: str, label: str, summary: str = "") -> float:
    """0..1 lexical fit between query and a concept label/summary."""
    q = (query or "").lower()
    name = (label or "").lower()
    if not q or not name:
        return 0.0
    name_score = fuzz.token_set_ratio(q, name) / 100.0
    sum_score = (fuzz.partial_ratio(q, (summary or "")[:300]) / 100.0) if summary else 0.0
    # Prefer token overlap with the label
    q_words = _clean_query_words(q)
    n_words = _clean_query_words(name)
    overlap = 0.0
    if q_words and n_words:
        overlap = len(q_words & n_words) / max(1, len(n_words))
    return max(name_score, sum_score * 0.85) * 0.7 + overlap * 0.3


def rank_concepts(query, top_k=None):
    """Rank all graph concepts for ``query`` using the embedder when available.

    Returns a list of dicts: ``{id, label, cos, blended, summary, lexical}``
    sorted by blended score descending. Uses a lightly expanded query for
    embedding only so vague AIML asks (e.g. AI agents) still land near ML nodes.

    Session 2: alias/acronym boosts + core-concept preference so short queries
    like ``LoRA`` pin ``low_rank_adaptation`` instead of long family variants.
    """
    top_k = top_k or st.TOP_K_RELATED
    if not (query or "").strip() or not st.CONCEPTS_DATA:
        return []

    # Normalize noisy chat queries for ranking (greetings, plurals) while
    # keeping enough content for embeddings.
    norm = normalize_user_query(query)
    rank_query = norm if norm else query
    query_lower = rank_query.lower()
    q_words = _clean_query_words(query_lower)
    # Also keep original tokens so "RAGS" still matches after plural fold
    q_words |= _clean_query_words((query or "").lower())
    embed_query = _expand_query_for_retrieval(rank_query)
    ranked = []

    def _score_row(cid, concept, cos_score):
        label = concept.get("label") or concept.get("name") or cid
        name = label.lower()
        summary = concept.get("summary") or ""
        from archipelago.inference.aliases import generate_aliases
        aliases = generate_aliases(concept)
        fuzzy_boost = (fuzz.token_set_ratio(query_lower, name) / 100.0) * 0.15
        name_words = [w.strip("?,.!-") for w in name.split() if w.strip("?,.!-")]
        if name_words and all(w in q_words for w in name_words):
            fuzzy_boost += 0.20
        if summary:
            fuzzy_boost += (fuzz.partial_ratio(query_lower, summary[:240]) / 100.0) * 0.08
        # Alias / acronym hits (use normalized core + original query)
        alias_boost = 0.0
        q_core = re.sub(
            r"^(what is|what's|whats|explain|tell me about|how does|define)\s+",
            "",
            query_lower,
            flags=re.I,
        ).strip(" ?!.")
        alias_set = {a.lower() for a in aliases}
        if q_core in alias_set:
            alias_boost += 0.28
        # Token-level acronym hit: "rag" in query tokens vs aliases (RAGS→rag)
        for tok in q_words:
            if len(tok) >= 2 and tok in alias_set:
                alias_boost = max(alias_boost, 0.26)
        for a in aliases:
            al = a.lower()
            if len(al) >= 3 and (al in query_lower or query_lower in al or al in (query or "").lower()):
                alias_boost = max(alias_boost, 0.12)
            # Plural/singular: alias "rag" matches query token "rags"
            if len(al) >= 2 and any(
                t == al or t == al + "s" or (t.endswith("s") and t[:-1] == al)
                for t in q_words
            ):
                alias_boost = max(alias_boost, 0.26)
        # Degree (connectivity) mild boost — denser curriculum hubs win ties
        degree = float(concept.get("degree") or 0)
        degree_boost = 0.08 * (degree / (degree + 12.0))
        core_boost = _core_concept_bonus(rank_query, cid, label, aliases)
        lexical = max(
            _score_lexical_fit(rank_query, label, summary),
            _score_lexical_fit(query, label, summary),
        )
        # Alias exact match also lifts lexical for strong-anchor threshold
        if alias_boost >= 0.26:
            lexical = max(lexical, 0.85)
        blended = cos_score + fuzzy_boost + lexical * 0.12 + alias_boost + degree_boost + core_boost
        return {
            "id": cid,
            "label": label,
            "summary": summary,
            "cos": cos_score,
            "lexical": lexical,
            "alias_boost": alias_boost,
            "core_boost": core_boost,
            "blended": blended,
        }

    if st.use_embeddings and st.CONCEPT_EMBEDDINGS_TENSOR is not None:
        try:
            query_emb = get_snowflake_embedding(embed_query)
            if query_emb is not None:
                device = st.CONCEPT_EMBEDDINGS_TENSOR.device
                query_emb_dev = query_emb.to(device)
                with torch.no_grad():
                    similarities = torch.mv(st.CONCEPT_EMBEDDINGS_TENSOR, query_emb_dev)
                for idx, cid in enumerate(st.CONCEPT_IDS):
                    cos_score = float(similarities[idx].item())
                    concept = st.CONCEPTS_DATA.get(cid, {})
                    ranked.append(_score_row(cid, concept, cos_score))
                ranked.sort(key=lambda r: r["blended"], reverse=True)
                return ranked[: max(1, int(top_k))]
        except Exception as e:
            _disable_embeddings(str(e))
            print(f"rank_concepts embedder failed ({e}); lexical ranking.")

    # Lexical ranking fallback (embeddings off)
    for cid, concept in st.CONCEPTS_DATA.items():
        name = concept.get("label") or concept.get("name") or cid
        summary = concept.get("summary") or ""
        lexical = _score_lexical_fit(query, name, summary)
        row = _score_row(cid, concept, lexical)
        # When offline, cos tracks lexical base
        row["cos"] = lexical
        ranked.append(row)
    ranked.sort(key=lambda r: r["blended"], reverse=True)
    return ranked[: max(1, int(top_k))]


def find_anchor_concept(query):
    """Return ``(concept_id, confidence)`` for a defensible strong anchor.

    Strong hits need high cosine *and* non-trivial lexical fit so vague queries
    do not pin onto an unrelated high-dim embedding neighbor.
    """
    ranked = rank_concepts(query, top_k=5)
    if not ranked:
        return None, 0.0

    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        
        candidate_lines = []
        for cand in ranked:
            lbl = cand.get("label") or cand.get("name") or cand["id"]
            summary = cand.get("summary") or ""
            candidate_lines.append(f"- ID: '{cand['id']}' | Label: '{lbl}' | Summary: '{summary}'")
        candidates_str = "\n".join(candidate_lines)

        system_prompt = (
            "You are a precise concept-matching system.\n"
            "Given a user query and a list of candidate concepts, identify which concept ID "
            "best matches the user query. You must only choose a concept ID from the list "
            "if it is a clear, direct, and correct match.\n"
            "If there is a match, reply ONLY with the matched concept's ID (e.g. 'low_rank_adaptation').\n"
            "If none of the concepts in the list matches the query, reply ONLY with 'None'.\n"
            "Do not explain, do not add introductory text, just output the raw ID or 'None'."
        )
        user_content = f"User Query: {query}\n\nCandidate Concepts:\n{candidates_str}\n\nAnswer:"

        response = client.chat(
            model=st.DEFAULT_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            think=False,
            options={"temperature": 0.0, "num_predict": 30},
        )
        llm_response = response.get("message", {}).get("content", "").strip()
        cleaned_response = llm_response.strip().strip("'\"`").strip()
        
        if cleaned_response.lower() == "none":
            return None, 0.0

        for cand in ranked:
            if cleaned_response == cand["id"] or cleaned_response.lower() == cand["id"].lower():
                cos = float(cand.get("cos") or 0.0)
                lexical = float(cand.get("lexical") or 0.0)
                alias_boost = float(cand.get("alias_boost") or 0.0)
                core_boost = float(cand.get("core_boost") or 0.0)
                # LLM confirmation + surface evidence → same confidence scale
                # as the heuristic path, so downstream gates behave identically.
                if alias_boost >= 0.25 or core_boost >= 0.35:
                    return cand["id"], max(cos, lexical, 0.9)
                return cand["id"], max(cos, lexical)
    except Exception as e:
        print(f"Ollama call failed or unavailable during find_anchor_concept: {e}")

    for cand in ranked:
        cos = float(cand.get("cos") or 0.0)
        blended = float(cand.get("blended") or 0.0)
        lexical = float(cand.get("lexical") or 0.0)
        alias_boost = float(cand.get("alias_boost") or 0.0)
        core_boost = float(cand.get("core_boost") or 0.0)
        # Session 2: alias/acronym or core-concept hit → strong anchor
        if alias_boost >= 0.25 or core_boost >= 0.35:
            print(
                f"Anchor Match (alias/core): {cand['id']} "
                f"(alias={alias_boost:.2f}, core={core_boost:.2f}, blended={blended:.4f})"
            )
            return cand["id"], max(cos, lexical, 0.9)
        # Pure high cosine + some label/summary fit
        if cos >= st.SEMANTIC_ANCHOR_THRESHOLD and lexical >= 0.22:
            print(
                f"Anchor Match (strong): {cand['id']} "
                f"(cos: {cos:.4f}, lex: {lexical:.4f}, blended: {blended:.4f})"
            )
            return cand["id"], cos
        # Very high cosine alone
        if cos >= st.SEMANTIC_ANCHOR_THRESHOLD + 0.12:
            print(f"Anchor Match (very high cos): {cand['id']} (cos: {cos:.4f})")
            return cand["id"], cos
        # Strong lexical when embeddings offline / weak
        if lexical >= 0.72 and (cos >= st.DOMAIN_SOFT_THRESHOLD or not st.use_embeddings):
            print(f"Anchor Match (lexical-strong): {cand['id']} (lex: {lexical:.4f})")
            return cand["id"], max(cos, lexical)

    best = ranked[0]
    print(
        f"No strong anchor; top={best['id']} cos={float(best['cos']):.4f} "
        f"lex={float(best.get('lexical') or 0):.4f} "
        f"(strong>={st.SEMANTIC_ANCHOR_THRESHOLD}, soft>={st.DOMAIN_SOFT_THRESHOLD})"
    )
    return None, float(best.get("cos") or 0.0)



def _has_surface_concept_hit(ranked: list, query: str) -> bool:
    """True when top ranked concepts share a label/alias/acronym with the query."""
    if not ranked:
        return False
    ql = (query or "").lower()
    norm = normalize_user_query(query).lower()
    q_tokens = _clean_query_words(ql) | _clean_query_words(norm)
    for r in ranked[:5]:
        if float(r.get("alias_boost") or 0) >= 0.12:
            return True
        if float(r.get("core_boost") or 0) >= 0.30:
            return True
        if float(r.get("lexical") or 0) >= 0.45:
            return True
        label = (r.get("label") or r.get("name") or "").lower()
        cid = (r.get("id") or "").lower().replace("_", " ")
        for tok in q_tokens:
            if len(tok) >= 3 and (tok in label or tok in cid):
                return True
        concept = st.CONCEPTS_DATA.get(r.get("id") or "", {})
        aliases = concept.get("aliases") or generate_aliases(concept)
        for a in aliases:
            al = (a or "").lower()
            if len(al) < 2:
                continue
            if al in ql or al in norm:
                return True
            if any(t == al or t == al + "s" or (t.endswith("s") and t[:-1] == al) for t in q_tokens):
                return True
    return False



def _has_strong_graph_evidence(ranked: list) -> bool:
    """True when the graph itself holds something clearly related to the query.

    Stricter than _has_surface_concept_hit: a shared generic token (e.g. the
    word "matrix" in a movie question) is NOT enough — we require an alias or
    acronym hit, a core-concept mapping, or high lexical/embedding similarity.
    This is the graph-grounded scope signal that replaces keyword-list vetoes.
    """
    for r in (ranked or [])[:5]:
        if float(r.get("alias_boost") or 0.0) >= 0.26:
            return True
        if float(r.get("core_boost") or 0.0) >= 0.35:
            return True
        if float(r.get("lexical") or 0.0) >= 0.60:
            return True
        if st.use_embeddings and float(r.get("cos") or 0.0) >= 0.62:
            return True
    return False


def _select_soft_anchor(ranked, query):
    """Pick the best soft anchor from ranked neighbors (embed + label + id fit).

    Session 2: prefer core concept IDs / short labels when the query is an
    acronym or short alias (LoRA → low_rank_adaptation, not family variants).
    """
    if not ranked:
        return None
    q = (query or "").lower()
    best, best_s = None, -1.0
    for r in ranked:
        cos = float(r.get("cos") or 0.0)
        lexical = float(r.get("lexical") or 0.0)
        id_lex = fuzz.token_set_ratio(q, (r.get("id") or "").replace("_", " ")) / 100.0
        label = (r.get("label") or "").lower()
        cid = r.get("id") or ""
        token_hit = 0.0
        for tw in _clean_query_words(q):
            if len(tw) >= 3 and tw in label.replace("-", " "):
                token_hit = max(token_hit, 0.22)
        core = float(r.get("core_boost") or 0.0) + float(r.get("alias_boost") or 0.0)
        if not core:
            concept = st.CONCEPTS_DATA.get(cid, {})
            core = _core_concept_bonus(query, cid, r.get("label") or "", generate_aliases(concept))
        # Prefer shorter labels when query is a short token (acronym)
        length_adj = 0.0
        q_core = re.sub(
            r"^(what is|what's|whats|explain|tell me about)\s+",
            "",
            q,
            flags=re.I,
        ).strip(" ?!.")
        if len(q_core) <= 6 and q_core.isalpha():
            words = len(label.split())
            if words <= 3:
                length_adj += 0.15
            elif words >= 6:
                length_adj -= 0.20
        score = cos * 0.40 + lexical * 0.28 + id_lex * 0.12 + token_hit + core * 0.55 + length_adj
        # Tie-break: already-ranked blended if present
        score += 0.05 * float(r.get("blended") or 0.0)
        if score > best_s:
            best_s = score
            best = r
    return best



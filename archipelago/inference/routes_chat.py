"""Chat API streaming endpoint and concept bootstrap."""
from __future__ import annotations

import json
import os
import threading

from flask import Response, request, jsonify

from ingestion_worker import graph_lock
from archipelago.auth import require_student_or_open
from archipelago.inference import state as st
from archipelago.inference.routing import resolve_query_routing
from archipelago.inference.neighborhood import get_graph_neighborhood
from archipelago.inference.curriculum import (
    find_curriculum_chains, format_curriculum_paths_section,
)
from archipelago.inference.citations import (
    build_concept_citation_map, build_citation_payloads, _resolve_printed_page,
)
from archipelago.inference.synthesis import (
    build_graph_notes, format_natural_fallback, synthesize_with_ollama_streaming,
    general_chat_reply, is_ollama_available, OLLAMA_UNAVAILABLE_MSG,
    render_library_books, render_library_chapters, render_library_chapter_lookup,
    identity_reply, onboarding_reply, not_indexed_reply, enforce_sterile_prose,
)
from archipelago.inference.library_queries import (
    get_books_for_topic, clean_topic_query, get_chapters_of_book, get_chapters_containing_concept,
)
from archipelago.inference.scope_gate import (
    OUT_OF_SCOPE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    IMPLEMENTATION_REFUSAL_MESSAGE,
)
from archipelago.inference.aliases import _node_name, generate_aliases
from archipelago.inference.embeddings import load_embedding_model, load_aura_model

@st.app.route("/api/chat", methods=["POST"])
@require_student_or_open
def api_chat():
    """Student-facing chat. No librarian upload/delete privileges."""
    from flask import Response
    req_data = request.get_json() or {}
    query = req_data.get("query", "").strip()
    mode = req_data.get("mode", "rag_synthesis")
    history = req_data.get("history", [])

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Default product path: embedder ranking + graph traversal + natural reply.
    # conversational_agent uses the same smart router (domain → graph, chitchat → free chat).
    if mode in ("rag_synthesis", "conversational_agent"):
        routing = resolve_query_routing(query, history=history)
        route = routing["route"]
        # Explicit synthesis flag still honored; default ON so answers are natural.
        wants_synthesis = req_data.get("synthesis", True) not in (False, "off", "none", 0, "0")

        # ── Out of Scope / defensive refusals (3 generic messages only) ─
        if route == "out_of_scope":
            reason = routing.get("reason", "out_of_scope_topic") or ""
            rl = reason.lower()
            intent_meta = (routing.get("slots") or {})
            if "implementation" in rl:
                out_msg = IMPLEMENTATION_REFUSAL_MESSAGE
                detail = (
                    f"Intent gate blocked implementation "
                    f"(intent={intent_meta.get('intent')}, "
                    f"method={intent_meta.get('intent_method')})."
                )
            elif "not_in_corpus" in rl or "entity" in rl:
                out_msg = NOT_IN_CORPUS_MESSAGE
                detail = (
                    f"Intent gate: entity/trivia not grounded in corpus "
                    f"(intent={intent_meta.get('intent')})."
                )
            elif "meta" in rl:
                # Same sterile boundary as OOS — do not leak constraints
                out_msg = OUT_OF_SCOPE_MESSAGE
                detail = "Intent gate blocked meta / system-prompt extraction."
            else:
                out_msg = OUT_OF_SCOPE_MESSAGE
                detail = (
                    f"Out of AIML library scope "
                    f"(intent={intent_meta.get('intent')}, reason={reason})."
                )

            closest = intent_meta.get("closest_concepts") or []
            # Soft "maybe you meant" bridge on every reject flavor that has
            # plausible graph neighbors — not just corpus misses.
            closest = [c for c in closest if c][:3]
            if closest:
                bridge = ", ".join(f"**{c}**" for c in closest)
                out_msg = (
                    f"{out_msg}\n\nIf you were after something nearby, the closest "
                    f"concepts on our shelves are: {bridge}."
                )

            def generate_out_of_scope():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 0.0, "reason": reason},
                    "logs": [{
                        "step": "Pass 1: Intent Gate & Scope",
                        "status": "Out of Scope",
                        "details": detail,
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                yield out_msg
            return Response(generate_out_of_scope(), mimetype="text/plain")

        # ── Identity ──────────────────────────────────────────────────
        if route == "identity":
            def generate_identity():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 1.0, "reason": "identity"},
                    "logs": [{
                        "step": "Pass 1: Intent",
                        "status": "Identity",
                        "details": "Assistant identity / capabilities question.",
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                yield identity_reply(query, history)
            return Response(generate_identity(), mimetype="text/plain")

        # ── Onboarding / start learning AIML ──────────────────────────
        if route == "onboarding":
            def generate_onboarding():
                related = routing.get("related") or []
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": related[:5],
                    "routing": {"route": route, "score": 1.0, "reason": "onboarding_syllabus"},
                    "logs": [{
                        "step": "Pass 1: Intent",
                        "status": "Onboarding",
                        "details": "Broad AIML syllabus / start-learning entry path.",
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                yield onboarding_reply(query, related)
            return Response(generate_onboarding(), mimetype="text/plain")

        # ── Small Talk ──────────────────────────────────────────────────
        if route == "small_talk":
            def generate_small_talk():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 1.0, "reason": "conversational_greeting"},
                    "logs": [{
                        "step": "Pass 1: Intent",
                        "status": "Small talk",
                        "details": "Conversational pleasantry detected.",
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                yield general_chat_reply(query, history)
            return Response(generate_small_talk(), mimetype="text/plain")

        # ── Library Query: Books Recommendation ───────────────────────
        if route == "library_books":
            topic = clean_topic_query(query)
            limit = int((routing.get("slots") or {}).get("limit") or 5)
            books = get_books_for_topic(query, limit=limit)
            notes = render_library_books(topic, books)
            def generate_books():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 1.0, "reason": "library_books"},
                    "logs": [{
                        "step": "Library Retrieval",
                        "status": "Success",
                        "details": f"Found {len(books)} books for topic '{topic}'.",
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                if wants_synthesis:
                    try:
                        text = synthesize_with_ollama_streaming(notes, evidence_ids=set(), user_query=query, history=history)
                        yield text if text else notes
                    except RuntimeError:
                        yield OLLAMA_UNAVAILABLE_MSG
                    except Exception:
                        yield OLLAMA_UNAVAILABLE_MSG
                else:
                    yield notes
            return Response(generate_books(), mimetype="text/plain")

        # ── Library Query: Chapters of Book ───────────────────────────
        if route == "library_chapters":
            res = get_chapters_of_book(query)
            if res:
                book_title, chapters = res
                notes = render_library_chapters(book_title, chapters)
                details = f"Retrieved {len(chapters)} chapters for '{book_title}'."
            else:
                book_title = query
                notes = f"Could not find any matching book/paper for '{query}' in the database."
                details = "No book match found."
            def generate_chapters():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 1.0, "reason": "library_chapters"},
                    "logs": [{
                        "step": "Library Retrieval",
                        "status": "Success" if res else "Not Found",
                        "details": details,
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                if wants_synthesis and res:
                    try:
                        text = synthesize_with_ollama_streaming(notes, evidence_ids=set(), user_query=query, history=history)
                        yield text if text else notes
                    except RuntimeError:
                        yield OLLAMA_UNAVAILABLE_MSG
                    except Exception:
                        yield OLLAMA_UNAVAILABLE_MSG
                else:
                    yield notes
            return Response(generate_chapters(), mimetype="text/plain")

        # ── Library Query: Chapter Lookup for Concept ─────────────────
        if route == "library_chapter_lookup":
            res = get_chapters_containing_concept(query)
            if res:
                book_title, concept_name, chapters = res
                notes = render_library_chapter_lookup(book_title, concept_name, chapters)
                details = f"Found {len(chapters)} chapters in '{book_title}' discussing '{concept_name}'."
            else:
                notes = f"Could not find matching book or concept for query: '{query}'."
                details = "No match found."
            def generate_lookup():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": [],
                    "routing": {"route": route, "score": 1.0, "reason": "library_chapter_lookup"},
                    "logs": [{
                        "step": "Library Retrieval",
                        "status": "Success" if res else "Not Found",
                        "details": details,
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                if wants_synthesis and res:
                    try:
                        text = synthesize_with_ollama_streaming(notes, evidence_ids=set(), user_query=query, history=history)
                        yield text if text else notes
                    except RuntimeError:
                        yield OLLAMA_UNAVAILABLE_MSG
                    except Exception:
                        yield OLLAMA_UNAVAILABLE_MSG
                else:
                    yield notes
            return Response(generate_lookup(), mimetype="text/plain")

        # ── Low Similarity Reject (honest not-indexed reply) ────────────
        if route == "low_similarity_reject":
            sterile = bool((routing.get("slots") or {}).get("sterile"))
            def generate_reject():
                closest = (routing.get("slots") or {}).get("closest_concepts") or []
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": routing.get("related") or [],
                    "routing": {"route": route, "score": routing.get("score"), "reason": routing.get("reason")},
                    "logs": [{
                        "step": "Pass 1: Intent & Embedder Gate",
                        "status": "Not indexed",
                        "details": (
                            f"Highest similarity score ({float(routing.get('score') or 0):.3f}) "
                            f"is below the rejection threshold ({st.REJECT_SIMILARITY_THRESHOLD}) "
                            f"with no strong lexical/alias surface hit."
                        ),
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                reply = not_indexed_reply(query, closest, natural=wants_synthesis and not sterile)
                if sterile:
                    reply = enforce_sterile_prose(reply, fallback=reply)
                yield reply
            return Response(generate_reject(), mimetype="text/plain")

        # ── General chat (low similarity / chitchat) ──────────────────────
        if route == "general_chat":
            def generate_general():
                payload = {
                    "anchor_concept": None,
                    "prerequisites": [],
                    "unlocks": [],
                    "citations": [],
                    "related_concepts": routing.get("related") or [],
                    "routing": {"route": route, "score": routing.get("score"), "reason": routing.get("reason")},
                    "logs": [{
                        "step": "Pass 1: Intent & Embedder Gate",
                        "status": "General chat",
                        "details": (
                            f"Similarity too low for graph grounding "
                            f"(score={float(routing.get('score') or 0):.3f} < soft "
                            f"{st.DOMAIN_SOFT_THRESHOLD}). Free conversational reply."
                        ),
                    }],
                }
                yield json.dumps(payload) + "\n[STREAM_START]\n"
                yield general_chat_reply(query, history)
            return Response(generate_general(), mimetype="text/plain")

        # ── Graph path (strong or soft domain) ────────────────────────────
        with graph_lock.read_lock():
            anchor_id = routing.get("anchor_id")
            related = routing.get("related") or []
            if not anchor_id or anchor_id not in st.CONCEPTS_DATA:
                # Soft path without a usable id → still try top related list
                if related and related[0]["id"] in st.CONCEPTS_DATA:
                    anchor_id = related[0]["id"]
                else:
                    def generate_orphan():
                        payload = {
                            "logs": [{
                                "step": "Pass 1: Retrieval",
                                "status": "Empty graph",
                                "details": "No concepts loaded in st.CONCEPTS_DATA.",
                            }],
                            "citations": [],
                        }
                        yield json.dumps(payload) + "\n[STREAM_START]\n"
                        yield general_chat_reply(query, history)
                    return Response(generate_orphan(), mimetype="text/plain")

            target_concept = st.CONCEPTS_DATA[anchor_id]
            prereqs, unlocks, _legacy_cites = get_graph_neighborhood(anchor_id, k=2)
            # Soft multi-anchor: pull light neighborhoods for top related peers
            related_nodes = []
            for r in related[:st.TOP_K_RELATED]:
                rid = r.get("id")
                if not rid or rid == anchor_id or rid not in st.CONCEPTS_DATA:
                    continue
                node = st.CONCEPTS_DATA[rid]
                related_nodes.append({
                    "id": rid,
                    "name": node.get("label") or node.get("name") or rid,
                    "summary": node.get("summary") or r.get("summary") or "",
                    "cos": r.get("cos"),
                })

            citation_map = build_concept_citation_map(target_concept, prereqs, unlocks)
            citation_payloads = build_citation_payloads(target_concept, prereqs, unlocks, citation_map)
            evidence_ids = {payload["evidence_id"] for payload in citation_payloads if payload["evidence_id"]}
            # Session 2: multi-hop curriculum chains (≤3 hops) with book/page links
            curriculum_paths = find_curriculum_chains(
                anchor_id, max_hops=3, citation_map=citation_map, max_paths=4
            )
            notes = build_graph_notes(
                query, target_concept, prereqs, unlocks, related, citation_map,
                bare_markers=True,
            )
            if curriculum_paths:
                notes = notes + "\n" + format_curriculum_paths_section(curriculum_paths)
            is_partial = bool((routing.get("slots") or {}).get("partial")) or (
                routing.get("reason") == "partial_coverage_low_cos"
            )
            natural_fallback = format_natural_fallback(
                query, target_concept, prereqs, unlocks, related, citation_map,
                curriculum_paths=curriculum_paths,
                partial=is_partial,
            )
            match_score = routing.get("score")
            step_logs = [
                {
                    "step": "Pass 1: Embedder Ranking",
                    "status": (
                        "Partial coverage"
                        if is_partial
                        else ("Success" if route == "graph_strong" else "Soft domain match")
                    ),
                    "details": (
                        f"Route={route}; anchor=**{_node_name(target_concept)}** "
                        f"(score={match_score if isinstance(match_score, (int, float)) else match_score}); "
                        f"reason={routing.get('reason')}; top related="
                        f"{', '.join(r.get('label', '') for r in related[:3])}"
                    ),
                },
                {
                    "step": "Pass 2: Graph Traversal & Citations",
                    "status": "Success",
                    "details": (
                        f"Traversed {len(prereqs)} prerequisites and {len(unlocks)} unlocks; "
                        f"{len(citation_payloads)} evidence records; "
                        f"{len(related_nodes)} embedder neighbors; "
                        f"{len(curriculum_paths)} multi-hop curriculum path(s)."
                    ),
                },
                {
                    "step": "Pass 3: Natural Synthesis",
                    "status": "Requested" if wants_synthesis else "Template",
                    "details": (
                        f"{st.DEFAULT_OLLAMA_MODEL} natural wording over graph notes "
                        "(falls back to structured natural summary if offline)"
                        if wants_synthesis
                        else "Structured natural summary without generator."
                    ),
                },
            ]

            def generate_graph():
                init_payload = {
                    "anchor_concept": target_concept,
                    "prerequisites": prereqs,
                    "unlocks": unlocks,
                    "related_concepts": related_nodes,
                    "curriculum_paths": curriculum_paths,
                    "citations": citation_payloads,
                    "routing": {
                        "route": route,
                        "score": routing.get("score"),
                        "reason": routing.get("reason"),
                    },
                    "citation_by_concept": {
                        concept_id: [
                            {
                                "evidence_id": evidence.get("evidence_id"),
                                "doc_id": evidence.get("doc_id"),
                                "page_number": evidence.get("page_number"),
                                "printed_page": _resolve_printed_page(evidence),
                                "section_title": evidence.get("section_title"),
                            }
                            for evidence in concept_evidence
                        ]
                        for concept_id, concept_evidence in citation_map.items()
                    },
                    "logs": step_logs,
                }
                sterile = bool((routing.get("slots") or {}).get("sterile"))
                yield json.dumps(init_payload) + "\n[STREAM_START]\n"
                if wants_synthesis:
                    try:
                        text = synthesize_with_ollama_streaming(
                            notes,
                            evidence_ids=evidence_ids,
                            user_query=query,
                            citation_payloads=citation_payloads,
                            sterile=sterile,
                            fallback_text=natural_fallback,
                            history=history,
                        )
                        if text:
                            yield text
                        else:
                            yield enforce_sterile_prose(natural_fallback) if sterile else natural_fallback
                        return
                    except RuntimeError:
                        yield OLLAMA_UNAVAILABLE_MSG
                        return
                    except Exception:
                        yield OLLAMA_UNAVAILABLE_MSG
                        return
                yield enforce_sterile_prose(natural_fallback) if sterile else natural_fallback

            return Response(generate_graph(), mimetype="text/plain")

    else:
        return jsonify({"error": f"Invalid mode: {mode}"}), 400


def init_concepts_data():
    try:
        with open(st.DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("visualization", {}).get("nodes", []) or data.get("nodes", [])
        st.CONCEPTS_DATA = {n["id"]: n for n in nodes}
        # Session 2: precompute aliases for acronym/alias-aware ranking
        for cid, concept in st.CONCEPTS_DATA.items():
            if "id" not in concept:
                concept["id"] = cid
            concept["aliases"] = generate_aliases(concept)
            print(f"Synchronously loaded {len(st.CONCEPTS_DATA)} concepts at startup (aliases ready).")
    except Exception as e:
        print(f"Error loading concepts at startup: {e}")

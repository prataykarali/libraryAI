"""Compatibility shim — full implementation lives in archipelago.inference.*

Existing tests and scripts: `import inference_server` still works.
Prefer: `from archipelago.inference...` or `python -m archipelago.apps.inference_app`.
"""
from __future__ import annotations

# Side-effect: register Flask routes
from archipelago.inference import bootstrap as _bootstrap  # noqa: F401
from archipelago.inference.state import (  # noqa: F401
    app, BASE_DIR, PDF_DIR, DATA_FILE, DB_PATH, db,
    CONCEPTS_DATA, CONCEPT_IDS, CONCEPT_EMBEDDINGS, CONCEPT_EMBEDDINGS_TENSOR,
    use_embeddings, embed_model, embed_tokenizer,
    aura_model, aura_tokenizer, aura_loaded,
    SEMANTIC_ANCHOR_THRESHOLD, DOMAIN_SOFT_THRESHOLD, DEFAULT_OLLAMA_MODEL,
    TOP_K_RELATED, PDF_BASE_URL, MAX_PREREQS_SHOWN, MAX_UNLOCKS_SHOWN,
    CITATION_ID_PATTERN, EVIDENCE_PER_CONCEPT,
)
from archipelago.inference.aliases import (  # noqa: F401
    _node_name, extract_acronym, generate_aliases, pdf_page_url, markdown_pdf_link,
    _core_concept_bonus, _clean_query_words,
)
from archipelago.inference.embeddings import (  # noqa: F401
    load_embedding_model, load_aura_model, build_concept_embeddings,
    get_snowflake_embedding, _disable_embeddings, _embed_device_preference,
)
from archipelago.inference.ranking import (  # noqa: F401
    rank_concepts, find_anchor_concept, _select_soft_anchor,
    _is_chitchat, _has_domain_terms, _is_learning_intent, _is_offtopic,
    _is_learning_or_domain_query, _expand_query_for_retrieval, _score_lexical_fit,
)
from archipelago.inference.routing import resolve_query_routing  # noqa: F401
from archipelago.inference.neighborhood import (  # noqa: F401
    get_concept_citations, get_graph_neighborhood,
)
from archipelago.inference.curriculum import (  # noqa: F401
    find_curriculum_chains, format_curriculum_paths_section, _hop_provenance,
)
from archipelago.inference.citations import (  # noqa: F401
    compile_narrative_recipe, build_concept_citation_map, validate_citations,
    citation_payload, build_citation_payloads, _cite_with_link,
    _resolve_printed_page, _page_display, _citation_label,
    _normalize_legacy_citation, _evidence_for_concept, _evidence_for_prerequisite,
)
from archipelago.inference.synthesis import (  # noqa: F401
    render_indexed_learning_path, synthesize_with_ollama, general_chat_reply,
    build_graph_notes, format_natural_fallback, generate_aura_synthesis, run_ollama_agent,
)
from archipelago.inference.routes_misc import (  # noqa: F401
    add_cors_headers, serve_pdf, readiness, ingestion_capabilities,
    ingest_upload, ingest_status, ingest_cancel, ingest_list, server_root,
)
from archipelago.inference.graph_access import _default_graph_db  # noqa: F401
from archipelago.inference.routes_chat import api_chat, init_concepts_data  # noqa: F401

# Mirror mutable state for tests that assign inference_server.CONCEPTS_DATA = ...
# Use a module-level property pattern via __getattr__/__setattr__ on the module.

import sys as _sys
import archipelago.inference.state as _st

_this = _sys.modules[__name__]

def __getattr__(name):
    if hasattr(_st, name):
        return getattr(_st, name)
    raise AttributeError(name)

# For assignments like inference_server.CONCEPTS_DATA = x
# Python 3.12: module-level __getattr__ works for reads; writes need explicit sync.
# Provide helpers tests already use by rebinding after import — tests set attributes
# on the module object. We override by wrapping in a custom module type is heavy;
# instead, patch common attributes as properties on a simple namespace.

class _Proxy:
    """Keep inference_server.CONCEPTS_DATA / .db / .use_embeddings in sync with state."""
    def __init__(self, name):
        self.name = name
    def __get__(self, obj, objtype=None):
        return getattr(_st, self.name)
    def __set__(self, obj, value):
        setattr(_st, self.name, value)

# Can't put descriptors on module easily without custom ModuleType.
# Tests do: inference_server.CONCEPTS_DATA = dict...
# After assignment, local module dict has CONCEPTS_DATA key shadowing __getattr__.
# Fix: re-export and document that tests should also update state — OR
# use a custom import hook. Simpler fix used below in conftest note.
# We'll sync on common pattern by making CONCEPTS_DATA a list/dict that is the SAME object.

# Already same object reference from import. Assignments REPLACE the binding on
# inference_server module only. Fix: provide setattr hook via install:

def _install_sync():
    import types
    mod = _sys.modules[__name__]
    class InferenceModule(types.ModuleType):
        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)
            if hasattr(_st, key):
                setattr(_st, key, value)
        def __getattribute__(self, key):
            if key.startswith('_'):
                return object.__getattribute__(self, key)
            try:
                return object.__getattribute__(self, key)
            except AttributeError:
                return getattr(_st, key)
    # Replace module class
    mod.__class__ = InferenceModule

_install_sync()

if __name__ == "__main__":
    from archipelago.apps.inference_app import main
    main()

"""Archipelago inference (feature-split modules).

Import from here or from inference_server (compat shim).
"""
from archipelago.inference.state import app, BASE_DIR, DATA_FILE, db, CONCEPTS_DATA, use_embeddings
from archipelago.inference.aliases import (
    extract_acronym, generate_aliases, pdf_page_url, markdown_pdf_link, _node_name,
)
from archipelago.inference.ranking import rank_concepts, find_anchor_concept
from archipelago.inference.routing import resolve_query_routing
from archipelago.inference.neighborhood import get_graph_neighborhood, get_concept_citations
from archipelago.inference.curriculum import find_curriculum_chains, format_curriculum_paths_section
from archipelago.inference.citations import (
    build_concept_citation_map, validate_citations, citation_payload,
    build_citation_payloads, compile_narrative_recipe,
)
from archipelago.inference.synthesis import (
    render_indexed_learning_path, format_natural_fallback, synthesize_with_ollama,
)
from archipelago.inference.routes_chat import api_chat, init_concepts_data
from archipelago.inference.embeddings import load_embedding_model, get_snowflake_embedding

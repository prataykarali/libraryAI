"""Archipelago OKF pipeline package.

Public API re-exports so `import okf` / `from okf import ...` exposes the same
names the monolithic okf_pipeline.py used to. Module layout:

    config        constants, prompts, schema vocabularies
    util          concept IDs + provenance record helpers
    canonicalize  name canonicalization and alias merging
    cleanup       junk filters, grounding, dedup, cycle breaking
    extraction    SLM extraction + local-model state (owns LOCAL_MODEL et al.)
    relations     second-pass relation extraction
    graph_db      KùzuDB ingestion and raw graph export
    exports       visualization / GraphRAG index / audit payloads
    evaluate      proxy accuracy metrics
    pipeline      orchestration (run_pipeline, add_document, finalize_and_build)
"""

from okf.config import (
    BASE_DIR,
    EXTRACTION_PROMPT_V15,
    MAX_CHARS_TO_SLM,
    MAX_PAGES_PER_DOC,
    MAX_RETRIES,
    MODEL_NAME,
    RELATION_PROMPT,
    VALID_DIFFICULTIES,
    VALID_RELATIONS,
    VALID_TYPES,
    infer_source_category,
)
from okf.util import (
    _dedupe_dicts,
    _record_sources,
    _source_record,
    create_concept_id,
)
from okf.canonicalize import (
    ALIAS_MAP,
    apply_canonicalization,
    build_canonical_map,
    canonicalize_name,
    is_same_concept_reference,
)
from okf.cleanup import (
    apply_grounding_filter,
    break_global_cycles,
    cleanup_and_canonicalize,
    dedupe_identical_records,
    is_valid_concept_name,
    merge_duplicate_results,
    prune_invalid_references,
    prune_unresolved_references,
)
from okf.extraction import (
    extract_chunks_with_model,
    extract_okf_v15,
    is_local_mode,
    is_model_loaded,
    load_local_model,
    normalize_okf_item,
)
from okf.relations import (
    extract_relations_for_record,
    relation_pass,
    run_relations_only,
)
from okf.graph_db import (
    _kuzu_escape,
    export_graph,
    ingest_to_kuzu,
)
from okf.exports import (
    audit_graph_export,
    build_graph_rag_index,
    build_visual_graph,
)
from okf.evaluate import evaluate_extraction
from okf.pipeline import (
    add_document,
    compute_doc_id,
    finalize_and_build,
    run_pipeline,
)

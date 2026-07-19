#!/usr/bin/env python3
"""
Full Archipelago Pipeline: PDF → OKF v1.5 → Canonicalize → KùzuDB Graph RAG

THIN BACKWARDS-COMPATIBLE SHIM. The implementation now lives in the okf/
package (okf/config.py, okf/extraction.py, okf/cleanup.py, ...); this module
re-exports the full public API so `from okf_pipeline import X` and
`python okf_pipeline.py --add ... --local` keep working unchanged.

Stages:
  1. Section-aware PDF chunking (pdf_ingestion.py)
  2. OKF v1.5 extraction via SLM (expanded schema)
  3. Entity canonicalization (alias resolution)
  4. KùzuDB MERGE ingestion (no duplicate nodes across documents)
"""

import sys
from types import ModuleType

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
    _local_path,
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
    _concept_key,
    _merge_key,
    apply_canonicalization,
    build_canonical_map,
    canonicalize_name,
    is_same_concept_reference,
)
from okf.cleanup import (
    _content_words,
    apply_grounding_filter,
    break_global_cycles,
    cleanup_and_canonicalize,
    dedupe_identical_records,
    is_valid_concept_name,
    merge_duplicate_results,
    prune_invalid_references,
    prune_unresolved_references,
)
from okf import extraction as _extraction_state
from okf import config as _config_state
from okf import pipeline as _pipeline_state
from okf.extraction import (
    _extract_json_payload,
    _generate_local,
    _normalize_related,
    _string_list,
    _strip_json_fences,
    extract_chunks_with_model,
    extract_okf_v15,
    load_local_model,
    normalize_okf_item,
)
from okf.relations import (
    _passage_candidates,
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


class _OkfPipelineShim(ModuleType):
    """Forward the local-model state globals to okf.extraction, which OWNS
    that state (load_local_model mutates it there). A plain `from ... import`
    would snapshot the pre-load None/True forever; forwarding both reads and
    writes keeps `okf_pipeline.LOCAL_MODEL` (and monkeypatching it) live."""

    _LIVE_STATE = ("LOCAL_MODEL", "LOCAL_TOKENIZER", "LOCAL_MODE")
    # Legacy tests and callers patch ``okf_pipeline.extract_okf_v15``.  The
    # implementation lives in ``okf.extraction`` now, and the extraction loop
    # resolves that function there at runtime, so keep that patch seam live.
    _LIVE_FUNCTIONS = ("extract_okf_v15", "load_local_model")

    def __getattr__(self, name):
        if name in _OkfPipelineShim._LIVE_STATE:
            return getattr(_extraction_state, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _OkfPipelineShim._LIVE_STATE:
            setattr(_extraction_state, name, value)
        elif name in _OkfPipelineShim._LIVE_FUNCTIONS:
            setattr(_extraction_state, name, value)
            if name == "load_local_model":
                setattr(_pipeline_state, name, value)
            super().__setattr__(name, value)
        elif name == "BASE_DIR":
            # The package split left orchestration holding its own imported
            # BASE_DIR.  Keep the legacy module's patch point authoritative so
            # embedding callers and tests can safely redirect every artifact
            # (JSON, graph DB and UI export) into a sandbox.
            setattr(_config_state, name, value)
            setattr(_pipeline_state, name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _OkfPipelineShim


if __name__ == "__main__":
    try:
        LOCAL_MODE
    except NameError:
        LOCAL_MODE = _extraction_state.LOCAL_MODE

    args = sys.argv[1:]
    resume_mode = "--resume" in args
    args = [a for a in args if a != "--resume"]

    # --relations-only: second-pass relation extraction over saved results
    # (load okf_results.json -> cleanup/canonicalize -> relation_pass ->
    # save + rebuild graph). No chunk re-extraction.
    relations_only = "--relations-only" in args
    args = [a for a in args if a != "--relations-only"]

    ollama_mode = "--ollama" in args
    args = [a for a in args if a != "--ollama"]
    local_mode = LOCAL_MODE and not ollama_mode

    # --local is accepted explicitly (LOCAL_MODE already defaults it on when a
    # local model folder exists, e.g. lib-qwen / aura-qwen); strip it so it
    # isn't mistaken for a path.
    args = [a for a in args if a != "--local"]

    # Optional uniform page cap: --max-pages N  (applies to every PDF)
    for i, a in enumerate(list(args)):
        if a == "--max-pages" and i + 1 < len(args):
            MAX_PAGES_PER_DOC = int(args[i + 1])
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--max-pages="):
            MAX_PAGES_PER_DOC = int(a.split("=", 1)[1])
            args = [x for x in args if x != a]
            break

    # Propagate the parsed page cap into the package config: run_pipeline and
    # add_document read okf.config.MAX_PAGES_PER_DOC at call time (the same
    # late-bound-global semantics the monolithic module had).
    import okf.config as _okf_config
    _okf_config.MAX_PAGES_PER_DOC = MAX_PAGES_PER_DOC

    # Incremental single-document ingestion: --add <path> [--limit N]
    add_path = None
    for i, a in enumerate(list(args)):
        if a == "--add" and i + 1 < len(args):
            add_path = args[i + 1]
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--add="):
            add_path = a.split("=", 1)[1]
            args = [x for x in args if x != a]
            break

    # --limit N: only process the first N prose chunks (fast CPU testing)
    chunk_limit = None
    for i, a in enumerate(list(args)):
        if a == "--limit" and i + 1 < len(args):
            chunk_limit = int(args[i + 1])
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--limit="):
            chunk_limit = int(a.split("=", 1)[1])
            args = [x for x in args if x != a]
            break

    # --evaluate <gold_path> or --evaluate=<gold_path>
    evaluate_gold_path = None
    for i, a in enumerate(list(args)):
        if a == "--evaluate" and i + 1 < len(args):
            evaluate_gold_path = args[i + 1]
            args = [x for j, x in enumerate(args) if j not in (i, i + 1)]
            break
        if a.startswith("--evaluate="):
            evaluate_gold_path = a.split("=", 1)[1]
            args = [x for x in args if x != a]
            break

    if relations_only:
        run_relations_only()
    elif add_path is not None:
        add_document(add_path, limit=chunk_limit, evaluate_gold_path=evaluate_gold_path)
    elif evaluate_gold_path and not args:
        import os
        import kuzu
        from okf.evaluate import evaluate_pipeline, print_report
        db_path = str(BASE_DIR / "okf_graph.db")
        if os.path.exists(db_path):
            db = kuzu.Database(db_path)
            report = evaluate_pipeline(db, evaluate_gold_path)
            print_report(report)
        else:
            print(f"ERROR: Database not found at {db_path}. Run the pipeline first.")
    else:
        input_path = args[0] if args else None
        run_pipeline(input_path, resume=resume_mode, local=local_mode, evaluate_gold_path=evaluate_gold_path)

"""Shim: okf.evaluate → okf.eval.*"""
from okf.eval.metrics import evaluate_extraction
from okf.eval.gold import (
    load_gold_graph, compare_concepts, compare_edges, evaluate_pipeline, print_report,
    _extract_names, _canonical_key, _extract_name_docs, _extract_edges,
)
from okf.eval.structural import structural_audit

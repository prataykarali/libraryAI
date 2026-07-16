#!/usr/bin/env python
"""Rebuild the merged graph from okf_results.json WITHOUT re-extraction.

Used after fixing the Kuzu string-escaping bug (doubled '' -> \\') so the
final graph is built with the corrected code. Chunk counts are recovered the
same way add_document does for existing entries: distinct (doc_id, chunk_id).
"""
import json
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("okf", "okf_pipeline.py")
okf = importlib.util.module_from_spec(spec)
sys.modules["okf"] = okf
spec.loader.exec_module(okf)

with open("okf_results.json", encoding="utf-8") as f:
    results = json.load(f)

chunk_count = len({(r.get("doc_id", ""), r.get("chunk_id", ""))
                   for r in results if r.get("chunk_id")})
print(f"Rebuilding graph from {len(results)} records / {chunk_count} distinct chunks")
okf.finalize_and_build(results, chunk_count, chunk_count)

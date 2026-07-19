#!/usr/bin/env python3
"""
Subagent worker: processes a batch of chunks and returns validated OKF concepts.
Run via Task tool with skill='research' or directly with LLM access.
"""

import json
import sys
import os

# Add the project root to path
sys.path.insert(0, "/home/pratay-karali/Desktop/libraryAI/libraryAI")

from generate_okf_dataset import (
    build_prompt, validate_concept, normalize_concept, 
    extract_json_array, call_llm
)

def process_batch(batch: list, batch_id: int) -> list:
    """Process a batch of chunks and return validated training pairs."""
    results = []
    
    for i, chunk in enumerate(batch):
        print(f"[Batch {batch_id}] Processing chunk {i+1}/{len(batch)}: {chunk['doc_id']} {chunk['chunk_id']}")
        
        prompt = build_prompt(chunk)
        response = call_llm(prompt)
        
        if not response:
            print(f"  No LLM response")
            continue
            
        concepts = extract_json_array(response)
        if not concepts:
            print(f"  No valid JSON extracted")
            continue
        
        validated = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            c = normalize_concept(c)
            ok, msg = validate_concept(c)
            if ok:
                # Add provenance
                c["doc_id"] = chunk["doc_id"]
                c["chunk_id"] = chunk["chunk_id"]
                c["page_number"] = chunk.get("page_number")
                c["section_title"] = chunk.get("section_title", "")
                validated.append(c)
            else:
                print(f"  Validation failed: {c.get('concept_name', 'unknown')} - {msg}")
        
        if validated:
            # Create training pair
            instruction = build_prompt(chunk).split("TEXT:\n")[0].strip() + "\n\nTEXT:\n" + chunk.get("text", "")[:3000]
            pair = {
                "instruction": instruction,
                "input": "",
                "output": json.dumps(validated, ensure_ascii=False)
            }
            results.append(pair)
            print(f"  OK: {len(validated)} concepts extracted")
        else:
            print(f"  No valid concepts after validation")
    
    return results

if __name__ == "__main__":
    # Read batch from stdin or file
    if len(sys.argv) > 1:
        batch_file = sys.argv[1]
    else:
        batch_file = "/home/pratay-karali/Desktop/libraryAI/libraryAI/batch_input.json"
    
    with open(batch_file) as f:
        batch = json.load(f)
    
    batch_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    results = process_batch(batch, batch_id)
    
    # Output results as JSON
    output_file = f"/home/pratay-karali/Desktop/libraryAI/libraryAI/batch_output_{batch_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Batch {batch_id} complete: {len(results)} training pairs")
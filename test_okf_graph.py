#!/usr/bin/env python3
"""
OKF Relationship Extraction - Extract prerequisites and unlocks across related concepts
Tests if Qwen 3.5 0.8B can identify learning progressions in 10 related facts
"""

import json
from okf_extraction import extract_batch
from mock_data import MOCK_TEXT_CHUNKS

def extract_and_find_relationships():
    """
    Extract OKF from all chunks, then find relationships between them
    """
    print("OKF EXTRACTION WITH RELATIONSHIP MAPPING")
    print("="*70)
    print(f"Processing {len(MOCK_TEXT_CHUNKS)} related concepts...\n")
    
    # Step 1: Extract OKF from all chunks
    text_chunks = [chunk["text"] for chunk in MOCK_TEXT_CHUNKS]
    results = extract_batch(text_chunks)
    
    if not results:
        print("Failed to extract OKF data")
        return
    
    # Step 2: Build a concept name mapping (lowercase for matching)
    concept_names = {r.get('concept_name', '').lower(): r for r in results}
    
    print("\n" + "="*70)
    print("EXTRACTED CONCEPTS")
    print("="*70)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('concept_name', 'Unknown')}")
    
    # Step 3: Find relationships
    print("\n" + "="*70)
    print("LEARNING PROGRESSION MAP")
    print("="*70)
    
    relationships = []
    
    for result in results:
        concept = result.get('concept_name', '')
        unlocks = result.get('unlocks', [])
        
        print(f"\n{concept}")
        print("-" * 70)
        
        # Check each unlock to see if it matches a concept in our batch
        found_unlocks = []
        # Flatten and validate unlocks list
        flat_unlocks = []
        for u in unlocks:
            if isinstance(u, list):
                flat_unlocks.extend([item for item in u if isinstance(item, str)])
            elif isinstance(u, str):
                flat_unlocks.append(u)
        
        for unlock in flat_unlocks:
            unlock_lower = unlock.lower()
            for existing_concept_lower, existing_result in concept_names.items():
                # Check if unlock matches or is similar to existing concept
                if (unlock_lower in existing_concept_lower or 
                    existing_concept_lower in unlock_lower or
                    unlock_lower.split()[0] in existing_concept_lower):
                    
                    existing_concept = existing_result.get('concept_name', '')
                    if existing_concept.lower() != concept.lower():
                        found_unlocks.append(existing_concept)
                        relationships.append({
                            "from": concept,
                            "to": existing_concept,
                            "relation": "unlocks"
                        })
        
        if found_unlocks:
            print(f"Unlocks (FOUND in dataset):")
            for unlock in found_unlocks:
                print(f"  [OK] -> {unlock}")
        
        other_unlocks = [u for u in unlocks if not any(
            u.lower() in cn.lower() or cn.lower() in u.lower() 
            for cn in concept_names.keys()
        )]
        if other_unlocks:
            print(f"Unlocks (external):")
            for unlock in other_unlocks:
                print(f"  -> {unlock}")
    
    # Step 4: Print Learning Graph
    print("\n" + "="*70)
    print("LEARNING PROGRESSION GRAPH")
    print("="*70)
    
    # Topological sort to show learning path
    learned = set()
    remaining = {r.get('concept_name', '') for r in results}
    path = []
    
    for _ in range(len(results)):
        for result in results:
            concept = result.get('concept_name', '')
            if concept in remaining and concept not in learned:
                raw_prereqs = result.get('prerequisites', [])
                prereqs = []
                for p in raw_prereqs:
                    if isinstance(p, list):
                        prereqs.extend([item.lower() for item in p if isinstance(item, str)])
                    elif isinstance(p, str):
                        prereqs.append(p.lower())
                
                # Check if prerequisites are satisfied
                prereq_satisfied = True
                for prereq in prereqs:
                    if any(prereq in cn or cn in prereq for cn in learned):
                        continue
                    elif any(prereq in cn or cn in prereq for cn in concept_names.keys() if cn not in learned):
                        prereq_satisfied = False
                        break
                
                if prereq_satisfied:
                    path.append(concept)
                    learned.add(concept)
                    remaining.discard(concept)
    
    # Add any remaining concepts
    for result in results:
        concept = result.get('concept_name', '')
        if concept not in learned:
            path.append(concept)
    
    print("\nSuggested Learning Order:")
    for i, concept in enumerate(path, 1):
        print(f"{i:2d}. {concept}")
    
    # Step 5: Save full relationship data
    output = {
        "concepts": [{"name": r.get('concept_name', ''), "summary": r.get('summary', '')} for r in results],
        "relationships": relationships,
        "learning_path": path,
        "full_okf_data": results
    }
    
    with open("okf_full_relationships.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Full data saved to: okf_full_relationships.json")
    print("="*70)
    
    return output

if __name__ == "__main__":
    extract_and_find_relationships()

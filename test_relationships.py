#!/usr/bin/env python3
"""
Test OKF extraction to check if prerequisites and unlocks are captured
and if relationships between mock data chunks are identified
"""

import json
from okf_extraction import extract_batch
from mock_data import MOCK_TEXT_CHUNKS

def analyze_relationships(results):
    """
    Analyze relationships between extracted concepts
    Shows prerequisites and unlocks for each concept
    """
    print("\n" + "="*60)
    print("OKF RELATIONSHIP ANALYSIS")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\nConcept {i+1}: {result.get('concept_name', 'Unknown')}")
        print("-" * 60)
        print(f"Summary: {result.get('summary', 'N/A')}")
        
        prerequisites = result.get('prerequisites', [])
        unlocks = result.get('unlocks', [])
        
        print(f"\nPrerequisites ({len(prerequisites)}):")
        if prerequisites:
            for prereq in prerequisites:
                print(f"  - {prereq}")
        else:
            print("  (none)")
        
        print(f"\nUnlocks ({len(unlocks)}):")
        if unlocks:
            for unlock in unlocks:
                print(f"  - {unlock}")
        else:
            print("  (none)")
    
    # Check cross-references
    print("\n" + "="*60)
    print("CROSS-REFERENCES BETWEEN CONCEPTS")
    print("="*60)
    
    if len(results) >= 2:
        concept1_name = results[0].get('concept_name', '').lower()
        concept2_name = results[1].get('concept_name', '').lower()
        
        # Check if concept2 is in concept1's unlocks
        concept1_unlocks = [u.lower() for u in results[0].get('unlocks', [])]
        concept2_prereqs = [p.lower() for p in results[1].get('prerequisites', [])]
        
        print(f"\n1. Does '{results[0]['concept_name']}' unlock '{results[1]['concept_name']}'?")
        related = any(concept2_name in unlock or concept1_name in prereq 
                     for unlock in concept1_unlocks for prereq in concept2_prereqs)
        print(f"   Related: {'YES [OK]' if related else 'NO [X]'}")
        
        print(f"\n2. Prerequisites for '{results[1]['concept_name']}':")
        prereqs = results[1].get('prerequisites', [])
        if prereqs:
            for p in prereqs:
                print(f"   - {p}")
        else:
            print("   (none listed)")
        
        print(f"\n3. Unlocks from '{results[0]['concept_name']}':")
        unlocks = results[0].get('unlocks', [])
        if unlocks:
            for u in unlocks:
                print(f"   - {u}")
        else:
            print("   (none listed)")

def test_mock_data_relationships():
    """Test if mock data chunks identify relationships with each other"""
    print("Testing OKF Extraction with Mock Data Relationships")
    print("="*60)
    print(f"Processing {len(MOCK_TEXT_CHUNKS)} text chunks...\n")
    
    # Extract OKF from all chunks
    text_chunks = [chunk["text"] for chunk in MOCK_TEXT_CHUNKS]
    results = extract_batch(text_chunks)
    
    # Analyze relationships
    if results:
        analyze_relationships(results)
        
        # Save detailed results
        output_file = "mock_relationships.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nDetailed results saved to: {output_file}")
        
        return results
    else:
        print("Failed to extract OKF data")
        return []

if __name__ == "__main__":
    test_mock_data_relationships()

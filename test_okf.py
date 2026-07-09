#!/usr/bin/env python3
"""
Demo script for testing OKF extraction
Run with: python test_okf.py
"""

import json
from okf_extraction import extract_okf, extract_batch
from mock_data import MOCK_TEXT_CHUNKS

def test_single_extraction():
    """Test extraction on a single text chunk"""
    print("TEST 1: Single Text Chunk Extraction")
    print("="*60)
    
    text = MOCK_TEXT_CHUNKS[0]["text"]
    print(f"Input text: {text[:100]}...\n")
    
    okf_data = extract_okf(text)
    
    if okf_data:
        print("\nExtracted OKF Data:")
        print(json.dumps(okf_data, indent=2))
    else:
        print("Failed to extract OKF data")

def test_batch_extraction():
    """Test extraction on multiple text chunks"""
    print("\n\nTEST 2: Batch Extraction")
    print("="*60)
    
    text_chunks = [chunk["text"] for chunk in MOCK_TEXT_CHUNKS]
    results = extract_batch(text_chunks)
    
    print("\n" + "="*60)
    print("FINAL BATCH RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Save results to file
    output_file = "mock_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    test_batch_extraction()
    # Uncomment to test single extraction:
    # test_single_extraction()

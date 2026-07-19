import json
from pathlib import Path
import re

DATA_DIR = Path("/home/pratay-karali/Desktop/libraryAI/libraryAI/training_data")

def analyze(file_path):
    print(f"Analyzing {file_path.name}...")
    total_examples = 0
    total_concepts = 0
    placeholders = 0
    generic_concepts = 0
    
    generic_list = {"machine learning", "optimization", "function", "parameter", "dataset", "distribution", "experiment", "entity", "vector", "value", "system", "method", "problem", "concept", "result", "theory", "process", "variable", "statement"}
    
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            total_examples += 1
            d = json.loads(line)
            out = json.loads(d["output"]) if isinstance(d["output"], str) else d["output"]
            for c in out:
                total_concepts += 1
                name = c.get("concept_name", "")
                summary = c.get("summary", "")
                
                # Check generic
                if name.lower() in generic_list:
                    generic_concepts += 1
                    
                # Check placeholder summary
                if re.search(r"as discussed in the passage|as discussed in the text", summary, re.I):
                    placeholders += 1
                    
    print(f"  Total examples: {total_examples}")
    print(f"  Total concepts: {total_concepts}")
    print(f"  Placeholder summaries: {placeholders} ({placeholders/total_concepts*100:.1f}%)")
    print(f"  Generic concepts: {generic_concepts} ({generic_concepts/total_concepts*100:.1f}%)")
    print()

analyze(DATA_DIR / "okf_train_pairs_v4_2.jsonl")
analyze(DATA_DIR / "okf_test_pairs_v4_2.jsonl")

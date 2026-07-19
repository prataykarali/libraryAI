import json
from pathlib import Path
import re

DATA_DIR = Path("/home/pratay-karali/Desktop/libraryAI/libraryAI/training_data")

GENERIC_CONCEPTS = {
    "machine learning", "optimization", "function", "parameter", "dataset", 
    "distribution", "experiment", "entity", "vector", "value", "system", 
    "method", "problem", "concept", "result", "theory", "process", 
    "variable", "statement", "element", "approach", "prediction", 
    "model", "data", "algorithm", "technique", "task", "object", "property"
}

def extract_text_from_instruction(instruction):
    match = re.search(r"TEXT:\n(.*)\n\nReturn ONLY", instruction, re.S)
    if match:
        return match.group(1).strip()
    # Fallback
    lines = instruction.split("\n")
    text_lines = []
    capture = False
    for line in lines:
        if line.startswith("TEXT:"):
            capture = True
            continue
        if line.startswith("Return ONLY"):
            capture = False
            continue
        if capture:
            text_lines.append(line)
    return "\n".join(text_lines).strip()

def find_real_definition(concept_name, text):
    # Clean text: remove newlines inside words, double spaces, etc.
    text_clean = re.sub(r'\s+', ' ', text)
    sentences = re.split(r"(?<=[.!?])\s+", text_clean)
    
    concept_name_lower = concept_name.lower()
    concept_words = set(w.lower() for w in re.findall(r"\b\w+\b", concept_name_lower))
    
    # 1. Look for sentences containing the exact concept name and a definition verb
    for s in sentences:
        if concept_name_lower in s.lower():
            if any(v in s.lower() for v in [" is ", " refers to ", " defines ", " represents ", " means ", " denotes "]):
                return s.strip()
                
    # 2. Look for any sentence containing the exact concept name
    for s in sentences:
        if concept_name_lower in s.lower():
            return s.strip()
            
    # 3. Look for sentences containing all words of the concept
    for s in sentences:
        s_words = set(w.lower() for w in re.findall(r"\b\w+\b", s))
        if concept_words.issubset(s_words):
            return s.strip()
            
    # 4. Fallback: return a clean placeholder instead of "as discussed in the passage"
    return f"{concept_name} is a key concept discussed in this technical material."

def clean_summary(summary):
    summary = summary.strip()
    summary = re.sub(r'\s+', ' ', summary)
    # Truncate if too long
    if len(summary) > 180:
        summary = summary[:177] + "..."
    # Ensure it ends with a period
    if summary and not summary.endswith(".") and not summary.endswith("..."):
        summary += "."
    return summary

def fix_dataset(file_path):
    print(f"Fixing {file_path.name}...")
    output_lines = []
    
    total_in = 0
    total_out = 0
    fixed_summaries = 0
    removed_concepts = 0
    
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            instruction = item.get("instruction", "")
            text = extract_text_from_instruction(instruction)
            
            output_val = item.get("output", "[]")
            concepts = json.loads(output_val) if isinstance(output_val, str) else output_val
            
            cleaned_concepts = []
            for c in concepts:
                name = c.get("concept_name", "").strip()
                total_in += 1
                
                # Check if generic
                if name.lower() in GENERIC_CONCEPTS:
                    removed_concepts += 1
                    continue
                    
                summary = c.get("summary", "").strip()
                # Check if placeholder summary
                is_placeholder = (
                    not summary or
                    len(summary) < 25 or
                    "as discussed in the passage" in summary.lower() or
                    "as discussed in the text" in summary.lower() or
                    summary.lower() == f"{name.lower()} as discussed."
                )
                
                if is_placeholder:
                    summary = find_real_definition(name, text)
                    fixed_summaries += 1
                    
                c["summary"] = clean_summary(summary)
                cleaned_concepts.append(c)
                
            # Second pass: clean references and self-references
            valid_names = {c["concept_name"].lower() for c in cleaned_concepts}
            for c in cleaned_concepts:
                name_lower = c["concept_name"].lower()
                
                # Clean prerequisites
                prereqs = c.get("prerequisites", [])
                prereqs = [p for p in prereqs if p.lower() != name_lower and p.lower() in valid_names]
                c["prerequisites"] = prereqs
                
                # Clean unlocks
                unlocks = c.get("unlocks", [])
                unlocks = [u for u in unlocks if u.lower() != name_lower and u.lower() in valid_names]
                c["unlocks"] = unlocks
                
                # Clean related_to
                related = c.get("related_to", [])
                cleaned_related = []
                for r in related:
                    target = r.get("concept", "")
                    if target.lower() != name_lower and target.lower() in valid_names:
                        cleaned_related.append(r)
                c["related_to"] = cleaned_related
                
            # Save cleaned concepts back
            item["output"] = json.dumps(cleaned_concepts)
            output_lines.append(json.dumps(item))
            total_out += len(cleaned_concepts)
            
    # Write back
    with open(file_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")
            
    print(f"  Concepts before: {total_in}")
    print(f"  Concepts after:  {total_out} (Removed {removed_concepts} generic concepts)")
    print(f"  Fixed summaries: {fixed_summaries}")
    print()

fix_dataset(DATA_DIR / "okf_train_pairs_v4_2.jsonl")
fix_dataset(DATA_DIR / "okf_test_pairs_v4_2.jsonl")

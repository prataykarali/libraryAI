#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR.parent / "aura-qwen"

def main():
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"Error: Model path does not exist at {MODEL_PATH}")
        sys.exit(1)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Narrative Recipe structured prompt payload
    narrative_recipe = (
        "USER QUERY:\n"
        "Why do I need to normalize a table before creating a B-Tree index?\n\n"
        "STRUCTURED TOPOLOGY (Narrative Recipe):\n"
        "1. Upstream Prerequisites:\n"
        "   - Database Normalization (reduces redundancy, ensures single-source attributes)\n"
        "   - First Normal Form (1NF), Second Normal Form (2NF), Third Normal Form (3NF)\n\n"
        "2. Target Concept:\n"
        "   - B-Tree Index (self-balancing tree that maps search keys to physical storage addresses)\n\n"
        "3. Downstream Applications (Unlocks):\n"
        "   - Query Optimization\n"
        "   - Faster range scans and point queries\n\n"
        "TEXTUAL CITATIONS:\n"
        "- Document: textbooks/Deisenroth_Math_For_ML.pdf, Section: Indexing and Storage, Page: 112\n"
        "  Passage: \"Normalization reduces data redundancy by ensuring that each data attribute is stored in exactly one logical place. When a B-Tree index is constructed, it stores keys mapping directly to unique physical record locations. If the database is not normalized, redundant duplicate rows force the B-Tree index to store multiple duplicate leaf pointers for the same logical attribute, leading to index bloat, slower leaf page traversal, and increased write amplification during updates.\"\n\n"
        "INSTRUCTION:\n"
        "You are the Generator model for the Archipelago knowledge system. Synthesize the user query, structured topology (Prerequisite -> Target -> Unlock), and textual citations into a coherent, fluid, and natural explanation. Do NOT output a JSON list of concepts. Instead, write a conversational and academic response explaining the relationship, using direct references to the concepts and citations above."
    )

    messages = [
        {"role": "user", "content": narrative_recipe}
    ]

    try:
        print("\nFormulating chat template prompt...")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("Running inference (max_new_tokens=512)...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("\n" + "="*80)
        print("MODEL RESPONSE:")
        print("="*80)
        print(response)
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()

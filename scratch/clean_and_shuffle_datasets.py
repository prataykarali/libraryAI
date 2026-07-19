import os
import json
import random
import unicodedata

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove control characters in U+0000 to U+001F except \n (0x0A) and \t (0x09)
    # 0x00 to 0x1F includes \r (0x0D), which will be removed.
    return "".join(c for c in text if not (0x00 <= ord(c) <= 0x1F and ord(c) not in (0x09, 0x0A)))

def clean_obj(obj):
    if isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_obj(item) for item in obj]
    elif isinstance(obj, str):
        return clean_text(obj)
    return obj

def check_for_control_chars(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for c in text:
        if 0x00 <= ord(c) <= 0x1F and ord(c) not in (0x09, 0x0A):
            return True
    return False

def check_obj_for_control_chars(obj) -> bool:
    if isinstance(obj, dict):
        return any(check_obj_for_control_chars(k) or check_obj_for_control_chars(v) for k, v in obj.items())
    elif isinstance(obj, list):
        return any(check_obj_for_control_chars(item) for item in obj)
    elif isinstance(obj, str):
        return check_for_control_chars(obj)
    return False

def main():
    # Set seed for reproducibility
    seed = 42
    rng = random.Random(seed)
    
    # Locate paths relative to script or CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_roots = [
        os.path.dirname(script_dir),                  # If script is inside scratch/ folder
        script_dir,                                   # If running from script directory itself
        os.path.join(os.path.dirname(script_dir), "libraryAI"),  # Root workspace/libraryAI
        os.getcwd(),                                  # Current working directory
    ]
    
    files_to_process = [
        "okf_train_pairs_v3.jsonl",
        "okf_test_pairs_v3.jsonl"
    ]
    
    for filename in files_to_process:
        found_path = None
        for root in possible_roots:
            p = os.path.join(root, "training_data", filename)
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            # Check one more path, assuming libraryAI is in CWD
            p_alt = os.path.join(os.getcwd(), "libraryAI", "training_data", filename)
            if os.path.exists(p_alt):
                found_path = p_alt
        
        if not found_path:
            raise FileNotFoundError(f"Could not find training_data/{filename} in any expected locations.")
            
        print(f"\nProcessing file: {found_path}")
        
        # Load rows
        rows = []
        with open(found_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON at line {i} in {filename}: {e}")
                        raise
                        
        print(f"Loaded {len(rows)} lines from {filename}")
        
        # Pre-clean check
        pre_control_count = sum(1 for r in rows if check_obj_for_control_chars(r))
        print(f"Rows containing forbidden control characters before cleaning: {pre_control_count}")
        
        # Clean rows
        cleaned_rows = [clean_obj(row) for row in rows]
        
        # Post-clean check
        post_control_count = sum(1 for r in cleaned_rows if check_obj_for_control_chars(r))
        print(f"Rows containing forbidden control characters after cleaning: {post_control_count}")
        assert post_control_count == 0, "Error: Control characters were not fully cleaned!"
        
        # Shuffle rows
        rng.shuffle(cleaned_rows)
        
        # Write back to original file
        with open(found_path, "w", encoding="utf-8") as f:
            for row in cleaned_rows:
                # Use ensure_ascii=False to preserve unicode strings nicely
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
        # Verify the saved file is valid JSONL and has no control characters
        verify_rows = []
        with open(found_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        verify_rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Verification Error: Decoded line {i} failed in saved {filename}: {e}")
                        raise
                        
        verify_control_count = sum(1 for r in verify_rows if check_obj_for_control_chars(r))
        print(f"Verification: Loaded {len(verify_rows)} lines from saved {filename}")
        print(f"Verification: Rows with forbidden control characters in saved file: {verify_control_count}")
        assert len(verify_rows) == len(rows), "Verification Error: Row count mismatch!"
        assert verify_control_count == 0, "Verification Error: Found control characters in saved file!"
        print(f"File {filename} successfully cleaned, shuffled, verified, and saved!")

if __name__ == "__main__":
    main()

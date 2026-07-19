import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_server as s

s.init_concepts_data()

print("All concept IDs matching 'agent':")
for cid, info in s.CONCEPTS_DATA.items():
    if "agent" in cid or "agent" in (info.get("label") or "").lower():
        print(f"  ID: {cid} | Label: {info.get('label')}")

print("\nTotal concepts loaded:", len(s.CONCEPTS_DATA))

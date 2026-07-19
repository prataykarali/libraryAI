import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_server as s

s.init_concepts_data()
s.load_embedding_model()

query = "i wanna build an AI agent"

# Check if agentic_system exists in concepts data
print(f"agentic_system in CONCEPTS_DATA: {'agentic_system' in s.CONCEPTS_DATA}")
if 'agentic_system' in s.CONCEPTS_DATA:
    print(f"Details: {s.CONCEPTS_DATA['agentic_system']}")
    
# Rank all concepts and find agentic_system
ranked = s.rank_concepts(query, top_k=500)
found = False
for idx, r in enumerate(ranked):
    if r['id'] == 'agentic_system':
        print(f"\nFound agentic_system at rank {idx+1}:")
        print(f"  Cosine: {r.get('cos'):.4f} | Lexical: {r.get('lexical'):.4f} | Blended: {r.get('blended'):.4f}")
        print(f"  Alias Boost: {r.get('alias_boost'):.4f} | Core Boost: {r.get('core_boost'):.4f}")
        found = True
        break
        
if not found:
    print("\nagentic_system not found in ranked concepts.")

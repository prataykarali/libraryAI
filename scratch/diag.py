import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_server as s

def run_diag(query):
    print("=" * 80)
    print(f"QUERY: '{query}'")
    print("=" * 80)
    
    # 1. Expand query
    expanded = s._expand_query_for_retrieval(query)
    print(f"Expanded Query: {expanded}")
    
    # 2. Check routing
    routing = s.resolve_query_routing(query)
    print(f"Routing Route: {routing['route']}")
    print(f"Routing Score: {routing.get('score')}")
    print(f"Routing Reason: {routing.get('reason')}")
    print(f"Routing Scope: {routing.get('scope')}")
    
    # 3. Check rank concepts
    print("\nTop 5 Concepts:")
    ranked = s.rank_concepts(query, top_k=5)
    for i, r in enumerate(ranked):
        print(f"  {i+1}. ID: {r['id']} | Label: {r['label']} | Cosine: {r.get('cos'):.4f} | Lexical: {r.get('lexical'):.4f} | Blended: {r.get('blended'):.4f}")
        print(f"     Summary: {r.get('summary')}")
        print(f"     Alias Boost: {r.get('alias_boost'):.4f} | Core Boost: {r.get('core_boost'):.4f}")

    # 4. Check find_anchor_concept
    anchor_id, anchor_score = s.find_anchor_concept(query)
    print(f"\nAnchor Concept: {anchor_id} (Score: {anchor_score})")

    # 5. Check if it falls under soft match reject logic
    best = ranked[0] if ranked else None
    best_cos = float(best["cos"]) if best else 0.0
    best_lex = float(best.get("lexical") or 0.0) if best else 0.0
    domain = s._is_learning_or_domain_query(query)
    chitchat = s._is_chitchat(query)
    offtopic = s._is_offtopic(query)
    has_domain = s._has_domain_terms(query)
    learning = s._is_learning_intent(query)
    
    print(f"\nIndicators:")
    print(f"  _is_learning_or_domain_query: {domain}")
    print(f"  _is_chitchat: {chitchat}")
    print(f"  _is_offtopic: {offtopic}")
    print(f"  _has_domain_terms: {has_domain}")
    print(f"  _is_learning_intent: {learning}")

if __name__ == "__main__":
    # Bootstrap concepts data
    s.init_concepts_data()
    # Try loading embedding model
    s.load_embedding_model()
    
    print(f"Settings:")
    print(f"  use_embeddings: {s.use_embeddings}")
    print(f"  SEMANTIC_ANCHOR_THRESHOLD: {s.SEMANTIC_ANCHOR_THRESHOLD}")
    print(f"  REJECT_SIMILARITY_THRESHOLD: {s.REJECT_SIMILARITY_THRESHOLD}")
    print(f"  DOMAIN_SOFT_THRESHOLD: {s.DOMAIN_SOFT_THRESHOLD}")
    print(f"  LEXICAL_ANCHOR_THRESHOLD: {s.LEXICAL_ANCHOR_THRESHOLD}")
    print(f"  Number of concepts loaded: {len(s.CONCEPTS_DATA)}")
    
    run_diag("i wanna build an AI agent")
    run_diag("hi can u suggest me how to learn about various sorts of RAGS")

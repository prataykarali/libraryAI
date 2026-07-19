import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import archipelago.inference.state as st
from archipelago.inference.aliases import generate_aliases
import archipelago.inference.routing as routing

# Mock fake concepts like the unit test does
concepts = {
    "retrieval_augmented_generation": {
        "id": "retrieval_augmented_generation",
        "label": "Retrieval-Augmented Generation",
        "name": "Retrieval-Augmented Generation (RAG)",
        "summary": "Retrieve documents then generate with an LM.",
        "tags": ["rag", "retrieval"],
        "difficulty": "advanced",
        "degree": 12,
    },
    "vector_rag": {
        "id": "vector_rag",
        "label": "Vector RAG",
        "name": "Vector RAG",
        "summary": "RAG over dense vector indexes.",
        "tags": ["rag", "vector"],
        "difficulty": "advanced",
        "degree": 4,
    },
}
for c in concepts.values():
    c["aliases"] = generate_aliases(c)

st.CONCEPTS_DATA = concepts
st.use_embeddings = False  # force lexical path
routing._run_dual_pass_guard = lambda q: False

q = "What is the theory of relativity?"
res = routing.resolve_query_routing(q)
import pprint
pprint.pprint(res)

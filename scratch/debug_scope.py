import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archipelago.inference.scope_gate import check_aiml_scope_via_llm, is_aiml_in_scope

queries = [
    "What were the causes of World War II?",
    "How do plants perform photosynthesis?",
    "What is the capital of France?",
    "Who wrote Shakespeare's plays?",
]

for q in queries:
    llm = check_aiml_scope_via_llm(q)
    in_scope, reason = is_aiml_in_scope(q)
    print(f"Query: '{q}'")
    print(f"  LLM response: {llm}")
    print(f"  In scope: {in_scope} | Reason: {reason}")

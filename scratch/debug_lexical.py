import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archipelago.inference.intent_gate import _score_lexical, _PROTOTYPES
import re

query = "Who wrote Shakespeare's plays?"
q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

stop_words = {"a", "an", "the", "to", "of", "in", "for", "and", "or", "is", "are", "what", "how", "do", "i", "me", "my", "this", "that", "who", "which", "you", "your", "we", "our", "us", "it", "its", "he", "she", "they", "them", "their", "about", "with", "at", "by", "from", "on"}

filtered_q_tokens = q_tokens - stop_words if q_tokens - stop_words else q_tokens
print(f"Query tokens: {q_tokens}")
print(f"Filtered query tokens: {filtered_q_tokens}")

for label, texts in _PROTOTYPES.items():
    print(f"\nCategory: {label}")
    for t in texts:
        t_tokens = set(re.findall(r"[a-z0-9]+", t.lower()))
        filtered_t_tokens = t_tokens - stop_words if t_tokens - stop_words else t_tokens
        inter = filtered_q_tokens & filtered_t_tokens
        union = filtered_q_tokens | filtered_t_tokens
        j = len(inter) / len(union) if union else 0
        distinctive = inter - stop_words
        score = j + 0.08 * len(distinctive)
        if score > 0:
            print(f"  Prototype: '{t}'")
            print(f"    Tokens: {t_tokens}")
            print(f"    Filtered tokens: {filtered_t_tokens}")
            print(f"    Intersection: {inter}")
            print(f"    Jaccard score: {score:.4f}")

import sys
import ollama

queries = [
    "What were the causes of World War II?",
    "How do plants perform photosynthesis?",
    "What is the capital of France?",
    "Who wrote Shakespeare's plays?",
]

_SCOPE_SYSTEM = "Is this query about AI, Machine Learning, or ML math? Answer 'yes' or 'no' only."

client = ollama.Client(host="http://localhost:11434")

for q in queries:
    response = client.chat(
        model="qwen3.5:0.8b",
        messages=[
            {"role": "system", "content": _SCOPE_SYSTEM},
            {"role": "user", "content": q},
        ],
        think=False,
        options={"temperature": 0.0, "num_predict": 8},
    )
    content = (response.get("message") or {}).get("content", "").strip()
    print(f"Query: '{q}' -> Raw response: '{content}'")

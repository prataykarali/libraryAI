import sys
import ollama

queries = [
    "What were the causes of World War II?",
    "How do plants perform photosynthesis?",
    "What is the capital of France?",
    "Who wrote Shakespeare's plays?",
]

prompts = [
    "Is the query about artificial intelligence, machine learning, deep learning, or machine learning mathematics? Answer YES or NO.",
    "Is this query related to AI, Machine Learning, or ML concepts/math? Answer YES or NO only.",
    "Reply YES if this query is about AI, machine learning, or neural networks. Reply NO if it is about history, biology, geography, politics, sports, cooking, or other general knowledge. Answer YES or NO only.",
]

client = ollama.Client(host="http://localhost:11434")

for i, prompt in enumerate(prompts):
    print(f"\n--- Prompt {i+1}: {prompt} ---")
    for q in queries:
        response = client.chat(
            model="qwen3.5:0.8b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": q},
            ],
            think=False,
            options={"temperature": 0.0, "num_predict": 8},
        )
        content = (response.get("message") or {}).get("content", "").strip()
        print(f"Query: '{q}' -> Response: '{content}'")


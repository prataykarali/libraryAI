import ollama
import time

client = ollama.Client(host="http://localhost:11434")

t0 = time.time()
resp = client.chat(
    model="qwen3.5:0.8b",
    messages=[
        {"role": "system", "content": "You are a concise technical writer. Define the concept in 1 sentence under 20 words based on the text. Return only the definition."},
        {"role": "user", "content": "Concept: Bayesian Inference\nText: Bayesian inference is about learning the distribution of random variables. For a dataset X, a parameter prior p(theta), and a likelihood, the posterior is p(theta|X)."}
    ],
    options={"temperature": 0.0, "num_predict": 30}
)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
print(resp.get("message", {}).get("content", ""))

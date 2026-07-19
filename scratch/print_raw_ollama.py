import sys
import ollama
import pprint

client = ollama.Client(host="http://localhost:11434")
try:
    response = client.chat(
        model="qwen3.5:0.8b",
        messages=[
            {"role": "user", "content": "Hello"},
        ],
    )
    print("Response without options:")
    pprint.pprint(response)
except Exception as e:
    print(f"Failed: {e}")

try:
    response2 = client.chat(
        model="qwen3.5:0.8b",
        messages=[
            {"role": "user", "content": "Hello"},
        ],
        think=False,
    )
    print("\nResponse with think=False:")
    pprint.pprint(response2)
except Exception as e:
    print(f"Failed with think=False: {e}")

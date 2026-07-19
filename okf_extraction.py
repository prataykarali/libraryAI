"""OKF extraction using the local Qwen GGUF model in the models folder, with Ollama fallback."""

import json
import os
import re
from pathlib import Path

import ollama

MODEL_NAME = os.environ.get("OKF_MODEL_NAME", "qwen3.5:0.8b")
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "qwen3.5-0.8b.gguf"

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional runtime dependency
    Llama = None

LLAMA_MODEL = None
if MODEL_FILE.exists() and Llama is not None:
    try:
        LLAMA_MODEL = Llama(model_path=str(MODEL_FILE), n_ctx=2048, n_threads=max(1, os.cpu_count() or 1))
    except Exception:  # pragma: no cover - runtime-specific
        LLAMA_MODEL = None

EXTRACTION_PROMPT = """
Extract OKF (Open Knowledge Framework) information from the following text.
Return ONLY valid JSON with these exact fields:
- concept_name: The main concept being taught (string)
- summary: A brief 1-2 sentence summary of the concept (string)
- prerequisites: List of concepts/skills needed BEFORE learning this (list of strings)
- unlocks: List of concepts/topics that can be learned AFTER this (list of strings)

CRITICAL EXTRACTION RULE: Do NOT extract commercial products, companies, cloud providers, software brands, or datasets (e.g., AWS, OpenAI, ChatGPT, HuggingFace, Wikipedia) as Teachable Concepts. Only extract fundamental mathematical, statistical, or theoretical AI/ML concepts that have academic depth.

Text to extract from:
{text}

Return ONLY the JSON object, no other text. Example format:
{{"concept_name": "...", "summary": "...", "prerequisites": [...], "unlocks": [...]}}
"""


def _generate_with_local_model(prompt: str) -> str | None:
    """Generate a response with the local GGUF model when available."""
    if LLAMA_MODEL is None:
        return None

    try:
        response = LLAMA_MODEL(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["</s>"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()
    except Exception as exc:  # pragma: no cover - runtime-specific
        print(f"Local model generation failed: {exc}")
        return None


def extract_okf(text: str) -> dict:
    """Extract OKF data from text using the local Qwen model when available."""
    prompt = EXTRACTION_PROMPT.format(text=text)

    print("Extracting OKF data...\n")

    full_response = ""
    try:
        if LLAMA_MODEL is not None:
            generated = _generate_with_local_model(prompt)
            if generated is not None:
                full_response = generated
            else:
                raise RuntimeError("Local model generation failed")
        else:
            response_stream = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                think=False,
            )
            for chunk in response_stream:
                token = chunk["message"]["content"]
                full_response += token
                print(token, end="", flush=True)
    except Exception as exc:
        print(f"\nError during extraction: {exc}")
        return None

    print("\n")

    try:
        cleaned = full_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]

        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON: {exc}")
        print(f"Raw response: {full_response}")
        return None

def extract_batch(text_chunks: list) -> list:
    """
    Extract OKF data from multiple text chunks
    Returns a list of OKF dictionaries
    """
    results = []
    for i, chunk in enumerate(text_chunks):
        print(f"\n{'='*60}")
        print(f"Processing chunk {i+1}/{len(text_chunks)}")
        print(f"{'='*60}\n")
        
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        okf_data = extract_okf(text)
        
        if okf_data:
            results.append(okf_data)
        else:
            print(f"Failed to extract OKF from chunk {i+1}")
    
    return results

if __name__ == "__main__":
    # Test with mock data
    from mock_data import MOCK_TEXT_CHUNKS
    
    print("OKF Extraction Test")
    print("="*60)
    
    results = extract_batch([chunk["text"] for chunk in MOCK_TEXT_CHUNKS])
    
    print("\n" + "="*60)
    print("FINAL RESULTS (JSON)")
    print("="*60)
    print(json.dumps(results, indent=2))

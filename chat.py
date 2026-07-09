#!/usr/bin/env python3
"""Inference loop for the local Qwen 3.5 0.8B GGUF model in the models folder."""

import os
from pathlib import Path

import ollama

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "qwen3.5-0.8b.gguf"
MODEL_NAME = "qwen3.5:0.8b"
OLLAMA_HOST = "http://localhost:11434"

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional runtime dependency
    Llama = None


LLAMA_MODEL = None
LLAMA_ERROR = None

if MODEL_FILE.exists() and Llama is not None:
    try:
        LLAMA_MODEL = Llama(model_path=str(MODEL_FILE), n_ctx=2048, n_threads=max(1, os.cpu_count() or 1))
    except Exception as exc:  # pragma: no cover - runtime-specific
        LLAMA_MODEL = None
        LLAMA_ERROR = exc


def _generate_with_local_model(prompt: str) -> str | None:
    """Generate text using the local GGUF model if the runtime is available."""
    if LLAMA_MODEL is None:
        return None

    try:
        response = LLAMA_MODEL(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["\nUser:", "</s>"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()
    except Exception as exc:  # pragma: no cover - runtime-specific
        print(f"Local model generation failed: {exc}")
        return None


def check_model():
    """Check whether the local model file or Ollama can be used."""
    if MODEL_FILE.exists():
        if LLAMA_MODEL is not None:
            print(f"✓ Local model ready: {MODEL_FILE}")
            return True
        if LLAMA_ERROR is not None:
            print(f"Local model file found but the runtime could not load it: {LLAMA_ERROR}")

    try:
        print(f"Checking Ollama at {OLLAMA_HOST}...")
        models = ollama.list()
        available_models = [m.model for m in models.models]

        if MODEL_NAME not in available_models:
            print(f"Error: Model '{MODEL_NAME}' not found in Ollama.")
            print(f"Available models: {available_models}")
            print(f"\nTo fix this:")
            print(f"1. Make sure Ollama is installed: https://ollama.ai")
            print(f"2. Pull the model: ollama pull {MODEL_NAME}")
            return False

        print(f"✓ Ollama model '{MODEL_NAME}' is available!")
        return True
    except Exception as exc:
        print(f"Error: Could not connect to Ollama at {OLLAMA_HOST}")
        print(f"Make sure Ollama is running. You can start it from the Ollama app.")
        print(f"Error: {exc}")
        return False


def inference_loop():
    """Run the interactive inference loop."""
    print("\n" + "=" * 60)
    print("Qwen 3.5 0.8B Inference Loop")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            conversation_history.append({"role": "user", "content": user_input})
            print("\nAssistant: ", end="", flush=True)

            full_response = ""
            try:
                if LLAMA_MODEL is not None:
                    prompt = "\n".join(
                        f"{item['role'].capitalize()}: {item['content']}" for item in conversation_history
                    ) + "\nAssistant:"
                    generated = _generate_with_local_model(prompt)
                    if generated is not None:
                        full_response = generated
                    else:
                        raise RuntimeError("Local model generation failed")
                else:
                    response_stream = ollama.chat(
                        model=MODEL_NAME,
                        messages=conversation_history,
                        stream=True,
                        think=False,
                    )
                    for chunk in response_stream:
                        token = chunk["message"]["content"]
                        full_response += token
                        print(token, end="", flush=True)

                if full_response:
                    print("\n")
                    conversation_history.append({"role": "assistant", "content": full_response})
                else:
                    print("\n")
            except Exception as exc:
                print(f"\nError during inference: {exc}")
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as exc:
            print(f"Error: {exc}")


def main():
    """Main entry point."""
    if not check_model():
        return

    inference_loop()


if __name__ == "__main__":
    main()

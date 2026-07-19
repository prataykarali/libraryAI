#!/usr/bin/env python3
"""Inference loop for the local aura-qwen-merged model via Hugging Face transformers."""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "aura-qwen-merged"

MODEL = None
TOKENIZER = None
MODEL_ERROR = None


def load_model():
    """Load the model and tokenizer."""
    global MODEL, TOKENIZER, MODEL_ERROR
    try:
        print(f"Loading model from {MODEL_PATH}...")
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully!")
        return True
    except Exception as exc:
        MODEL_ERROR = exc
        print(f"Error loading model: {exc}")
        return False


def check_model():
    """Check whether the model can be loaded."""
    if MODEL is not None:
        print(f"✓ Model ready: {MODEL_PATH}")
        return True

    if not MODEL_PATH.exists():
        print(f"Error: Model directory not found at {MODEL_PATH}")
        return False

    return load_model()


def inference_loop():
    """Run the interactive inference loop."""
    print("\n" + "=" * 60)
    print("Aura Qwen Merged Inference Loop")
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
                # Apply chat template
                prompt = TOKENIZER.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

                streamer = TextIteratorStreamer(
                    TOKENIZER, skip_prompt=True, skip_special_tokens=True
                )

                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    streamer=streamer,
                )

                thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
                thread.start()

                for token in streamer:
                    full_response += token
                    print(token, end="", flush=True)

                thread.join()

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
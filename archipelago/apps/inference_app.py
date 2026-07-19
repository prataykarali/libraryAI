"""Run inference server: python -m archipelago.apps.inference_app"""
from __future__ import annotations

import os
import threading

from archipelago.inference.bootstrap import app, init_concepts_data
from archipelago.inference.embeddings import load_embedding_model, load_aura_model


def main():
    init_concepts_data()
    threading.Thread(target=load_embedding_model, daemon=True).start()
    if os.getenv("ARCHIPELAGO_LOAD_AURA", "0") == "1":
        threading.Thread(target=load_aura_model, daemon=True).start()
    print("\nArchipelago Inference — http://localhost:5051\n")
    app.run(host=os.environ.get("ARCHIPELAGO_BIND", "127.0.0.1"), port=5051, debug=False)


if __name__ == "__main__":
    main()

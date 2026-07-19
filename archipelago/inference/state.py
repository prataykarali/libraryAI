"""Shared runtime state and config for Archipelago inference."""
from __future__ import annotations

import os
import re
from pathlib import Path

from flask import Flask

# Project root = libraryAI/ (two levels up from this file: inference/ -> archipelago/ -> root)
BASE_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = BASE_DIR / "pdfs"
DATA_FILE = BASE_DIR / "okf_graph.json"
DB_PATH = str(BASE_DIR / "okf_graph.db")

app = Flask(__name__)

# Lazy Kuzu handle — do NOT open the DB at import time.
# Eager open caused exclusive-lock failures during pytest collection and
# concurrent tool/server access.
_db = None
_db_read_only = False


class _LazyDB:
    """Proxy so ``st.db`` still works as a Database-like object."""

    def __getattr__(self, name):
        return getattr(get_db(), name)

    def __repr__(self):
        return f"<LazyKuzuDB path={DB_PATH!r} open={_db is not None}>"


def get_db(*, read_only: bool | None = None):
    """Return the process-wide Kuzu Database, opening on first use.

    Prefer read-write for the inference server. If the exclusive lock is held
    elsewhere, fall back to read-only so chat/routing still works.
    """
    global _db, _db_read_only
    if _db is not None:
        return _db
    import kuzu

    want_ro = bool(read_only) if read_only is not None else False
    # Env escape hatch for test/tools that must not fight the live server
    if os.environ.get("ARCHIPELAGO_DB_READ_ONLY", "").lower() in ("1", "true", "yes"):
        want_ro = True

    try:
        _db = kuzu.Database(DB_PATH, read_only=want_ro)
        _db_read_only = want_ro
        return _db
    except Exception as e:
        if want_ro:
            raise
        # Retry read-only if exclusive lock is held
        msg = str(e).lower()
        if "lock" in msg or "busy" in msg or "could not set lock" in msg:
            print(f"Kuzu exclusive open failed ({e}); falling back to read-only.")
            _db = kuzu.Database(DB_PATH, read_only=True)
            _db_read_only = True
            return _db
        raise


# Public alias used throughout the codebase: st.db
db = _LazyDB()


def reload_db():
    """Re-open the database after a write swap and refresh concept cache."""
    global _db, _db_read_only
    _db = None
    _db_read_only = False
    get_db(read_only=False)
    try:
        from archipelago.inference.routes_chat import init_concepts_data
        init_concepts_data()
    except Exception as e:
        print(f"Error in init_concepts_data during reload: {e}")


EMBED_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
embed_tokenizer = None
embed_model = None
use_embeddings = False

AURA_MODEL_PATH = BASE_DIR.parent / "aura-qwen"
aura_tokenizer = None
aura_model = None
aura_loaded = False

CONCEPT_EMBEDDINGS = {}
CONCEPT_IDS = []
CONCEPT_EMBEDDINGS_TENSOR = None
CONCEPTS_DATA = {}

SEMANTIC_ANCHOR_THRESHOLD = float(os.getenv("ARCHIPELAGO_SEMANTIC_THRESHOLD", "0.52"))
# Soft graph reject only when cos is below this AND lexical/alias surface fit is poor.
REJECT_SIMILARITY_THRESHOLD = float(os.getenv("ARCHIPELAGO_REJECT_THRESHOLD", "0.40"))
# Hard kill-switch: below this cosine (and no strong lexical surface hit) → refuse.
# Raised toward 0.75 so pop-culture false friends need more than weak vector match.
KILL_SWITCH_THRESHOLD = float(os.getenv("ARCHIPELAGO_KILL_SWITCH", "0.75"))
DOMAIN_SOFT_THRESHOLD = float(os.getenv("ARCHIPELAGO_DOMAIN_SOFT_THRESHOLD", "0.28"))
LEXICAL_ANCHOR_THRESHOLD = int(os.getenv("ARCHIPELAGO_LEXICAL_THRESHOLD", "80"))
DEFAULT_OLLAMA_MODEL = os.getenv("ARCHIPELAGO_OLLAMA_MODEL", "qwen3.5:0.8b")
TOP_K_RELATED = int(os.getenv("ARCHIPELAGO_TOP_K_RELATED", "5"))
PDF_BASE_URL = os.getenv("ARCHIPELAGO_PDF_BASE_URL", "http://localhost:5051")
MAX_PREREQS_SHOWN = 3
MAX_UNLOCKS_SHOWN = 3
EVIDENCE_PER_CONCEPT = 1
CITATION_ID_PATTERN = re.compile(r"\[(S\d+):")

_GRAPH_DB = None
_GRAPH_DB_IMPORT_FAILED = False

_DOMAIN_TERMS = (
    "ai", "a.i", "ml", "aiml", "ai/ml", "ai-ml",
    "machine learning", "deep learning", "neural", "model",
    "agent", "agents", "agentic", "llm", "language model", "transformer", "attention",
    "bert", "gpt", "rag", "rags", "retrieval", "lora", "fine-tun", "finetun", "embedding",
    "vector", "knowledge graph", "dataset", "training", "inference", "nlp",
    "computer vision", "comp vision", "cnn", "rnn", "gradient", "optimizer",
    "reinforcement", "supervised", "unsupervised", "peft", "parameter-efficient",
    "pretrain", "pre-train", "token", "encoder", "decoder", "multi-head",
    "self-attention", "vision", "image recognition", "object detection",
    "framework", "frameworks", "pytorch", "tensorflow", "hugging face", "huggingface",
    "langchain", "llamaindex", "openai", "anthropic", "diffusion", "stable diffusion",
    "classification", "regression", "clustering", "generative", "foundation model",
    "representation", "representations", "latent space", "manifold", "generalization",
    "gru", "lstm"
)

_LEARNING_INTENT = (
    "learn", "learning", "teach", "explain", "what is", "what's", "whats",
    "how does", "how do", "how to", "tell me about", "about ", "study",
    "understand", "intro to", "introduction", "prereq", "curriculum",
    "difference between", "vs ", " versus ", "why do we", "when to use",
    "i wanna", "i want to", "can you explain", "help me with",
)

# Light assist only — primary OOD path is intent_gate prototypes, not this list.
_OFFTOPIC_MARKERS = (
    "weather", "temperature outside", "stock price", "recipe", "cook",
    "football", "soccer", "cricket score", "who won", "joke", "horoscope",
    "astronomy", "astrophysics", "constellation", "planet jupiter",
    "britney spears", "celebrity gossip", "super bowl", "world cup",
    "batman", "superman", "harry potter", "sourdough", "calorie", "calories",
    "wine pairing", "my ex", "pyramids", "ancient egypt",
    "relativity", "physics", "chemistry", "biology", "photosynthesis",
    "shakespeare", "world war", "history", "literature"
)

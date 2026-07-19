"""Wire Flask routes onto state.app."""
from archipelago.inference import routes_misc  # noqa: F401
from archipelago.inference import routes_chat  # noqa: F401
from archipelago.inference.state import app
from archipelago.inference.routes_chat import init_concepts_data

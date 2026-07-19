"""
Archipelago Chat UI Server
Serves the premium, standalone chat workspace UI on port 5052.
"""

from flask import Flask, send_from_directory
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "chat_ui"

app = Flask(__name__, static_folder=str(STATIC_DIR))

# CORS configuration
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

@app.route("/")
def index():
    """Serves the main Chat interface index.html"""
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    """Serves any static resources within the chat_ui directory"""
    return send_from_directory(str(STATIC_DIR), filename)

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Archipelago Chat UI Server                      ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Open: http://localhost:5052                      ║")
    print("╚══════════════════════════════════════════════════╝\n")
    app.run(host=os.environ.get("ARCHIPELAGO_BIND", "127.0.0.1"), port=5052, debug=False)

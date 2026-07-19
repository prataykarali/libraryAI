"""Unit tests for archipelago.auth librarian vs student roles."""
from __future__ import annotations

import pytest
from flask import Flask, jsonify

from archipelago.auth import require_token, require_librarian, require_student_or_open


@pytest.fixture()
def client(monkeypatch):
    app = Flask(__name__)

    @app.route("/protected", methods=["POST"])
    @require_token
    def protected():
        return jsonify({"ok": True}), 200

    @app.route("/librarian", methods=["POST"])
    @require_librarian
    def librarian():
        return jsonify({"ok": True, "role": "librarian"}), 200

    @app.route("/chat", methods=["POST"])
    @require_student_or_open
    def chat():
        return jsonify({"ok": True, "role": "student"}), 200

    return app.test_client()


def test_no_token_env_open_access(client, monkeypatch):
    monkeypatch.delenv("ARCHIPELAGO_TOKEN", raising=False)
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}


def test_empty_token_env_open_access(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "")
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected")
    assert resp.status_code == 200


def test_token_set_missing_header_401(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "s3cret")
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


def test_token_set_wrong_token_401(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "s3cret")
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_token_set_bearer_header_200(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "s3cret")
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}


def test_token_set_x_api_token_header_200(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "s3cret")
    monkeypatch.delenv("ARCHIPELAGO_LIBRARIAN_TOKEN", raising=False)
    resp = client.post("/protected", headers={"X-API-Token": "s3cret"})
    assert resp.status_code == 200


def test_librarian_token_preferred(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_TOKEN", "legacy")
    monkeypatch.setenv("ARCHIPELAGO_LIBRARIAN_TOKEN", "lib-secret")
    # Legacy token alone is not enough when librarian token is set
    resp = client.post("/librarian", headers={"Authorization": "Bearer legacy"})
    assert resp.status_code == 401
    resp2 = client.post("/librarian", headers={"Authorization": "Bearer lib-secret"})
    assert resp2.status_code == 200
    resp3 = client.post("/librarian", headers={"X-Librarian-Token": "lib-secret"})
    assert resp3.status_code == 200


def test_student_chat_open_even_with_librarian_token(client, monkeypatch):
    monkeypatch.setenv("ARCHIPELAGO_LIBRARIAN_TOKEN", "lib-secret")
    # Chat must never require librarian credentials
    resp = client.post("/chat")
    assert resp.status_code == 200
    assert resp.get_json()["role"] == "student"


def test_wraps_preserves_function_name():
    @require_token
    def my_view():
        pass

    assert my_view.__name__ == "my_view"

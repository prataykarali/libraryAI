"""Role-based auth for Archipelago APIs.

Roles
-----
* **Student** (chat UI): public read + chat. No upload / delete / manual graph edits.
* **Librarian** (graph UI Librarian tab): mutating corpus ops require a token.

Tokens
------
* ``ARCHIPELAGO_LIBRARIAN_TOKEN`` — preferred for librarian mutations.
* ``ARCHIPELAGO_TOKEN`` — accepted as librarian token if librarian-specific
  env is unset (back-compat with older deploys).

When neither env is set (local dev / unit tests), librarian checks are open so
the test suite keeps working. Production should set a librarian token.
"""
from __future__ import annotations

import functools
import hmac
import os

from flask import jsonify, request


def _extract_token() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):].strip()
    return request.headers.get("X-API-Token") or request.headers.get("X-Librarian-Token")


def librarian_token_expected() -> str:
    """Return the expected librarian secret (empty = auth disabled)."""
    return (
        os.environ.get("ARCHIPELAGO_LIBRARIAN_TOKEN", "").strip()
        or os.environ.get("ARCHIPELAGO_TOKEN", "").strip()
    )


def is_librarian_request() -> bool:
    """True when the request presents a valid librarian token (or auth is off)."""
    expected = librarian_token_expected()
    if not expected:
        return True
    provided = _extract_token()
    if provided is None:
        return False
    return hmac.compare_digest(provided, expected)


def require_token(fn):
    """Legacy alias: same as require_librarian (mutating endpoints only).

    Prefer ``require_librarian`` in new code. Kept so existing imports continue
    to protect ingest/manual routes.
    """
    return require_librarian(fn)


def require_librarian(fn):
    """Librarian-only: upload, delete document, manual concept/edge, cancel jobs.

    Student chat must NOT use this decorator.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        expected = librarian_token_expected()
        if not expected:
            # Open mode for local/dev/tests — still librarian-only in the UI.
            return fn(*args, **kwargs)
        provided = _extract_token()
        if provided is None or not hmac.compare_digest(provided, expected):
            return jsonify({
                "error": "unauthorized",
                "detail": (
                    "Librarian token required. Set Authorization: Bearer <token> "
                    "or X-Librarian-Token. Student chat cannot mutate the corpus."
                ),
            }), 401
        return fn(*args, **kwargs)
    return wrapper


def require_student_or_open(fn):
    """Chat/read endpoints — always open (students). Optional soft token ignored.

    Kept as an explicit marker that this route is student-facing and must never
    be gated by the librarian secret.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

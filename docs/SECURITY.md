# Security Configuration

Archipelago's Flask services (inference on 5051, graph UI on 5050, chat UI on 5052)
are locked down for the departmental pilot via two environment variables.

## Network bind — `ARCHIPELAGO_BIND`

By default every server binds to `127.0.0.1` (localhost only). To expose a server
on the LAN for the pilot, opt in deliberately:

```bash
export ARCHIPELAGO_BIND=0.0.0.0   # or a specific interface IP
.venv/bin/python -m archipelago.apps.inference_app
```

Only set this when token auth (below) is also enabled.

## API token auth — `ARCHIPELAGO_TOKEN`

When `ARCHIPELAGO_TOKEN` is **unset or empty, auth is disabled** — all endpoints
are open. This keeps local development and the test suite working unchanged.

When set, mutating/valuable endpoints require the token on every request:

- `POST /api/chat` (chat / RAG synthesis)
- `POST /api/ingest` (PDF upload)
- `GET|POST /api/ingest/<job_id>/cancel`

Accepted headers (either works):

```
Authorization: Bearer <token>
X-API-Token: <token>
```

Requests without a valid token get `401 {"error": "unauthorized"}`.

Read-only endpoints stay open: static files, `/api/readiness`,
`/api/ingest` (GET list), `/api/ingest/<job_id>` (GET status),
`/api/ingest/capabilities`, and the graph-data GET endpoints on the graph server.

## Enabling auth for the pilot

```bash
export ARCHIPELAGO_TOKEN="$(openssl rand -hex 32)"
export ARCHIPELAGO_BIND=0.0.0.0
.venv/bin/python -m archipelago.apps.inference_app
```

Share the token with pilot users; clients send it as
`Authorization: Bearer <token>`. Rotate by restarting with a new value.

Implementation: `archipelago/auth.py` (`require_token` decorator, applied under
`@app.route` so the route decorator stays outermost).
